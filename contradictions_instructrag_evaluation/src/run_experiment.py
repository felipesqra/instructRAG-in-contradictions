from __future__ import annotations

import argparse
from dataclasses import replace
import sys
from pathlib import Path
from typing import Any

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **kwargs):
        return iterable

from config import EvaluationConfig, load_config
from data_loader import DataLoaderError, load_evaluation_examples
from llm_clients import create_llm_client
from metrics import METRIC_COLUMNS, MetricComputer, aggregate_metrics, paired_comparison, verify_original_metric_methodology
from output_writer import OutputWriter
from rag_runner import run_examples
from rationale_loader import RationaleLoaderError, load_rationales
from utils import ensure_dir, slugify, timestamp_for_path, utc_now_iso


HEAVY_METRICS = {"semantic_cosine", "semantic_dot", "vsim"}

PER_EXAMPLE_FIELDS = [
    "run_id",
    "question_index",
    "question",
    "remedy",
    "retrieval_condition",
    "model_arm",
    "provider",
    "model_name",
    "temperature",
    "max_new_tokens",
    "official_answer",
    "generated_answer",
    "row_has_negative_template",
    "answer_contains_negative_template",
    "rouge1",
    "rouge2",
    "rougeL",
    "semantic_cosine",
    "semantic_dot",
    "vsim",
    "jsd",
    "kld",
    "status",
    "error",
]

AGGREGATE_FIELDS = [
    "model_arm",
    "provider",
    "model_name",
    "retrieval_condition",
    "n_examples",
    *[f"{metric}_mean" for metric in METRIC_COLUMNS],
]

PAIRED_COMPARISON_FIELDS = [
    "retrieval_condition",
    "n_paired_examples",
    *[
        field
        for metric in METRIC_COLUMNS
        for field in (
            f"baseline_{metric}_mean",
            f"instructrag_{metric}_mean",
            f"delta_{metric}",
        )
    ],
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 2 Baseline RAG vs InstructRAG-ICL RAG comparison.")
    parser.add_argument("--config", required=True, help="Path to config/evaluation.yaml")
    parser.add_argument(
        "--validate-data-only",
        action="store_true",
        help="Load original data, Phase 1 rationales, prompts, and metric source checks without calling models.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config = load_config(args.config)
        data_bundle = load_evaluation_examples(
            config.project.contradictions_repo_path,
            config.selection,
            config.retrieval,
        )
        rationale_bundle = load_rationales(config.project.rationales_dir, config.rationales)
        metric_methodology = verify_original_metric_methodology(config.project.contradictions_repo_path)
    except (DataLoaderError, RationaleLoaderError, ValueError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if args.validate_data_only:
        _print_validation(config, data_bundle, rationale_bundle, metric_methodology)
        return 0

    try:
        baseline_client = create_llm_client(config.models.baseline, config.deepseek, config.huggingface, config.groq)
        instruct_client = create_llm_client(config.models.instructrag_icl, config.deepseek, config.huggingface, config.groq)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1

    run_id = f"{timestamp_for_path()}_{slugify(config.project.run_name)}"
    run_dir = ensure_dir(config.project.output_dir / run_id)
    writer = OutputWriter(run_dir)
    writer.copy_config(config.config_path)

    started_at = utc_now_iso()
    artifacts = run_examples(
        run_id=run_id,
        config=config,
        examples=list(tqdm(data_bundle.examples, desc="Generating answers")),
        rationale_bundle=rationale_bundle,
        original_prompt_template=data_bundle.source_summary["original_prompt_template"],
        baseline_client=baseline_client,
        instruct_client=instruct_client,
    )

    if config.outputs.save_raw_generations:
        writer.write_jsonl("raw_generations.jsonl", artifacts.raw_rows)
    if config.outputs.save_prompts:
        writer.write_jsonl("prompts.jsonl", artifacts.prompt_rows)
    if config.outputs.save_documents:
        writer.write_jsonl("documents.jsonl", artifacts.document_rows)
    print(f"Generation artifacts saved to: {run_dir}")

    light_metrics, heavy_metrics = _split_requested_metrics(config.metrics.metrics_to_compute)
    metric_rows: list[dict[str, Any]] = []
    metrics_execution = {
        "requested_metrics": config.metrics.metrics_to_compute,
        "light_metrics": light_metrics,
        "heavy_metrics": heavy_metrics,
        "computed_metrics": [],
        "skipped_metrics": [],
    }

    if not config.metrics.compute_per_example:
        metrics_execution["skipped_metrics"] = config.metrics.metrics_to_compute
        print("Per-example metric computation is disabled in the config.")
    elif not config.metrics.metrics_to_compute:
        print("No metrics were requested in the config.")
    elif light_metrics:
        if _ask_permission(
            "Os JSONL ja foram salvos. Calcular agora as metricas leves "
            f"({', '.join(light_metrics)})?"
        ):
            metric_rows = _compute_metrics(
                run_id,
                config,
                artifacts.raw_rows,
                light_metrics,
                desc="Computing light metrics",
            )
            metrics_execution["computed_metrics"].extend(light_metrics)
            _write_metric_outputs(writer, config, metric_rows)
            print(f"Light metric outputs saved to: {run_dir}")
        else:
            metrics_execution["skipped_metrics"].extend(light_metrics + heavy_metrics)
            print("Metric computation skipped before starting the light metrics.")
    else:
        print("No light metrics were requested in the config.")

    may_offer_heavy_metrics = (
        config.metrics.compute_per_example
        and heavy_metrics
        and (not light_metrics or all(metric in metrics_execution["computed_metrics"] for metric in light_metrics))
    )
    if may_offer_heavy_metrics:
        if _ask_permission(
            "Calcular agora as metricas pesadas "
            f"({', '.join(heavy_metrics)})? Esta etapa pode consumir muita RAM."
        ):
            heavy_metric_rows = _compute_metrics(
                run_id,
                config,
                artifacts.raw_rows,
                heavy_metrics,
                desc="Computing heavy metrics",
            )
            metric_rows = _merge_metric_rows(metric_rows, heavy_metric_rows)
            metrics_execution["computed_metrics"].extend(heavy_metrics)
            _write_metric_outputs(writer, config, metric_rows)
            print(f"Full metric outputs saved to: {run_dir}")
        else:
            metrics_execution["skipped_metrics"].extend(heavy_metrics)
            print("Heavy metrics skipped.")

    writer.write_summary(
        {
            "run_id": run_id,
            "started_at": started_at,
            "finished_at": utc_now_iso(),
            "output_dir": str(run_dir),
            "num_question_remedy_condition_examples": len(data_bundle.examples),
            "num_skipped_selected_examples": len(data_bundle.skipped_examples),
            "skipped_selected_examples": data_bundle.skipped_examples,
            "num_raw_generation_rows": len(artifacts.raw_rows),
            "num_metric_rows": len(metric_rows),
            "metrics_execution": metrics_execution,
            "num_rows_with_negative_template": sum(1 for row in artifacts.raw_rows if row["row_has_negative_template"]),
            "selected_remedies": data_bundle.selected_remedies,
            "phase1_rationale_run_dir": str(rationale_bundle.run_dir),
            "phase1_rationale_file": str(rationale_bundle.rationale_file),
            "data_source_summary": data_bundle.source_summary,
            "metric_methodology": metric_methodology,
            "original_rag_pipeline_reuse": {
                "rag_pipeline_found": data_bundle.source_summary["rag_pipeline"],
                "prompt_template_reused": data_bundle.source_summary["original_prompt_template"],
                "document_order_reuse": "Phase 2 uses ranked documents from the JSONs produced for the original pipeline and applies identical documents to both prompt arms.",
                "original_provider_status": "The cloned rag_pipeline.py exposes placeholders for the model client, so provider=original_rag_pipeline raises a clear error unless manually adapted.",
            },
        }
    )
    print(f"Phase 2 run saved to: {run_dir}")
    return 0


def _ask_permission(question: str) -> bool:
    try:
        answer = input(f"{question} [s/N]: ").strip().lower()
    except EOFError:
        print("No interactive input was available; treating as 'no'.")
        return False
    return answer in {"s", "sim", "y", "yes"}


def _split_requested_metrics(metrics_to_compute: list[str]) -> tuple[list[str], list[str]]:
    light_metrics: list[str] = []
    heavy_metrics: list[str] = []
    seen: set[str] = set()
    for metric in metrics_to_compute:
        if metric in seen:
            continue
        seen.add(metric)
        if metric in HEAVY_METRICS:
            heavy_metrics.append(metric)
        else:
            light_metrics.append(metric)
    return light_metrics, heavy_metrics


def _write_metric_outputs(
    writer: OutputWriter,
    config: EvaluationConfig,
    metric_rows: list[dict[str, Any]],
) -> None:
    metric_rows_without_negative = [row for row in metric_rows if not row["row_has_negative_template"]]
    aggregate_with_negative = aggregate_metrics(metric_rows)
    aggregate_without_negative = aggregate_metrics(metric_rows_without_negative)
    paired_with_negative = paired_comparison(metric_rows)
    paired_without_negative = paired_comparison(metric_rows_without_negative)

    if config.outputs.save_per_example_metrics:
        writer.write_csv("per_example_metrics_all_rows.csv", metric_rows, PER_EXAMPLE_FIELDS)
        writer.write_csv("per_example_metrics_without_negative_rows.csv", metric_rows_without_negative, PER_EXAMPLE_FIELDS)
    if config.outputs.save_aggregated_metrics:
        writer.write_csv("aggregate_metrics_with_negative_rows.csv", aggregate_with_negative, AGGREGATE_FIELDS)
        writer.write_csv("aggregate_metrics_without_negative_rows.csv", aggregate_without_negative, AGGREGATE_FIELDS)
    if config.outputs.save_comparison_tables:
        writer.write_csv("paired_comparison_with_negative_rows.csv", paired_with_negative, PAIRED_COMPARISON_FIELDS)
        writer.write_csv("paired_comparison_without_negative_rows.csv", paired_without_negative, PAIRED_COMPARISON_FIELDS)


def _merge_metric_rows(
    base_rows: list[dict[str, Any]],
    extra_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not base_rows:
        return extra_rows

    merged_rows = [dict(row) for row in base_rows]
    by_key = {_metric_row_key(row): row for row in merged_rows}
    for extra_row in extra_rows:
        key = _metric_row_key(extra_row)
        target = by_key.get(key)
        if target is None:
            target = dict(extra_row)
            by_key[key] = target
            merged_rows.append(target)
            continue

        for metric in METRIC_COLUMNS:
            if extra_row.get(metric) is not None:
                target[metric] = extra_row[metric]
        if target.get("status") == "success" and extra_row.get("status") != "success":
            target["status"] = extra_row.get("status")
            target["error"] = extra_row.get("error")
    return merged_rows


def _metric_row_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row["question_index"],
        row["question"],
        row["remedy"],
        row["retrieval_condition"],
        row["model_arm"],
    )


def _print_validation(config, data_bundle, rationale_bundle, metric_methodology) -> None:
    print("Phase 2 data validation succeeded.")
    print(f"Contradictions repo: {config.project.contradictions_repo_path}")
    print(f"Original rag_pipeline.py: {data_bundle.source_summary['rag_pipeline']}")
    print(f"Selected remedies: {', '.join(data_bundle.selected_remedies)}")
    print(f"Selected question/remedy/condition examples: {len(data_bundle.examples)}")
    print(f"Skipped selected examples: {len(data_bundle.skipped_examples)}")
    print(f"Phase 1 rationale run: {rationale_bundle.run_dir}")
    print(f"Phase 1 rationales loaded: {len(rationale_bundle.examples)}")
    print(f"Metric methodology source: {metric_methodology['source']}")
    first = data_bundle.examples[0]
    print("First target example:")
    print(f"  question_index: {first.question_index}")
    print(f"  remedy: {first.remedy}")
    print(f"  retrieval_condition: {first.retrieval_condition}")
    print(f"  documents_used: {len(first.documents)}")


def _compute_metrics(
    run_id: str,
    config: EvaluationConfig,
    raw_rows: list[dict[str, Any]],
    metrics_to_compute: list[str],
    desc: str,
) -> list[dict[str, Any]]:
    if not config.metrics.compute_per_example:
        return []
    metrics_config = replace(config.metrics, metrics_to_compute=metrics_to_compute)
    computer = MetricComputer(config.project.contradictions_repo_path, metrics_config)
    rows = []
    for raw in tqdm(raw_rows, desc=desc):
        rows.append(
            _metric_row(
                run_id,
                raw,
                "baseline_rag",
                config.models.baseline.provider,
                config.models.baseline.model_name,
                config.models.baseline.temperature,
                config.models.baseline.max_new_tokens,
                raw.get("baseline_answer", ""),
                raw.get("baseline_contains_negative_template", False),
                computer,
            )
        )
        rows.append(
            _metric_row(
                run_id,
                raw,
                "instructrag_icl_rag",
                config.models.instructrag_icl.provider,
                config.models.instructrag_icl.model_name,
                config.models.instructrag_icl.temperature,
                config.models.instructrag_icl.max_new_tokens,
                raw.get("instructrag_answer", ""),
                raw.get("instructrag_contains_negative_template", False),
                computer,
            )
        )
    return rows


def _metric_row(
    run_id: str,
    raw: dict[str, Any],
    model_arm: str,
    provider: str,
    model_name: str,
    temperature: float,
    max_new_tokens: int,
    answer: str,
    answer_negative: bool,
    computer: MetricComputer,
) -> dict[str, Any]:
    metrics = {metric: None for metric in METRIC_COLUMNS}
    status = raw.get("status", "success")
    error = raw.get("error")
    if status == "success":
        try:
            metrics = computer.compute(raw.get("official_answer", ""), answer)
        except Exception as exc:
            status = "metric_error"
            error = f"{type(exc).__name__}: {exc}"
    return {
        "run_id": run_id,
        "question_index": raw["question_index"],
        "question": raw["question"],
        "remedy": raw["remedy"],
        "retrieval_condition": raw["retrieval_condition"],
        "model_arm": model_arm,
        "provider": provider,
        "model_name": model_name,
        "temperature": temperature,
        "max_new_tokens": max_new_tokens,
        "official_answer": raw["official_answer"],
        "generated_answer": answer,
        "row_has_negative_template": raw["row_has_negative_template"],
        "answer_contains_negative_template": answer_negative,
        **metrics,
        "status": status,
        "error": error,
    }


if __name__ == "__main__":
    raise SystemExit(main())
