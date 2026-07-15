#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

try:
    import numpy as np
    import pandas as pd
except ImportError as exc:  # pragma: no cover - exercised only in missing envs.
    raise SystemExit(
        "Missing analysis dependencies. Install this directory's requirements.txt "
        "or run with the experiment venv after installing pandas/numpy/scipy/matplotlib."
    ) from exc

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

try:
    from scipy import stats
except ImportError:  # pragma: no cover
    stats = None


METRICS = [
    "rouge1",
    "rouge2",
    "rougeL",
    "semantic_cosine",
    "semantic_dot",
    "vsim",
    "jsd",
    "kld",
]

HIGHER_IS_BETTER = {
    "rouge1": True,
    "rouge2": True,
    "rougeL": True,
    "semantic_cosine": True,
    "semantic_dot": True,
    "vsim": True,
    "jsd": False,
    "kld": False,
}

METRIC_LABELS = {
    "rouge1": "ROUGE-1",
    "rouge2": "ROUGE-2",
    "rougeL": "ROUGE-L",
    "semantic_cosine": "Semantic cosine",
    "semantic_dot": "Semantic dot",
    "vsim": "VSIM",
    "jsd": "JSD",
    "kld": "KLD",
}

ARTICLE_METRICS = [
    "rouge1",
    "rouge2",
    "rougeL",
    "semantic_cosine",
    "vsim",
    "jsd",
    "kld",
]

CONDITION_ORDER = ["most_similar", "most_contradictory", "least_contradictory"]
ARM_ORDER = ["baseline_rag", "instructrag_icl_rag"]
BOOTSTRAP_ITERATIONS = 10000
BOOTSTRAP_SEED = 20260706

EXPECTED_FILES = [
    "aggregate_metrics_with_negative_rows.csv",
    "aggregate_metrics_without_negative_rows.csv",
    "per_example_metrics_all_rows.csv",
    "per_example_metrics_without_negative_rows.csv",
    "paired_comparison_with_negative_rows.csv",
    "paired_comparison_without_negative_rows.csv",
    "raw_generations.jsonl",
    "run_summary.json",
    "config_used.yaml",
]


@dataclass
class RunData:
    run_dir: Path
    files_used: dict[str, str]
    missing_files: list[str]
    aggregate_with_negative: pd.DataFrame
    aggregate_without_negative: pd.DataFrame
    per_example_all: pd.DataFrame
    per_example_without_negative: pd.DataFrame
    paired_with_negative: pd.DataFrame
    paired_without_negative: pd.DataFrame
    raw_generations: pd.DataFrame
    run_summary: dict[str, Any]
    config: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze Phase 2 baseline RAG vs InstructRAG-ICL results."
    )
    parser.add_argument("--run-dir", type=Path, default=None, help="Path to a Phase 2 run directory.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Base directory where a timestamped analysis folder will be created.",
    )
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance level after FDR correction.")
    return parser.parse_args()


def workspace_root() -> Path:
    return Path(__file__).resolve().parent.parent


def default_phase2_outputs_dir() -> Path:
    return workspace_root() / "contradictions_instructrag_evaluation" / "outputs"


def find_latest_run() -> Path:
    outputs_dir = default_phase2_outputs_dir()
    if not outputs_dir.exists():
        raise FileNotFoundError(f"Phase 2 outputs directory not found: {outputs_dir}")

    candidates = sorted(outputs_dir.glob("*_phase2_rag_comparison"), reverse=True)
    complete = []
    partial = []
    for path in candidates:
        if not path.is_dir():
            continue
        has_core = (
            (path / "run_summary.json").exists()
            and (path / "per_example_metrics_all_rows.csv").exists()
            and (path / "aggregate_metrics_with_negative_rows.csv").exists()
        )
        if has_core:
            complete.append(path)
        else:
            partial.append(path)

    if complete:
        return complete[0]
    if partial:
        return partial[0]
    raise FileNotFoundError(f"No Phase 2 run directories found inside: {outputs_dir}")


def make_timestamped_output_dir(base_dir: Path | None) -> Path:
    if base_dir is None:
        base_dir = Path(__file__).resolve().parent / "analysis_outputs"
    if not base_dir.is_absolute():
        base_dir = (Path.cwd() / base_dir).resolve()
    base_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    out_dir = base_dir / f"{stamp}_phase2_results_analysis"
    suffix = 1
    while out_dir.exists():
        out_dir = base_dir / f"{stamp}_phase2_results_analysis_{suffix:02d}"
        suffix += 1
    out_dir.mkdir(parents=True)
    (out_dir / "figures").mkdir()
    return out_dir


def read_json(path: Path, warnings: list[str]) -> dict[str, Any]:
    if not path.exists():
        warnings.append(f"Missing JSON file: {path.name}")
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_yaml(path: Path, warnings: list[str]) -> dict[str, Any]:
    if not path.exists():
        warnings.append(f"Missing YAML file: {path.name}")
        return {}
    if yaml is None:
        warnings.append("PyYAML is not installed; config_used.yaml could not be parsed.")
        return {}
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    return loaded if isinstance(loaded, dict) else {}


def read_csv_optional(path: Path, warnings: list[str]) -> pd.DataFrame:
    if not path.exists():
        warnings.append(f"Missing CSV file: {path.name}")
        return pd.DataFrame()
    return pd.read_csv(path)


def read_jsonl_optional(path: Path, warnings: list[str]) -> pd.DataFrame:
    if not path.exists():
        warnings.append(f"Missing JSONL file: {path.name}")
        return pd.DataFrame()
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def normalize_bool_series(series: pd.Series) -> pd.Series:
    if series.empty:
        return series
    if series.dtype == bool:
        return series
    return series.map(
        lambda value: (
            True
            if str(value).strip().lower() == "true"
            else False
            if str(value).strip().lower() == "false"
            else np.nan
        )
    )


def normalize_metrics(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    for metric in METRICS:
        if metric in df.columns:
            df[metric] = pd.to_numeric(df[metric], errors="coerce")
        mean_col = f"{metric}_mean"
        if mean_col in df.columns:
            df[mean_col] = pd.to_numeric(df[mean_col], errors="coerce")
    for col in ["row_has_negative_template", "answer_contains_negative_template"]:
        if col in df.columns:
            df[col] = normalize_bool_series(df[col])
    return df


def load_run(run_dir: Path) -> RunData:
    run_dir = run_dir.resolve()
    warnings: list[str] = []
    files_used: dict[str, str] = {}
    missing_files = [name for name in EXPECTED_FILES if not (run_dir / name).exists()]
    for name in EXPECTED_FILES:
        path = run_dir / name
        if path.exists():
            files_used[name] = str(path)

    aggregate_with_negative = normalize_metrics(
        read_csv_optional(run_dir / "aggregate_metrics_with_negative_rows.csv", warnings)
    )
    aggregate_without_negative = normalize_metrics(
        read_csv_optional(run_dir / "aggregate_metrics_without_negative_rows.csv", warnings)
    )
    per_example_all = normalize_metrics(
        read_csv_optional(run_dir / "per_example_metrics_all_rows.csv", warnings)
    )
    per_example_without_negative = normalize_metrics(
        read_csv_optional(run_dir / "per_example_metrics_without_negative_rows.csv", warnings)
    )
    paired_with_negative = normalize_metrics(
        read_csv_optional(run_dir / "paired_comparison_with_negative_rows.csv", warnings)
    )
    paired_without_negative = normalize_metrics(
        read_csv_optional(run_dir / "paired_comparison_without_negative_rows.csv", warnings)
    )
    raw_generations = read_jsonl_optional(run_dir / "raw_generations.jsonl", warnings)
    for col in [
        "row_has_negative_template",
        "baseline_contains_negative_template",
        "instructrag_contains_negative_template",
    ]:
        if col in raw_generations.columns:
            raw_generations[col] = normalize_bool_series(raw_generations[col])
    run_summary = read_json(run_dir / "run_summary.json", warnings)
    config = read_yaml(run_dir / "config_used.yaml", warnings)

    missing_files = sorted(set(missing_files + [warning.split(": ", 1)[1] for warning in warnings if warning.startswith("Missing ")]))
    return RunData(
        run_dir=run_dir,
        files_used=files_used,
        missing_files=missing_files,
        aggregate_with_negative=aggregate_with_negative,
        aggregate_without_negative=aggregate_without_negative,
        per_example_all=per_example_all,
        per_example_without_negative=per_example_without_negative,
        paired_with_negative=paired_with_negative,
        paired_without_negative=paired_without_negative,
        raw_generations=raw_generations,
        run_summary=run_summary,
        config=config,
    )


def sorted_conditions(values: Iterable[Any]) -> list[str]:
    seen = [str(value) for value in values if pd.notna(value)]
    known = [condition for condition in CONDITION_ORDER if condition in seen]
    extra = sorted(condition for condition in set(seen) if condition not in CONDITION_ORDER)
    return known + extra


def sorted_arms(values: Iterable[Any]) -> list[str]:
    seen = [str(value) for value in values if pd.notna(value)]
    known = [arm for arm in ARM_ORDER if arm in seen]
    extra = sorted(arm for arm in set(seen) if arm not in ARM_ORDER)
    return known + extra


def value_or_none(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if pd.isna(value):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_ready(v) for v in value]
    if isinstance(value, tuple):
        return [json_ready(v) for v in value]
    return value_or_none(value)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(json_ready(data), ensure_ascii=False, indent=2), encoding="utf-8")


def round_for_article(value: Any, digits: int = 4) -> Any:
    if value is None or pd.isna(value):
        return ""
    return round(float(value), digits)


def planned_question_remedy_pairs(config: dict[str, Any], summary: dict[str, Any]) -> int | None:
    selection = config.get("selection", {}) if isinstance(config.get("selection"), dict) else {}
    num_questions = selection.get("num_questions")
    num_remedies = selection.get("num_remedies")
    if isinstance(num_questions, int) and isinstance(num_remedies, int):
        return num_questions * num_remedies
    if summary.get("num_question_remedy_condition_examples") and selection.get("retrieval_conditions"):
        try:
            return int(summary["num_question_remedy_condition_examples"]) // len(selection["retrieval_conditions"])
        except Exception:
            return None
    return None


def unique_question_remedy_count(df: pd.DataFrame) -> int | None:
    cols = [col for col in ["question_index", "question", "remedy"] if col in df.columns]
    if not cols or df.empty:
        return None
    return int(df[cols].drop_duplicates().shape[0])


def create_sample_tables(data: RunData, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = data.raw_generations
    per_all = data.per_example_all

    if not raw.empty:
        total_generation_rows = int(len(raw))
        effective_pairs = unique_question_remedy_count(raw)
        retrieval_conditions = len(sorted_conditions(raw["retrieval_condition"])) if "retrieval_condition" in raw else None
    else:
        total_generation_rows = int(data.run_summary.get("num_raw_generation_rows", 0) or 0)
        effective_pairs = unique_question_remedy_count(per_all)
        retrieval_conditions = (
            len(sorted_conditions(per_all["retrieval_condition"])) if "retrieval_condition" in per_all else None
        )

    model_arms = len(sorted_arms(per_all["model_arm"])) if "model_arm" in per_all else None
    total_response_rows = int(len(per_all)) if not per_all.empty else int(data.run_summary.get("num_metric_rows", 0) or 0)

    sample_summary = pd.DataFrame(
        [
            {
                "planned_question_remedy_pairs": planned_question_remedy_pairs(data.config, data.run_summary),
                "effective_question_remedy_pairs": effective_pairs,
                "retrieval_conditions": retrieval_conditions,
                "model_arms": model_arms,
                "total_generation_rows": total_generation_rows,
                "total_response_rows": total_response_rows,
            }
        ]
    )
    sample_summary.to_csv(out_dir / "sample_summary.csv", index=False)

    if per_all.empty:
        sample_by_condition_and_arm = pd.DataFrame()
    else:
        df = per_all.copy()
        group_cols = ["retrieval_condition", "model_arm"]
        sample_by_condition_and_arm = (
            df.groupby(group_cols, dropna=False)
            .agg(
                response_rows=("model_arm", "size"),
                unique_question_remedy_pairs=("remedy", "count"),
                numeric_metric_rows=("rouge1", lambda s: int(pd.to_numeric(s, errors="coerce").notna().sum())),
            )
            .reset_index()
        )
        key_cols = [col for col in ["question_index", "question", "remedy"] if col in df.columns]
        if key_cols:
            unique_counts = (
                df.groupby(group_cols, dropna=False)[key_cols]
                .apply(lambda x: x.drop_duplicates().shape[0])
                .reset_index(name="unique_question_remedy_pairs")
            )
            sample_by_condition_and_arm = sample_by_condition_and_arm.drop(
                columns=["unique_question_remedy_pairs"]
            ).merge(unique_counts, on=group_cols, how="left")
        sample_by_condition_and_arm = order_condition_arm(sample_by_condition_and_arm)
    sample_by_condition_and_arm.to_csv(out_dir / "sample_by_condition_and_arm.csv", index=False)
    return sample_summary, sample_by_condition_and_arm


def order_condition_arm(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    if "retrieval_condition" in df.columns:
        condition_rank = {condition: i for i, condition in enumerate(CONDITION_ORDER)}
        df["_condition_rank"] = df["retrieval_condition"].map(condition_rank).fillna(999)
    if "model_arm" in df.columns:
        arm_rank = {arm: i for i, arm in enumerate(ARM_ORDER)}
        df["_arm_rank"] = df["model_arm"].map(arm_rank).fillna(999)
    sort_cols = [col for col in ["_condition_rank", "_arm_rank", "retrieval_condition", "model_arm"] if col in df.columns]
    df = df.sort_values(sort_cols).drop(columns=[col for col in ["_condition_rank", "_arm_rank"] if col in df.columns])
    return df.reset_index(drop=True)


def create_aggregate_tables(
    aggregate: pd.DataFrame, out_dir: Path, suffix: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if aggregate.empty:
        empty = pd.DataFrame()
        empty.to_csv(out_dir / f"aggregate_table_{suffix}.csv", index=False)
        empty.to_csv(out_dir / f"article_table_main_metrics_{suffix}.csv", index=False)
        return empty, empty

    columns = [
        "model_arm",
        "retrieval_condition",
        "n_examples",
        *[f"{metric}_mean" for metric in METRICS],
    ]
    existing = [col for col in columns if col in aggregate.columns]
    table = order_condition_arm(aggregate[existing])
    table.to_csv(out_dir / f"aggregate_table_{suffix}.csv", index=False)

    article_rows = []
    for _, row in table.iterrows():
        article_rows.append(
            {
                "Abordagem": row.get("model_arm"),
                "Condição": row.get("retrieval_condition"),
                "ROUGE-1": round_for_article(row.get("rouge1_mean")),
                "ROUGE-2": round_for_article(row.get("rouge2_mean")),
                "ROUGE-L": round_for_article(row.get("rougeL_mean")),
                "Semantic cosine": round_for_article(row.get("semantic_cosine_mean")),
                "VSIM": round_for_article(row.get("vsim_mean")),
                "JSD": round_for_article(row.get("jsd_mean")),
                "KLD": round_for_article(row.get("kld_mean")),
            }
        )
    article = pd.DataFrame(article_rows)
    article.to_csv(out_dir / f"article_table_main_metrics_{suffix}.csv", index=False)
    return table, article


def aggregate_value(aggregate: pd.DataFrame, arm: str, condition: str, metric: str) -> float | None:
    if aggregate.empty:
        return None
    subset = aggregate[
        (aggregate.get("model_arm") == arm)
        & (aggregate.get("retrieval_condition") == condition)
    ]
    col = f"{metric}_mean"
    if subset.empty or col not in subset.columns:
        return None
    value = subset.iloc[0][col]
    if pd.isna(value):
        return None
    return float(value)


def create_rq1_deltas(aggregate: pd.DataFrame, out_dir: Path, suffix: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = []
    conditions = sorted_conditions(aggregate["retrieval_condition"]) if not aggregate.empty and "retrieval_condition" in aggregate else []
    for condition in conditions:
        for metric in METRICS:
            baseline = aggregate_value(aggregate, "baseline_rag", condition, metric)
            instruct = aggregate_value(aggregate, "instructrag_icl_rag", condition, metric)
            if baseline is None or instruct is None:
                delta = None
                better = "missing"
            elif HIGHER_IS_BETTER[metric]:
                delta = instruct - baseline
                better = "instructrag_icl_rag" if delta > 0 else "baseline_rag" if delta < 0 else "tie"
            else:
                delta = baseline - instruct
                better = "instructrag_icl_rag" if delta > 0 else "baseline_rag" if delta < 0 else "tie"
            rows.append(
                {
                    "retrieval_condition": condition,
                    "metric": metric,
                    "baseline_mean": baseline,
                    "instructrag_mean": instruct,
                    "delta_instructrag_advantage": delta,
                    "better_method": better,
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / f"rq1_deltas_by_condition_{suffix}.csv", index=False)
    if suffix in {"with_negative_rows", "all_rows"}:
        df.to_csv(out_dir / "rq1_deltas_by_condition.csv", index=False)

    valid = df[df["better_method"].isin(["instructrag_icl_rag", "baseline_rag", "tie"])] if not df.empty else df
    summary = {
        "analysis_subset": suffix,
        "num_comparisons": int(len(valid)),
        "instructrag_better": int((valid["better_method"] == "instructrag_icl_rag").sum()) if not valid.empty else 0,
        "baseline_better": int((valid["better_method"] == "baseline_rag").sum()) if not valid.empty else 0,
        "ties": int((valid["better_method"] == "tie").sum()) if not valid.empty else 0,
        "percent_favoring_instructrag": (
            float((valid["better_method"] == "instructrag_icl_rag").mean() * 100) if not valid.empty else None
        ),
    }
    return df, summary


def degradation_from_values(most_similar: float, most_contradictory: float, metric: str) -> float:
    if HIGHER_IS_BETTER[metric]:
        return most_similar - most_contradictory
    return most_contradictory - most_similar


def interpret_rq2_row(baseline_degradation: float | None, instruct_degradation: float | None, reduction: float | None) -> str:
    if baseline_degradation is None or instruct_degradation is None or reduction is None:
        return "Resultado inconclusivo/empate"
    if baseline_degradation < 0 and instruct_degradation < 0:
        return "Não houve degradação; houve melhora em most_contradictory"
    if reduction > 0:
        return "InstructRAG-ICL apresentou menor degradação"
    if reduction < 0:
        return "Baseline apresentou menor degradação"
    return "Resultado inconclusivo/empate"


def create_rq2_degradation(aggregate: pd.DataFrame, out_dir: Path, suffix: str) -> pd.DataFrame:
    rows = []
    for metric in METRICS:
        baseline_sim = aggregate_value(aggregate, "baseline_rag", "most_similar", metric)
        baseline_con = aggregate_value(aggregate, "baseline_rag", "most_contradictory", metric)
        instruct_sim = aggregate_value(aggregate, "instructrag_icl_rag", "most_similar", metric)
        instruct_con = aggregate_value(aggregate, "instructrag_icl_rag", "most_contradictory", metric)
        if None in [baseline_sim, baseline_con, instruct_sim, instruct_con]:
            baseline_deg = None
            instruct_deg = None
            reduction = None
        else:
            baseline_deg = degradation_from_values(baseline_sim, baseline_con, metric)
            instruct_deg = degradation_from_values(instruct_sim, instruct_con, metric)
            reduction = baseline_deg - instruct_deg
        rows.append(
            {
                "metric": metric,
                "baseline_most_similar": baseline_sim,
                "baseline_most_contradictory": baseline_con,
                "baseline_degradation": baseline_deg,
                "instructrag_most_similar": instruct_sim,
                "instructrag_most_contradictory": instruct_con,
                "instructrag_degradation": instruct_deg,
                "degradation_reduction": reduction,
                "interpretation": interpret_rq2_row(baseline_deg, instruct_deg, reduction),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / f"rq2_degradation_most_similar_to_most_contradictory_{suffix}.csv", index=False)
    if suffix in {"with_negative_rows", "all_rows"}:
        df.to_csv(out_dir / "rq2_degradation_most_similar_to_most_contradictory.csv", index=False)
    return df


def rankdata_fallback(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty(len(values), dtype=float)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[order[j]] == values[order[i]]:
            j += 1
        rank = (i + 1 + j) / 2.0
        ranks[order[i:j]] = rank
        i = j
    return ranks


def wilcoxon_signed_rank(values: np.ndarray) -> tuple[float | None, float | None]:
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return None, None
    nonzero = values[values != 0]
    if len(nonzero) == 0:
        return 0.0, 1.0
    if stats is not None:
        try:
            result = stats.wilcoxon(nonzero, zero_method="wilcox", alternative="two-sided", method="auto")
            return float(result.statistic), float(result.pvalue)
        except TypeError:
            result = stats.wilcoxon(nonzero, zero_method="wilcox", alternative="two-sided")
            return float(result.statistic), float(result.pvalue)
        except ValueError:
            return 0.0, 1.0

    ranks = rankdata_fallback(np.abs(nonzero))
    w_pos = float(ranks[nonzero > 0].sum())
    w_neg = float(ranks[nonzero < 0].sum())
    statistic = min(w_pos, w_neg)
    n = len(nonzero)
    mean_w = n * (n + 1) / 4
    sd_w = math.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    if sd_w == 0:
        return statistic, 1.0
    z = (statistic - mean_w) / sd_w
    p = math.erfc(abs(z) / math.sqrt(2))
    return statistic, p


def rank_biserial_effect(values: np.ndarray) -> float | None:
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    nonzero = values[values != 0]
    if len(nonzero) == 0:
        return 0.0 if len(values) else None
    if stats is not None:
        ranks = stats.rankdata(np.abs(nonzero))
    else:
        ranks = rankdata_fallback(np.abs(nonzero))
    w_pos = float(ranks[nonzero > 0].sum())
    w_neg = float(ranks[nonzero < 0].sum())
    denom = w_pos + w_neg
    if denom == 0:
        return 0.0
    return (w_pos - w_neg) / denom


def bootstrap_mean_ci(values: np.ndarray, rng: np.random.Generator) -> tuple[float | None, float | None]:
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return None, None
    if len(values) == 1:
        return float(values[0]), float(values[0])
    sample_indices = rng.integers(0, len(values), size=(BOOTSTRAP_ITERATIONS, len(values)))
    means = values[sample_indices].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def benjamini_hochberg(p_values: list[float | None]) -> list[float | None]:
    indexed = [(i, p) for i, p in enumerate(p_values) if p is not None and not pd.isna(p)]
    adjusted: list[float | None] = [None] * len(p_values)
    if not indexed:
        return adjusted
    indexed.sort(key=lambda item: item[1])
    m = len(indexed)
    raw = [0.0] * m
    for rank, (_, p) in enumerate(indexed, start=1):
        raw[rank - 1] = min(float(p) * m / rank, 1.0)
    cumulative = 1.0
    for pos in range(m - 1, -1, -1):
        cumulative = min(cumulative, raw[pos])
        original_index = indexed[pos][0]
        adjusted[original_index] = cumulative
    return adjusted


def add_fdr_and_significance(df: pd.DataFrame, alpha: float) -> pd.DataFrame:
    if df.empty or "p_value" not in df.columns:
        return df
    df = df.copy()
    df["p_value_fdr"] = benjamini_hochberg(df["p_value"].tolist())
    df["significant_at_alpha"] = df["p_value_fdr"].map(
        lambda p: bool(p <= alpha) if p is not None and not pd.isna(p) else False
    )
    return df


def create_rq1_stat_tests(per_example: pd.DataFrame, alpha: float, out_dir: Path, suffix: str) -> pd.DataFrame:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows = []
    if per_example.empty:
        df = pd.DataFrame()
        df.to_csv(out_dir / f"statistical_tests_rq1_{suffix}.csv", index=False)
        return df

    key_cols = ["question_index", "question", "remedy", "retrieval_condition"]
    missing_cols = [col for col in key_cols + ["model_arm"] if col not in per_example.columns]
    if missing_cols:
        df = pd.DataFrame(
            [{"warning": f"Missing columns for RQ1 statistical tests: {', '.join(missing_cols)}"}]
        )
        df.to_csv(out_dir / f"statistical_tests_rq1_{suffix}.csv", index=False)
        return df

    for condition in sorted_conditions(per_example["retrieval_condition"]):
        condition_df = per_example[per_example["retrieval_condition"] == condition]
        for metric in METRICS:
            if metric not in condition_df.columns:
                values = np.array([], dtype=float)
            else:
                pivot = condition_df.pivot_table(
                    index=key_cols,
                    columns="model_arm",
                    values=metric,
                    aggfunc="first",
                )
                if "baseline_rag" in pivot.columns and "instructrag_icl_rag" in pivot.columns:
                    pair_df = pivot[["baseline_rag", "instructrag_icl_rag"]].dropna()
                    if HIGHER_IS_BETTER[metric]:
                        values = (pair_df["instructrag_icl_rag"] - pair_df["baseline_rag"]).to_numpy(dtype=float)
                    else:
                        values = (pair_df["baseline_rag"] - pair_df["instructrag_icl_rag"]).to_numpy(dtype=float)
                else:
                    values = np.array([], dtype=float)

            statistic, p_value = wilcoxon_signed_rank(values)
            ci_low, ci_high = bootstrap_mean_ci(values, rng)
            mean_diff = float(np.mean(values)) if len(values) else None
            median_diff = float(np.median(values)) if len(values) else None
            rows.append(
                {
                    "retrieval_condition": condition,
                    "metric": metric,
                    "n_pairs": int(len(values)),
                    "mean_diff_oriented": mean_diff,
                    "median_diff_oriented": median_diff,
                    "bootstrap_ci_low": ci_low,
                    "bootstrap_ci_high": ci_high,
                    "wilcoxon_statistic": statistic,
                    "p_value": p_value,
                    "p_value_fdr": None,
                    "effect_size": rank_biserial_effect(values),
                    "significant_at_alpha": False,
                    "better_method_by_mean": (
                        "instructrag_icl_rag"
                        if mean_diff is not None and mean_diff > 0
                        else "baseline_rag"
                        if mean_diff is not None and mean_diff < 0
                        else "tie"
                        if mean_diff == 0
                        else "missing"
                    ),
                }
            )
    df = add_fdr_and_significance(pd.DataFrame(rows), alpha)
    df.to_csv(out_dir / f"statistical_tests_rq1_{suffix}.csv", index=False)
    return df


def create_rq2_stat_tests(per_example: pd.DataFrame, alpha: float, out_dir: Path, suffix: str) -> pd.DataFrame:
    rng = np.random.default_rng(BOOTSTRAP_SEED + 2)
    rows = []
    if per_example.empty:
        df = pd.DataFrame()
        df.to_csv(out_dir / f"statistical_tests_rq2_{suffix}.csv", index=False)
        return df

    key_cols = ["question_index", "question", "remedy"]
    missing_cols = [col for col in key_cols + ["retrieval_condition", "model_arm"] if col not in per_example.columns]
    if missing_cols:
        df = pd.DataFrame(
            [{"warning": f"Missing columns for RQ2 statistical tests: {', '.join(missing_cols)}"}]
        )
        df.to_csv(out_dir / f"statistical_tests_rq2_{suffix}.csv", index=False)
        return df

    for metric in METRICS:
        if metric not in per_example.columns:
            values = np.array([], dtype=float)
            baseline_deg = np.array([], dtype=float)
            instruct_deg = np.array([], dtype=float)
        else:
            pivot = per_example.pivot_table(
                index=key_cols,
                columns=["model_arm", "retrieval_condition"],
                values=metric,
                aggfunc="first",
            )
            required = [
                ("baseline_rag", "most_similar"),
                ("baseline_rag", "most_contradictory"),
                ("instructrag_icl_rag", "most_similar"),
                ("instructrag_icl_rag", "most_contradictory"),
            ]
            if all(col in pivot.columns for col in required):
                pair_df = pivot[required].dropna()
                baseline_sim = pair_df[("baseline_rag", "most_similar")].to_numpy(dtype=float)
                baseline_con = pair_df[("baseline_rag", "most_contradictory")].to_numpy(dtype=float)
                instruct_sim = pair_df[("instructrag_icl_rag", "most_similar")].to_numpy(dtype=float)
                instruct_con = pair_df[("instructrag_icl_rag", "most_contradictory")].to_numpy(dtype=float)
                if HIGHER_IS_BETTER[metric]:
                    baseline_deg = baseline_sim - baseline_con
                    instruct_deg = instruct_sim - instruct_con
                else:
                    baseline_deg = baseline_con - baseline_sim
                    instruct_deg = instruct_con - instruct_sim
                values = baseline_deg - instruct_deg
            else:
                values = np.array([], dtype=float)
                baseline_deg = np.array([], dtype=float)
                instruct_deg = np.array([], dtype=float)

        statistic, p_value = wilcoxon_signed_rank(values)
        ci_low, ci_high = bootstrap_mean_ci(values, rng)
        mean_reduction = float(np.mean(values)) if len(values) else None
        mean_baseline = float(np.mean(baseline_deg)) if len(baseline_deg) else None
        mean_instruct = float(np.mean(instruct_deg)) if len(instruct_deg) else None
        interpretation = interpret_rq2_row(mean_baseline, mean_instruct, mean_reduction)
        rows.append(
            {
                "metric": metric,
                "n_pairs": int(len(values)),
                "mean_degradation_baseline": mean_baseline,
                "mean_degradation_instructrag": mean_instruct,
                "mean_degradation_reduction": mean_reduction,
                "median_degradation_reduction": float(np.median(values)) if len(values) else None,
                "bootstrap_ci_low": ci_low,
                "bootstrap_ci_high": ci_high,
                "wilcoxon_statistic": statistic,
                "p_value": p_value,
                "p_value_fdr": None,
                "effect_size": rank_biserial_effect(values),
                "significant_at_alpha": False,
                "interpretation": interpretation,
            }
        )
    df = add_fdr_and_significance(pd.DataFrame(rows), alpha)
    df.to_csv(out_dir / f"statistical_tests_rq2_{suffix}.csv", index=False)
    return df


def setup_matplotlib():
    try:
        mpl_cache = Path("/tmp") / "contradictions_instructrag_analysis_matplotlib"
        mpl_cache.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import TwoSlopeNorm
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "matplotlib is required to create the requested PNG figures. "
            "Install contradictions_instructrag_analysis/requirements.txt."
        ) from exc
    return plt, TwoSlopeNorm


def plot_grouped_bar(
    aggregate: pd.DataFrame,
    metric: str,
    out_path: Path,
    title_suffix: str,
    ylabel: str | None = None,
) -> None:
    if aggregate.empty:
        return
    plt, _ = setup_matplotlib()
    conditions = sorted_conditions(aggregate["retrieval_condition"])
    arms = [arm for arm in ARM_ORDER if arm in set(aggregate["model_arm"])]
    if not conditions or not arms:
        return

    x = np.arange(len(conditions))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    colors = {"baseline_rag": "#2f6fbb", "instructrag_icl_rag": "#2e8b57"}
    offsets = np.linspace(-width / 2, width / 2, len(arms))
    for offset, arm in zip(offsets, arms):
        values = [
            aggregate_value(aggregate, arm, condition, metric)
            for condition in conditions
        ]
        ax.bar(x + offset, values, width=width, label=arm, color=colors.get(arm))
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=20, ha="right")
    ax.set_ylabel(ylabel or METRIC_LABELS[metric])
    title = f"{METRIC_LABELS[metric]} por condição"
    if title_suffix:
        title = f"{title} ({title_suffix})"
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_rq1_heatmap(df: pd.DataFrame, out_path: Path, title_suffix: str) -> None:
    if df.empty:
        return
    plt, TwoSlopeNorm = setup_matplotlib()
    conditions = sorted_conditions(df["retrieval_condition"])
    metrics = [metric for metric in METRICS if metric in set(df["metric"])]
    matrix = np.full((len(metrics), len(conditions)), np.nan)
    for i, metric in enumerate(metrics):
        for j, condition in enumerate(conditions):
            subset = df[(df["metric"] == metric) & (df["retrieval_condition"] == condition)]
            if not subset.empty:
                matrix[i, j] = subset.iloc[0]["delta_instructrag_advantage"]
    if matrix.size == 0:
        return
    finite = matrix[np.isfinite(matrix)]
    if finite.size == 0:
        return
    max_abs = float(np.max(np.abs(finite))) or 1.0
    norm = TwoSlopeNorm(vmin=-max_abs, vcenter=0, vmax=max_abs)
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    im = ax.imshow(matrix, cmap="RdBu", norm=norm, aspect="auto")
    ax.set_xticks(np.arange(len(conditions)))
    ax.set_xticklabels(conditions, rotation=20, ha="right")
    ax.set_yticks(np.arange(len(metrics)))
    ax.set_yticklabels([METRIC_LABELS[m] for m in metrics])
    title = "Delta orientado para vantagem InstructRAG-ICL"
    if title_suffix:
        title = f"{title} ({title_suffix})"
    ax.set_title(title)
    for i in range(len(metrics)):
        for j in range(len(conditions)):
            value = matrix[i, j]
            if np.isfinite(value):
                ax.text(j, i, f"{value:.4f}", ha="center", va="center", fontsize=8)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Delta orientado")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_rq2_degradation(df: pd.DataFrame, out_path: Path, title_suffix: str) -> None:
    if df.empty:
        return
    plt, _ = setup_matplotlib()
    metrics = list(df["metric"])
    x = np.arange(len(metrics))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ax.bar(
        x - width / 2,
        df["baseline_degradation"],
        width=width,
        label="baseline_rag",
        color="#2f6fbb",
    )
    ax.bar(
        x + width / 2,
        df["instructrag_degradation"],
        width=width,
        label="instructrag_icl_rag",
        color="#2e8b57",
    )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([METRIC_LABELS[m] for m in metrics], rotation=25, ha="right")
    ax.set_ylabel("Degradação: positivo indica piora")
    title = "Degradação most_similar -> most_contradictory"
    if title_suffix:
        title = f"{title} ({title_suffix})"
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def create_figures(
    aggregate_with_negative: pd.DataFrame,
    rq1_with_negative: pd.DataFrame,
    rq2_with_negative: pd.DataFrame,
    out_dir: Path,
) -> list[str]:
    figures_dir = out_dir / "figures"
    created: list[str] = []

    for metric, base_name, label in [
        ("semantic_cosine", "bar_semantic_cosine_by_condition", "Semantic cosine"),
        ("rougeL", "bar_rougeL_by_condition", "ROUGE-L"),
        ("vsim", "bar_vsim_by_condition", "VSIM"),
        ("jsd", "bar_jsd_by_condition", "JSD (menor é melhor)"),
    ]:
        path = figures_dir / f"{base_name}.png"
        plot_grouped_bar(aggregate_with_negative, metric, path, "", ylabel=label)
        if path.exists():
            created.append(str(path))

    path = figures_dir / "rq1_delta_heatmap.png"
    plot_rq1_heatmap(rq1_with_negative, path, "")
    if path.exists():
        created.append(str(path))

    path = figures_dir / "rq2_degradation_comparison.png"
    plot_rq2_degradation(rq2_with_negative, path, "")
    if path.exists():
        created.append(str(path))

    return created


def first_number(df: pd.DataFrame, column: str) -> Any:
    if df.empty or column not in df.columns:
        return None
    return df.iloc[0][column]


def fmt_num(value: Any, digits: int = 4) -> str:
    if value is None or pd.isna(value):
        return "n/d"
    return f"{float(value):.{digits}f}"


def fmt_pct(value: Any, digits: int = 1) -> str:
    if value is None or pd.isna(value):
        return "n/d"
    return f"{float(value) * 100:.{digits}f}%"


def significant_count(df: pd.DataFrame) -> int:
    if df.empty or "significant_at_alpha" not in df.columns:
        return 0
    return int(df["significant_at_alpha"].fillna(False).sum())


def available_tests_count(df: pd.DataFrame) -> int:
    if df.empty or "p_value" not in df.columns:
        return 0
    return int(df["p_value"].notna().sum())


def determine_h1(summary: dict[str, Any], stats_df: pd.DataFrame) -> str:
    total = summary.get("num_comparisons") or 0
    instruct = summary.get("instructrag_better") or 0
    baseline = summary.get("baseline_better") or 0
    sig_instruct = 0
    if not stats_df.empty and "significant_at_alpha" in stats_df.columns:
        sig_instruct = int(
            (
                stats_df["significant_at_alpha"].fillna(False)
                & (stats_df.get("better_method_by_mean") == "instructrag_icl_rag")
            ).sum()
        )
    if total == 0:
        return "inconclusiva"
    if instruct > total / 2 and sig_instruct > 0:
        return "apoiada pelos resultados descritivos e por parte dos testes pareados"
    if instruct > total / 2:
        return "parcialmente apoiada pelos resultados descritivos"
    if baseline > instruct:
        return "rejeitada pelos resultados descritivos"
    return "inconclusiva"


def determine_h2(rq2_df: pd.DataFrame, rq2_stats: pd.DataFrame) -> str:
    if rq2_df.empty:
        return "inconclusiva"
    valid = rq2_df.dropna(subset=["degradation_reduction"])
    if valid.empty:
        return "inconclusiva"
    no_degradation = (
        (valid["baseline_degradation"] < 0) & (valid["instructrag_degradation"] < 0)
    ).sum()
    instruct_less = (valid["degradation_reduction"] > 0).sum()
    baseline_less = (valid["degradation_reduction"] < 0).sum()
    sig_positive = 0
    if not rq2_stats.empty and "significant_at_alpha" in rq2_stats.columns:
        sig_positive = int(
            (
                rq2_stats["significant_at_alpha"].fillna(False)
                & (rq2_stats.get("mean_degradation_reduction") > 0)
            ).sum()
        )
    if no_degradation >= len(valid) / 2 and instruct_less > 0:
        return "parcialmente apoiada, pois houve melhor desempenho relativo em most_contradictory sem degradação clara a reduzir"
    if instruct_less > len(valid) / 2 and sig_positive > 0:
        return "apoiada pelos resultados descritivos e por parte dos testes pareados"
    if instruct_less > len(valid) / 2:
        return "parcialmente apoiada pelos resultados descritivos"
    if baseline_less > instruct_less:
        return "rejeitada pelos resultados descritivos"
    return "inconclusiva"


def create_article_markdown(
    out_dir: Path,
    sample_summary: pd.DataFrame,
    rq1_summary_with: dict[str, Any],
    rq2_with_negative: pd.DataFrame,
    stats_rq1_with: pd.DataFrame,
    stats_rq2_with: pd.DataFrame,
    alpha: float,
) -> None:
    sample = sample_summary.iloc[0].to_dict() if not sample_summary.empty else {}
    h1_label = determine_h1(rq1_summary_with, stats_rq1_with)
    h2_label = determine_h2(rq2_with_negative, stats_rq2_with)

    rq2_valid = rq2_with_negative.dropna(subset=["degradation_reduction"])
    rq2_instruct_less = int((rq2_valid["degradation_reduction"] > 0).sum()) if not rq2_valid.empty else 0
    rq2_baseline_less = int((rq2_valid["degradation_reduction"] < 0).sum()) if not rq2_valid.empty else 0
    rq2_no_degradation = int(
        ((rq2_valid["baseline_degradation"] < 0) & (rq2_valid["instructrag_degradation"] < 0)).sum()
    ) if not rq2_valid.empty else 0

    rq1_sig_with = significant_count(stats_rq1_with)
    rq1_tests_with = available_tests_count(stats_rq1_with)
    rq2_sig_with = significant_count(stats_rq2_with)
    rq2_tests_with = available_tests_count(stats_rq2_with)

    text = f"""# Resultados e Discussão

## Apresentação da amostra

O experimento analisou {int(sample.get('effective_question_remedy_pairs') or 0)} pares efetivos de pergunta-medicamento, a partir de {int(sample.get('planned_question_remedy_pairs') or 0)} pares planejados, quando essa informação estava disponível na configuração. Foram avaliadas {int(sample.get('retrieval_conditions') or 0)} condições de recuperação e {int(sample.get('model_arms') or 0)} abordagens: `baseline_rag` e `instructrag_icl_rag`. No total, os arquivos registram {int(sample.get('total_generation_rows') or 0)} linhas de geração pergunta-medicamento-condição e {int(sample.get('total_response_rows') or 0)} respostas avaliadas por abordagem.

## Resultados agregados

As tabelas agregadas mostram o desempenho médio por abordagem e condição de recuperação para ROUGE-1, ROUGE-2, ROUGE-L, semantic cosine, semantic dot, VSIM, JSD e KLD. Nas métricas ROUGE, semantic cosine, semantic dot e VSIM, valores maiores indicam melhor desempenho; em JSD e KLD, valores menores indicam melhor desempenho.

`instructrag_icl_rag` obteve vantagem descritiva em {rq1_summary_with.get('instructrag_better', 0)} de {rq1_summary_with.get('num_comparisons', 0)} comparações métrica-condição ({rq1_summary_with.get('percent_favoring_instructrag', 0):.1f}%). Esses resultados agregados indicam melhora descritiva frequente da abordagem com exemplos in-context, embora a interpretação dependa dos testes pareados.

## Resposta à RQ1

A RQ1 investigou se o uso de InstructRAG-ICL melhora a qualidade das respostas em comparação à linha de base RAG. Considerando os deltas orientados para que valores positivos sempre favoreçam `instructrag_icl_rag`, a hipótese H1 é {h1_label}. Essa conclusão é descritiva quando baseada nas médias agregadas e estatística apenas nos casos em que os testes pareados com correção FDR indicaram significância.

Nos testes pareados da RQ1, {rq1_sig_with} de {rq1_tests_with} comparações foram significativas após FDR com alpha={alpha}. Assim, quando a significância não aparece de forma consistente em todas as métricas e condições, os resultados sugerem tendência ou melhora descritiva, mas não sustentam uma afirmação forte de superioridade estatística geral.

## Resposta à RQ2

A RQ2 avaliou a degradação ao passar de `most_similar` para `most_contradictory`. `instructrag_icl_rag` apresentou menor degradação em {rq2_instruct_less} métricas, enquanto `baseline_rag` apresentou menor degradação em {rq2_baseline_less} métricas. Em {rq2_no_degradation} métricas, os valores médios indicaram melhora em `most_contradictory` em vez de queda de desempenho.

Com base nessas regras, a hipótese H2 é {h2_label}. Quando não há degradação real de `most_similar` para `most_contradictory`, H2 não pode ser confirmada exatamente na forma prevista. Nesses casos, um melhor resultado em `most_contradictory` deve ser tratado como achado relevante, mas diferente da hipótese original de robustez contra degradação.

## Significância estatística

Os testes pareados foram conduzidos por instância, comparando as abordagens para a mesma pergunta, medicamento e condição de recuperação. Para RQ2, a unidade pareada foi a diferença de degradação por pergunta-medicamento entre `most_similar` e `most_contradictory`. Os p-valores foram corrigidos por FDR Benjamini-Hochberg dentro de cada família de testes.

Na RQ2, {rq2_sig_with} de {rq2_tests_with} testes foram significativos após FDR. Esses números devem ser interpretados juntamente com o tamanho de efeito, a direção média das diferenças e os intervalos de confiança bootstrap.

## Resultados inesperados

Os resultados verificam explicitamente se `most_contradictory` teve desempenho igual ou superior a `most_similar`. Quando isso ocorre, uma explicação plausível é que documentos com maior escore automático de contradição estimada também podem conter evidências semanticamente relevantes para a pergunta. Além disso, métricas automáticas podem capturar proximidade textual ou semântica sem penalizar adequadamente inconsistências clínicas. Por fim, a contradição foi estimada automaticamente, não validada clinicamente de forma manual neste experimento.

## Principais achados

1. A amostra efetiva incluiu {int(sample.get('effective_question_remedy_pairs') or 0)} pares pergunta-medicamento e {int(sample.get('total_response_rows') or 0)} respostas por abordagem/condição.
2. `instructrag_icl_rag` favoreceu {rq1_summary_with.get('instructrag_better', 0)} de {rq1_summary_with.get('num_comparisons', 0)} comparações métrica-condição.
3. Para RQ2, `instructrag_icl_rag` degradou menos em {rq2_instruct_less} métricas; em {rq2_no_degradation} métricas, houve melhora em `most_contradictory` para ambas as abordagens.
"""
    (out_dir / "article_results_discussion_draft.md").write_text(text, encoding="utf-8")


def latex_escape(value: Any) -> str:
    text = "" if value is None or pd.isna(value) else str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def dataframe_to_latex_table(df: pd.DataFrame, caption: str, label: str, max_rows: int | None = None) -> str:
    if df.empty:
        return f"% {caption}: no data available.\n"
    if max_rows is not None:
        df = df.head(max_rows)
    cols = list(df.columns)
    align = "l" * len(cols)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\scriptsize",
        rf"\caption{{{latex_escape(caption)}}}",
        rf"\label{{{latex_escape(label)}}}",
        rf"\begin{{tabular}}{{{align}}}",
        r"\toprule",
        " & ".join(latex_escape(col) for col in cols) + r" \\",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        lines.append(" & ".join(latex_escape(row[col]) for col in cols) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def prepare_rq1_latex(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    out = df[["retrieval_condition", "metric", "delta_instructrag_advantage", "better_method"]].copy()
    out["delta_instructrag_advantage"] = out["delta_instructrag_advantage"].map(lambda x: fmt_num(x))
    return out


def prepare_rq2_latex(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    out = df[
        ["metric", "baseline_degradation", "instructrag_degradation", "degradation_reduction", "interpretation"]
    ].copy()
    for col in ["baseline_degradation", "instructrag_degradation", "degradation_reduction"]:
        out[col] = out[col].map(lambda x: fmt_num(x))
    return out


def prepare_rq1_stats_latex(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    out = df.sort_values("p_value_fdr", na_position="last")[
        [
            "retrieval_condition",
            "metric",
            "n_pairs",
            "mean_diff_oriented",
            "p_value_fdr",
            "effect_size",
            "significant_at_alpha",
        ]
    ].copy()
    for col in ["mean_diff_oriented", "p_value_fdr", "effect_size"]:
        out[col] = out[col].map(lambda x: fmt_num(x))
    return out


def prepare_rq2_stats_latex(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    out = df.sort_values("p_value_fdr", na_position="last")[
        [
            "metric",
            "n_pairs",
            "mean_degradation_baseline",
            "mean_degradation_instructrag",
            "mean_degradation_reduction",
            "p_value_fdr",
            "effect_size",
            "significant_at_alpha",
        ]
    ].copy()
    for col in [
        "mean_degradation_baseline",
        "mean_degradation_instructrag",
        "mean_degradation_reduction",
        "p_value_fdr",
        "effect_size",
    ]:
        out[col] = out[col].map(lambda x: fmt_num(x))
    return out


def create_latex_tables(
    out_dir: Path,
    sample_summary: pd.DataFrame,
    article_aggregate_with: pd.DataFrame,
    rq1_with: pd.DataFrame,
    rq2_with: pd.DataFrame,
    stats_rq1_with: pd.DataFrame,
    stats_rq2_with: pd.DataFrame,
) -> None:
    sample = sample_summary.copy()
    if not sample.empty:
        sample = sample[
            [
                "planned_question_remedy_pairs",
                "effective_question_remedy_pairs",
                "retrieval_conditions",
                "model_arms",
                "total_generation_rows",
                "total_response_rows",
            ]
        ]

    content = "\n".join(
        [
            "% Auto-generated by contradictions_instructrag_analysis/analyze_phase2_results.py",
            "% Requires \\usepackage{booktabs}.",
            dataframe_to_latex_table(sample, "Resumo da amostra", "tab:sample-summary"),
            dataframe_to_latex_table(article_aggregate_with, "Métricas agregadas", "tab:aggregate-main", max_rows=12),
            dataframe_to_latex_table(prepare_rq1_latex(rq1_with), "Deltas orientados da RQ1", "tab:rq1-deltas", max_rows=24),
            dataframe_to_latex_table(prepare_rq2_latex(rq2_with), "Degradação most_similar para most_contradictory", "tab:rq2-degradation", max_rows=8),
            dataframe_to_latex_table(prepare_rq1_stats_latex(stats_rq1_with), "Testes pareados da RQ1", "tab:rq1-statistics", max_rows=24),
            dataframe_to_latex_table(prepare_rq2_stats_latex(stats_rq2_with), "Testes pareados da RQ2", "tab:rq2-statistics", max_rows=8),
        ]
    )
    (out_dir / "article_tables.tex").write_text(content, encoding="utf-8")


def create_analysis_summary(
    out_dir: Path,
    data: RunData,
    sample_summary: pd.DataFrame,
    rq1_summary_with: dict[str, Any],
    rq2_with: pd.DataFrame,
    stats_rq1_with: pd.DataFrame,
    stats_rq2_with: pd.DataFrame,
    figures: list[str],
    alpha: float,
) -> None:
    per_all = data.per_example_all
    conditions = sorted_conditions(per_all["retrieval_condition"]) if not per_all.empty and "retrieval_condition" in per_all else []
    arms = sorted_arms(per_all["model_arm"]) if not per_all.empty and "model_arm" in per_all else []
    loaded_lines = {
        "aggregate_metrics": int(len(data.aggregate_with_negative)),
        "per_example_metrics": int(len(data.per_example_all)),
        "paired_comparison": int(len(data.paired_with_negative)),
        "raw_generations": int(len(data.raw_generations)),
    }
    stats_available = {
        "rq1": available_tests_count(stats_rq1_with) > 0,
        "rq2": available_tests_count(stats_rq2_with) > 0,
    }
    rq2_summary = {
        "all_rows": {
            "metrics_instructrag_degraded_less": int((rq2_with["degradation_reduction"] > 0).sum()) if not rq2_with.empty else 0,
            "metrics_baseline_degraded_less": int((rq2_with["degradation_reduction"] < 0).sum()) if not rq2_with.empty else 0,
            "metrics_no_degradation_both": int(((rq2_with["baseline_degradation"] < 0) & (rq2_with["instructrag_degradation"] < 0)).sum()) if not rq2_with.empty else 0,
        },
    }
    summary = {
        "analysis_created_at": datetime.now().isoformat(timespec="seconds"),
        "alpha": alpha,
        "analysis_scope": "all_rows",
        "run_dir": str(data.run_dir),
        "analysis_output_dir": str(out_dir),
        "loaded_rows": loaded_lines,
        "metrics_analyzed": METRICS,
        "conditions_found": conditions,
        "model_arms_found": arms,
        "sample_summary": sample_summary.iloc[0].to_dict() if not sample_summary.empty else {},
        "main_results_rq1": {"all_rows": rq1_summary_with},
        "main_results_rq2": rq2_summary,
        "statistical_tests_available": stats_available,
        "statistical_tests_significant_counts": {
            "rq1": significant_count(stats_rq1_with),
            "rq2": significant_count(stats_rq2_with),
        },
        "figures_created": figures,
        "missing_files": data.missing_files,
        "warnings": [
            "Statistical tests use only paired rows with numeric metric values for the tested metric.",
            "Aggregate n_examples columns may include rows where individual metric values were unavailable, following the experiment output files.",
        ],
    }
    write_json(out_dir / "analysis_summary.json", summary)


def print_terminal_summary(
    out_dir: Path,
    rq1_summary_with: dict[str, Any],
    rq2_with: pd.DataFrame,
    stats_rq1_with: pd.DataFrame,
    stats_rq2_with: pd.DataFrame,
    missing_files: list[str],
) -> None:
    key_files = [
        "sample_summary.csv",
        "sample_by_condition_and_arm.csv",
        "aggregate_table_all_rows.csv",
        "rq1_deltas_by_condition.csv",
        "rq2_degradation_most_similar_to_most_contradictory.csv",
        "statistical_tests_rq1_all_rows.csv",
        "statistical_tests_rq2_all_rows.csv",
        "article_results_discussion_draft.md",
        "article_tables.tex",
        "analysis_summary.json",
    ]
    print("\nAnalysis results saved to:")
    print(f"  {out_dir}")
    print("\nMain files generated:")
    for name in key_files:
        print(f"  - {name}")
    print("  - figures/*.png")

    rq2_instruct_less = int((rq2_with["degradation_reduction"] > 0).sum()) if not rq2_with.empty else 0
    rq2_baseline_less = int((rq2_with["degradation_reduction"] < 0).sum()) if not rq2_with.empty else 0
    print("\nShort findings:")
    print(
        "  - RQ1: "
        f"InstructRAG favored in {rq1_summary_with.get('instructrag_better', 0)}/"
        f"{rq1_summary_with.get('num_comparisons', 0)} comparisons."
    )
    print(
        "  - RQ2: "
        f"InstructRAG degraded less in {rq2_instruct_less} metrics; "
        f"baseline degraded less in {rq2_baseline_less} metrics."
    )

    tests_executed = any(
        available_tests_count(df) > 0
        for df in [stats_rq1_with, stats_rq2_with]
    )
    print("\nStatistical tests:")
    print(f"  - Executed: {'yes' if tests_executed else 'no'}")
    print(f"  - RQ1 significant: {significant_count(stats_rq1_with)}")
    print(f"  - RQ2 significant: {significant_count(stats_rq2_with)}")

    print("\nMissing expected files:")
    if missing_files:
        for name in missing_files:
            print(f"  - {name}")
    else:
        print("  - none")


def main() -> int:
    args = parse_args()
    if args.alpha <= 0 or args.alpha >= 1:
        raise SystemExit("--alpha must be between 0 and 1.")

    run_dir = args.run_dir.resolve() if args.run_dir else find_latest_run()
    if not run_dir.exists() or not run_dir.is_dir():
        raise SystemExit(f"Run directory does not exist or is not a directory: {run_dir}")
    out_dir = make_timestamped_output_dir(args.output_dir)
    data = load_run(run_dir)

    sample_summary, sample_by_condition_and_arm = create_sample_tables(data, out_dir)
    aggregate_with, article_with = create_aggregate_tables(
        data.aggregate_with_negative, out_dir, "all_rows"
    )

    rq1_with, rq1_summary_with = create_rq1_deltas(data.aggregate_with_negative, out_dir, "all_rows")
    write_json(out_dir / "rq1_summary.json", {"all_rows": rq1_summary_with})

    rq2_with = create_rq2_degradation(data.aggregate_with_negative, out_dir, "all_rows")

    stats_rq1_with = create_rq1_stat_tests(data.per_example_all, args.alpha, out_dir, "all_rows")
    stats_rq2_with = create_rq2_stat_tests(data.per_example_all, args.alpha, out_dir, "all_rows")

    figures = create_figures(
        data.aggregate_with_negative,
        rq1_with,
        rq2_with,
        out_dir,
    )

    create_article_markdown(
        out_dir,
        sample_summary,
        rq1_summary_with,
        rq2_with,
        stats_rq1_with,
        stats_rq2_with,
        args.alpha,
    )
    create_latex_tables(
        out_dir,
        sample_summary,
        article_with,
        rq1_with,
        rq2_with,
        stats_rq1_with,
        stats_rq2_with,
    )
    create_analysis_summary(
        out_dir,
        data,
        sample_summary,
        rq1_summary_with,
        rq2_with,
        stats_rq1_with,
        stats_rq2_with,
        figures,
        args.alpha,
    )
    print_terminal_summary(
        out_dir,
        rq1_summary_with,
        rq2_with,
        stats_rq1_with,
        stats_rq2_with,
        data.missing_files,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
