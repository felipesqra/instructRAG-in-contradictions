from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm is listed in requirements.
    def tqdm(iterable, **kwargs):
        return iterable

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from config import RationaleGenerationConfig, load_config
from data_loader import DataDiscoveryError, RationaleExample, load_selected_examples
from llm_clients import create_llm_client
from output_writer import OutputWriter
from prompt_builder import PromptBundle, build_prompt
from utils import ensure_dir, slugify, timestamp_for_path, utc_now_iso


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate InstructRAG-style denoising rationales.")
    parser.add_argument("--config", required=True, help="Path to rationale_generation.yaml")
    parser.add_argument(
        "--validate-data-only",
        action="store_true",
        help="Load selected real data and build prompts without calling a model.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config = load_config(args.config)
        bundle = load_selected_examples(config.project.contradictions_repo_path, config.selection)
    except (DataDiscoveryError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if args.validate_data_only:
        _print_validation_summary(config, bundle)
        return 0

    try:
        client = create_llm_client(config.model, config.deepseek, config.huggingface)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1

    run_id = f"{timestamp_for_path()}_{slugify(config.project.run_name)}"
    run_dir = ensure_dir(config.project.output_dir / run_id)
    writer = OutputWriter(run_dir, config.outputs)
    writer.copy_config(config.config_path)

    rationale_records: List[Dict[str, Any]] = []
    document_records: List[Dict[str, Any]] = []
    skipped_generation_records: List[Dict[str, Any]] = []
    started_at = utc_now_iso()

    for example in tqdm(bundle.examples, desc="Generating rationales"):
        try:
            prompt = _build_prompt_for_example(config, example)
            documents_used = _select_documents(config, example)
            rationale = client.generate(prompt)
            record = _build_success_record(run_id, config, example, documents_used, prompt, rationale)
        except Exception as exc:
            skipped_generation_records.append(_build_generation_skip_record(example, exc))
            continue
        rationale_records.append(record)
        document_records.append(_build_document_record(run_id, example, documents_used))

    writer.write_rationales(rationale_records)
    writer.write_prompts(rationale_records)
    writer.write_documents(document_records)
    writer.write_csv_summary(rationale_records)
    writer.write_summary(
        {
            "run_id": run_id,
            "started_at": started_at,
            "finished_at": utc_now_iso(),
            "provider": config.model.provider,
            "model_name": _effective_model_name(config),
            "num_selected_examples": len(bundle.examples),
            "num_examples": len(rationale_records),
            "num_success": len(rationale_records),
            "num_error": 0,
            "num_skipped_data_examples": len(bundle.report.skipped_examples),
            "num_skipped_generation_examples": len(skipped_generation_records),
            "skipped_generation_examples": skipped_generation_records,
            "output_dir": str(run_dir),
            "source_summary": bundle.source_summary,
            "data_discovery": bundle.report,
        }
    )
    print(f"Rationale generation run saved to: {run_dir}")
    return 0


def _print_validation_summary(config: RationaleGenerationConfig, bundle) -> None:
    print("Data validation succeeded.")
    print(f"Contradictions repository: {config.project.contradictions_repo_path}")
    print(f"Selected remedies: {', '.join(bundle.report.selected_remedies)}")
    print(f"Selected examples: {len(bundle.examples)}")
    print(f"Skipped examples: {len(bundle.report.skipped_examples)}")
    print(f"Official answers loaded: {bundle.source_summary['official_answers_loaded']}")
    print(f"Abstract files available: {bundle.source_summary['abstract_files_available']}")
    first = bundle.examples[0]
    documents_used = _select_documents(config, first)
    prompt = _build_prompt_for_example(config, first)
    print("First selected example:")
    print(f"  question_index: {first.question_index}")
    print(f"  remedy: {first.remedy}")
    print(f"  question: {first.question}")
    print(f"  official_answer_chars: {len(first.official_answer)}")
    print(f"  documents_available: {len(first.documents)}")
    print(f"  documents_used_by_config: {len(documents_used)}")
    print(f"  prompt_chars: {len(prompt.prompt_text)}")


def _build_prompt_for_example(
    config: RationaleGenerationConfig,
    example: RationaleExample,
) -> PromptBundle:
    return build_prompt(
        question=example.question,
        official_answer=example.official_answer,
        documents=_select_documents(config, example),
        prompt_config=config.prompt,
        documents_config=config.documents,
    )


def _select_documents(
    config: RationaleGenerationConfig,
    example: RationaleExample,
):
    return example.documents[: config.documents.max_documents_per_rationale]


def _classify_rationale(rationale: str, negative_template: str, status: str) -> tuple[str, bool, Any]:
    if status != "success":
        return "unknown", False, None
    negative_template_appended = rationale.strip().endswith(negative_template)
    if negative_template_appended:
        return "denoising_negative", True, False
    return "supporting_documents", False, True


def _build_success_record(
    run_id: str,
    config: RationaleGenerationConfig,
    example: RationaleExample,
    documents_used,
    prompt: PromptBundle,
    rationale: str,
) -> Dict[str, Any]:
    rationale_type, negative_template_appended, has_supporting_documents = _classify_rationale(
        rationale,
        config.prompt.negative_template,
        "success",
    )
    return _base_record(
        run_id=run_id,
        config=config,
        example=example,
        documents_used=documents_used,
        prompt=prompt,
        rationale=rationale,
        rationale_type=rationale_type,
        has_supporting_documents=has_supporting_documents,
        negative_template_appended=negative_template_appended,
        status="success",
        error=None,
    )


def _build_generation_skip_record(example: RationaleExample, exc: Exception) -> Dict[str, Any]:
    return {
        "question_index": example.question_index,
        "question": example.question,
        "remedy": example.remedy,
        "error": f"{type(exc).__name__}: {exc}",
    }


def _base_record(
    *,
    run_id: str,
    config: RationaleGenerationConfig,
    example: RationaleExample,
    documents_used,
    prompt: PromptBundle,
    rationale: str,
    rationale_type: str,
    has_supporting_documents,
    negative_template_appended: bool,
    status: str,
    error,
) -> Dict[str, Any]:
    return {
        "run_id": run_id,
        "created_at": utc_now_iso(),
        "provider": config.model.provider,
        "model_name": _effective_model_name(config),
        "temperature": config.model.temperature,
        "max_new_tokens": config.model.max_new_tokens,
        "question_index": example.question_index,
        "question": example.question,
        "remedy": example.remedy,
        "official_answer": example.official_answer,
        "num_documents_available": len(example.documents),
        "num_documents_used": len(documents_used),
        "document_ids_used": [doc.doc_id for doc in documents_used if doc.doc_id],
        "rationale": rationale,
        "rationale_type": rationale_type,
        "has_supporting_documents": has_supporting_documents,
        "uses_model_knowledge_fallback": False,
        "negative_template_appended": negative_template_appended,
        "prompt": prompt.prompt_text,
        "source_metadata": example.metadata,
        "status": status,
        "error": error,
    }


def _build_document_record(run_id: str, example: RationaleExample, documents_used) -> Dict[str, Any]:
    return {
        "run_id": run_id,
        "question_index": example.question_index,
        "question": example.question,
        "remedy": example.remedy,
        "num_documents_available": len(example.documents),
        "num_documents_used": len(documents_used),
        "documents": [doc.to_dict() for doc in documents_used],
        "source_metadata": example.metadata,
    }


def _effective_model_name(config: RationaleGenerationConfig) -> str:
    if config.model.provider == "huggingface_local":
        return config.huggingface.model_name
    return config.model.model_name


if __name__ == "__main__":
    raise SystemExit(main())
