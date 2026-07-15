from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from config import EvaluationConfig
from data_loader import EvaluationExample
from llm_clients import BaseLLMClient
from prompt_builder import PromptBundle, build_baseline_prompt, build_instructrag_icl_prompt
from rationale_loader import RationaleBundle, select_icl_examples
from utils import contains_template, utc_now_iso


@dataclass
class RunArtifacts:
    raw_rows: list[dict[str, Any]]
    prompt_rows: list[dict[str, Any]]
    document_rows: list[dict[str, Any]]


def run_examples(
    *,
    run_id: str,
    config: EvaluationConfig,
    examples: list[EvaluationExample],
    rationale_bundle: RationaleBundle,
    original_prompt_template: str,
    baseline_client: BaseLLMClient,
    instruct_client: BaseLLMClient,
) -> RunArtifacts:
    raw_rows = []
    prompt_rows = []
    document_rows = []
    for example in examples:
        icl_examples = select_icl_examples(
            rationale_bundle,
            example.question_index,
            example.question,
            example.remedy,
            config.rationales,
            config.selection,
        )
        baseline_prompt = build_baseline_prompt(
            example,
            original_prompt_template,
            config.prompt.baseline,
            config.rationales.negative_template,
        )
        instruct_prompt = build_instructrag_icl_prompt(
            example,
            icl_examples,
            config.prompt.instructrag_icl,
            config.rationales.negative_template,
        )

        baseline_answer = ""
        instruct_answer = ""
        status = "success"
        error = None
        try:
            baseline_answer = baseline_client.generate(baseline_prompt)
            instruct_answer = instruct_client.generate(instruct_prompt)
        except Exception as exc:
            status = "error"
            error = f"{type(exc).__name__}: {exc}"

        baseline_negative = contains_template(baseline_answer, config.rationales.negative_template)
        instruct_negative = contains_template(instruct_answer, config.rationales.negative_template)
        row_negative = baseline_negative or instruct_negative
        doc_ids = [doc.doc_id for doc in example.documents if doc.doc_id]
        icl_ids = [icl.rationale_id for icl in icl_examples]

        raw_rows.append(
            {
                "run_id": run_id,
                "created_at": utc_now_iso(),
                "question_index": example.question_index,
                "question": example.question,
                "remedy": example.remedy,
                "retrieval_condition": example.retrieval_condition,
                "official_answer": example.official_answer,
                "num_documents_used": len(example.documents),
                "document_ids_used": doc_ids,
                "baseline_provider": config.models.baseline.provider,
                "baseline_model_name": config.models.baseline.model_name,
                "baseline_temperature": config.models.baseline.temperature,
                "baseline_max_new_tokens": config.models.baseline.max_new_tokens,
                "baseline_answer": baseline_answer,
                "instructrag_provider": config.models.instructrag_icl.provider,
                "instructrag_model_name": config.models.instructrag_icl.model_name,
                "instructrag_temperature": config.models.instructrag_icl.temperature,
                "instructrag_max_new_tokens": config.models.instructrag_icl.max_new_tokens,
                "instructrag_answer": instruct_answer,
                "icl_rationale_ids_used": icl_ids,
                "baseline_contains_negative_template": baseline_negative,
                "instructrag_contains_negative_template": instruct_negative,
                "row_has_negative_template": row_negative,
                "status": status,
                "error": error,
            }
        )
        prompt_rows.extend(
            [
                _prompt_row(run_id, example, "baseline_rag", baseline_prompt, icl_ids),
                _prompt_row(run_id, example, "instructrag_icl_rag", instruct_prompt, icl_ids),
            ]
        )
        document_rows.append(
            {
                "run_id": run_id,
                "question_index": example.question_index,
                "question": example.question,
                "remedy": example.remedy,
                "retrieval_condition": example.retrieval_condition,
                "documents": [doc.to_dict() for doc in example.documents],
                "source_metadata": example.metadata,
            }
        )
    return RunArtifacts(raw_rows=raw_rows, prompt_rows=prompt_rows, document_rows=document_rows)


def _prompt_row(
    run_id: str,
    example: EvaluationExample,
    model_arm: str,
    prompt: PromptBundle,
    icl_ids: list[str],
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "question_index": example.question_index,
        "question": example.question,
        "remedy": example.remedy,
        "retrieval_condition": example.retrieval_condition,
        "model_arm": model_arm,
        "icl_rationale_ids_used": icl_ids if model_arm == "instructrag_icl_rag" else [],
        "prompt": prompt.prompt_text,
    }
