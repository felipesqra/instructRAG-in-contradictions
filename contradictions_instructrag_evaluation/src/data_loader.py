from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from config import RetrievalConfig, SelectionConfig
from utils import normalize_text, safe_relpath


class DataLoaderError(RuntimeError):
    pass


@dataclass
class RetrievedDocument:
    doc_id: str | None
    text: str
    score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "text": self.text,
            "score": self.score,
            "metadata": self.metadata,
        }


@dataclass
class EvaluationExample:
    question_index: int
    question: str
    remedy: str
    official_answer: str
    retrieval_condition: str
    documents: list[RetrievedDocument]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DataBundle:
    examples: list[EvaluationExample]
    selected_remedies: list[str]
    skipped_examples: list[str]
    source_summary: dict[str, Any]


def load_evaluation_examples(
    repo_path: Path,
    selection: SelectionConfig,
    retrieval: RetrievalConfig,
) -> DataBundle:
    repo_path = Path(repo_path).resolve()
    rag_pipeline = repo_path / "Codes" / "rag_pipeline.py"
    if not rag_pipeline.exists():
        raise DataLoaderError(f"Original rag_pipeline.py not found: {rag_pipeline}")

    answer_map, answer_files, skipped_answer_files = _load_official_answers(repo_path)
    if not answer_map:
        raise DataLoaderError("No official answers found in MedicalContradictionDetection-RAG/Result/*-response.json")

    abstract_files = sorted((repo_path / "Datasets" / "Contradiction").glob("*/*_abstract.json"))
    if not abstract_files:
        raise DataLoaderError("No *_abstract.json files found under Datasets/Contradiction.")

    remedy_files = []
    for path in abstract_files:
        payload = _read_json(path)
        remedy = normalize_text(payload.get("medicine")) or path.parent.name
        remedy_files.append((remedy, path))
    remedy_files.sort(key=lambda item: item[0].lower())

    if selection.num_remedies > len(remedy_files):
        raise DataLoaderError(f"Requested {selection.num_remedies} remedies, only {len(remedy_files)} available.")
    if selection.remedy_selection_strategy == "random":
        rng = random.Random(selection.random_seed)
        selected = sorted(rng.sample(remedy_files, selection.num_remedies), key=lambda item: item[0].lower())
    else:
        selected = remedy_files[: selection.num_remedies]

    examples: list[EvaluationExample] = []
    skipped_examples: list[str] = []
    for remedy, abstract_file in selected:
        try:
            payload = _read_json(abstract_file)
        except Exception as exc:
            skipped_examples.append(
                f"{remedy} skipped while reading {safe_relpath(abstract_file, repo_path)}: "
                f"{type(exc).__name__}: {exc}"
            )
            continue
        query_blocks = payload.get("retrieved_ranked_docs", [])
        if not isinstance(query_blocks, list):
            skipped_examples.append(f"{safe_relpath(abstract_file, repo_path)} missing retrieved_ranked_docs list.")
            continue

        contradiction_file = _find_file(abstract_file.parent, remedy, "contradiction")
        contradiction_by_query = _load_contradiction_by_query(contradiction_file) if contradiction_file else {}

        for question_index in range(
            selection.question_start_index,
            selection.question_start_index + selection.num_questions,
        ):
            if question_index >= len(query_blocks):
                skipped_examples.append(
                    f"{remedy} has no question index {question_index}; "
                    f"available 0..{max(len(query_blocks) - 1, 0)}."
                )
                continue
            block = query_blocks[question_index]
            if not isinstance(block, dict):
                skipped_examples.append(f"{remedy} question index {question_index} is not a JSON object.")
                continue
            question = normalize_text(block.get("query"))
            if not question:
                skipped_examples.append(f"{remedy} question index {question_index} missing query.")
                continue
            official = answer_map.get(question)
            if not official:
                skipped_examples.append(f"No official answer found for selected question: {question}")
                continue
            docs = _extract_documents(block.get("retrieved_docs", []))
            contradiction_records = contradiction_by_query.get(question, [])
            for condition in retrieval.retrieval_conditions:
                condition_docs = select_documents_for_condition(
                    docs,
                    contradiction_records,
                    condition,
                    retrieval.max_documents_per_query,
                    retrieval.preserve_original_document_order,
                )
                examples.append(
                    EvaluationExample(
                        question_index=question_index,
                        question=question,
                        remedy=remedy,
                        official_answer=official["answer"],
                        retrieval_condition=condition,
                        documents=condition_docs,
                        metadata={
                            "abstract_file": safe_relpath(abstract_file, repo_path),
                            "contradiction_file": safe_relpath(contradiction_file, repo_path) if contradiction_file else None,
                            "ground_truth_source_file": official["source_file"],
                            "ground_truth_record_index": official["record_index"],
                            "short_ground_truth": official.get("short_ground_truth"),
                            "long_ground_truth": official.get("long_ground_truth"),
                            "retrieval_rule": _retrieval_rule_description(condition),
                        },
                    )
                )

    if not examples:
        details = "\n".join(f"  - {item}" for item in skipped_examples[:50])
        raise DataLoaderError(
            "No selected evaluation examples could be loaded after skipping incomplete rows."
            + (f"\nSkipped examples:\n{details}" if details else "")
        )

    return DataBundle(
        examples=examples,
        selected_remedies=[item[0] for item in selected],
        skipped_examples=skipped_examples,
        source_summary={
            "rag_pipeline": safe_relpath(rag_pipeline, repo_path),
            "original_prompt_template": extract_original_prompt_template(rag_pipeline),
            "official_answer_files": [safe_relpath(path, repo_path) for path in answer_files],
            "skipped_official_answer_files": skipped_answer_files,
            "official_answer_policy": "long_ground_truth when present, otherwise short_ground_truth",
            "retrieved_document_source": "Datasets/Contradiction/<remedy>/<remedy>_abstract.json retrieved_ranked_docs[*].retrieved_docs",
            "skipped_selected_examples_count": len(skipped_examples),
            "retrieval_condition_rules": {
                name: _retrieval_rule_description(name)
                for name in retrieval.retrieval_conditions
            },
        },
    )


def extract_original_prompt_template(rag_pipeline_path: Path) -> str:
    text = rag_pipeline_path.read_text(encoding="utf-8")
    match = re.search(r'template="([^"]+)"', text)
    if match:
        return match.group(1).encode("utf-8").decode("unicode_escape")
    return "Given the following medical abstracts:\n{context}\nAnswer the question:\n{input}"


def _load_official_answers(repo_path: Path) -> tuple[dict[str, dict[str, Any]], list[Path], list[str]]:
    answers: dict[str, dict[str, Any]] = {}
    used_files: list[Path] = []
    skipped_files: list[str] = []
    for path in sorted((repo_path / "Result").glob("*-response.json")):
        try:
            payload = _read_json(path)
        except json.JSONDecodeError as exc:
            skipped_files.append(f"{safe_relpath(path, repo_path)}: {exc}")
            continue
        rows = payload.get("data", []) if isinstance(payload, dict) else []
        file_used = False
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                continue
            question = normalize_text(row.get("question"))
            short_answer = normalize_text(row.get("short_ground_truth"))
            long_answer = normalize_text(row.get("long_ground_truth"))
            answer = long_answer or short_answer
            if question and answer and question not in answers:
                answers[question] = {
                    "answer": answer,
                    "source_file": safe_relpath(path, repo_path),
                    "record_index": index,
                    "short_ground_truth": short_answer or None,
                    "long_ground_truth": long_answer or None,
                }
                file_used = True
        if file_used:
            used_files.append(path)
    return answers, used_files, skipped_files


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _find_file(directory: Path, remedy: str, suffix_name: str) -> Path | None:
    candidates = [
        directory / f"{remedy}_{suffix_name}.json",
        directory / f"{directory.name}_{suffix_name}.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    matches = sorted(directory.glob(f"*_{suffix_name}.json"))
    return matches[0] if matches else None


def _load_contradiction_by_query(path: Path) -> dict[str, list[dict[str, Any]]]:
    payload = _read_json(path)
    grouped: dict[str, list[dict[str, Any]]] = {}
    if not isinstance(payload, list):
        return grouped
    for row in payload:
        if not isinstance(row, dict):
            continue
        query = normalize_text(row.get("query"))
        if query:
            grouped.setdefault(query, []).append(row)
    return grouped


def _extract_documents(raw_docs: Any) -> list[RetrievedDocument]:
    docs: list[RetrievedDocument] = []
    if not isinstance(raw_docs, list):
        return docs
    for index, row in enumerate(raw_docs, start=1):
        if not isinstance(row, dict):
            docs.append(RetrievedDocument(doc_id=None, text=normalize_text(row), metadata={"rank_position": index}))
            continue
        doc_id = row.get("pmid") or row.get("doc_id") or row.get("id")
        score = _coerce_float(row.get("score") or row.get("retrieval_score"))
        metadata = {
            key: value
            for key, value in row.items()
            if key not in {"abstract", "text", "content", "score", "retrieval_score"}
        }
        metadata["rank_position"] = index
        docs.append(
            RetrievedDocument(
                doc_id=str(doc_id) if doc_id else None,
                text=normalize_text(row.get("abstract") or row.get("text") or row.get("content")),
                score=score,
                metadata=metadata,
            )
        )
    return docs


def _coerce_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def select_documents_for_condition(
    documents: list[RetrievedDocument],
    contradiction_records: list[dict[str, Any]],
    condition: str,
    max_documents: int,
    preserve_original_order: bool,
) -> list[RetrievedDocument]:
    if condition == "most_similar":
        selected = documents[:max_documents]
    else:
        doc_scores = _doc_contradiction_scores(contradiction_records)
        if not doc_scores:
            selected = documents[:max_documents] if condition == "most_contradictory" else documents[-max_documents:]
        else:
            reverse = condition == "most_contradictory"
            selected = sorted(
                documents,
                key=lambda doc: doc_scores.get(int(doc.metadata.get("rank_position", 0)), 0.0),
                reverse=reverse,
            )[:max_documents]
    if preserve_original_order:
        selected = sorted(selected, key=lambda doc: int(doc.metadata.get("rank_position", 0)))
    return selected


def _doc_contradiction_scores(records: list[dict[str, Any]]) -> dict[int, float]:
    scores: dict[int, float] = {}
    for row in records:
        pair = str(row.get("abstract_pair", ""))
        nums = [int(x) for x in re.findall(r"abstract(\d+)", pair)]
        score = _coerce_float(row.get("best_contradiction_score")) or 0.0
        for num in nums:
            scores[num] = max(scores.get(num, 0.0), score)
    return scores


def _retrieval_rule_description(condition: str) -> str:
    if condition == "most_similar":
        return "First K documents in the ranked order preserved by rag_pipeline.py FirstKOrderRetriever."
    if condition == "most_contradictory":
        return "Documents with highest max best_contradiction_score from <remedy>_contradiction.json, then restored to original order if configured."
    if condition == "least_contradictory":
        return "Documents with lowest max best_contradiction_score from <remedy>_contradiction.json, then restored to original order if configured."
    return condition
