from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

try:
    import pandas as pd
except Exception:  # pragma: no cover - pandas is listed as a dependency.
    pd = None

from config import SelectionConfig
from utils import normalize_text, safe_relpath


SUPPORTED_SUFFIXES = {".csv", ".json", ".jsonl", ".pkl", ".pickle", ".parquet"}
LIKELY_SEARCH_DIRS = ("Datasets", "Result", "outputs", "data", "Codes")


class DataDiscoveryError(RuntimeError):
    pass


@dataclass
class RetrievedDocument:
    doc_id: Optional[str]
    text: str
    score: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "text": self.text,
            "score": self.score,
            "metadata": self.metadata,
        }


@dataclass
class OfficialAnswer:
    question: str
    answer: str
    source_file: str
    record_index: int
    short_ground_truth: Optional[str]
    long_ground_truth: Optional[str]


@dataclass
class RationaleExample:
    question_index: int
    question: str
    remedy: str
    official_answer: str
    documents: List[RetrievedDocument]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataDiscoveryReport:
    searched_dirs: List[str]
    inspected_files: List[str]
    abstract_files: List[str]
    ground_truth_files: List[str]
    selected_remedies: List[str]
    missing_fields: List[str] = field(default_factory=list)
    skipped_examples: List[str] = field(default_factory=list)


@dataclass
class DatasetBundle:
    examples: List[RationaleExample]
    report: DataDiscoveryReport
    source_summary: Dict[str, Any]


def load_selected_examples(repo_path: Path, selection: SelectionConfig) -> DatasetBundle:
    repo_path = Path(repo_path).resolve()
    if not repo_path.exists():
        raise DataDiscoveryError(f"Contradictions repository does not exist: {repo_path}")

    searched_dirs = [safe_relpath(repo_path / name, repo_path) for name in LIKELY_SEARCH_DIRS]
    inspected_files = _discover_structured_files(repo_path)
    ground_truths, ground_truth_files = _load_official_answers(repo_path, inspected_files)
    abstract_files = _discover_abstract_files(repo_path, inspected_files)

    report = DataDiscoveryReport(
        searched_dirs=searched_dirs,
        inspected_files=[safe_relpath(path, repo_path) for path in inspected_files],
        abstract_files=[safe_relpath(path, repo_path) for path in abstract_files],
        ground_truth_files=[safe_relpath(path, repo_path) for path in ground_truth_files],
        selected_remedies=[],
    )

    if not ground_truths:
        report.missing_fields.append("No official answers found in Result/*-response.json files.")
        raise DataDiscoveryError(_format_discovery_error(repo_path, report))
    if not abstract_files:
        report.missing_fields.append("No valid *_abstract.json files found under Datasets/Contradiction.")
        raise DataDiscoveryError(_format_discovery_error(repo_path, report))

    remedy_files = _index_remedy_files(repo_path, abstract_files)
    selected_remedy_files = _select_remedy_files(remedy_files, selection)
    report.selected_remedies = [remedy for remedy, _ in selected_remedy_files]

    examples: List[RationaleExample] = []
    skipped_selected: List[str] = []
    requested_question_indices = range(
        selection.question_start_index,
        selection.question_start_index + selection.num_questions,
    )

    for remedy, abstract_file in selected_remedy_files:
        try:
            remedy_examples = _load_examples_for_remedy(
                repo_path=repo_path,
                remedy=remedy,
                abstract_file=abstract_file,
                requested_question_indices=requested_question_indices,
                ground_truths=ground_truths,
                skipped_selected=skipped_selected,
            )
        except Exception as exc:
            skipped_selected.append(
                f"{remedy} skipped while loading {safe_relpath(abstract_file, repo_path)}: "
                f"{type(exc).__name__}: {exc}"
            )
            continue
        examples.extend(remedy_examples)

    report.skipped_examples.extend(skipped_selected)
    if not examples:
        report.missing_fields.append("No selected question-remedy examples could be loaded.")
        raise DataDiscoveryError(_format_discovery_error(repo_path, report))

    _attach_contradiction_metadata(repo_path, examples)

    source_summary = {
        "official_answers_loaded": len(ground_truths),
        "ground_truth_files": report.ground_truth_files,
        "abstract_files_available": len(abstract_files),
        "selected_remedies": report.selected_remedies,
        "selected_examples": len(examples),
        "skipped_examples": len(report.skipped_examples),
        "data_join": "retrieved_ranked_docs[*].query joined to Result/*-response.json question",
        "official_answer_policy": "long_ground_truth when available, otherwise short_ground_truth",
        "document_source": "Datasets/Contradiction/<remedy>/<remedy>_abstract.json retrieved_docs",
    }
    return DatasetBundle(examples=examples, report=report, source_summary=source_summary)


def _discover_structured_files(repo_path: Path) -> List[Path]:
    files: List[Path] = []
    for dirname in LIKELY_SEARCH_DIRS:
        directory = repo_path / dirname
        if not directory.exists():
            continue
        for path in directory.rglob("*"):
            if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
                files.append(path.resolve())
    return sorted(files)


def _load_structured_file(path: Path) -> Any:
    suffix = path.suffix.lower()
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    if suffix == ".csv":
        if pd is None:
            raise DataDiscoveryError("pandas is required to read CSV files.")
        return pd.read_csv(path).to_dict(orient="records")
    if suffix in {".pkl", ".pickle"}:
        if pd is None:
            raise DataDiscoveryError("pandas is required to read pickle files.")
        return pd.read_pickle(path)
    if suffix == ".parquet":
        if pd is None:
            raise DataDiscoveryError("pandas is required to read parquet files.")
        return pd.read_parquet(path).to_dict(orient="records")
    raise DataDiscoveryError(f"Unsupported structured data file type: {path}")


def _load_official_answers(
    repo_path: Path,
    inspected_files: Iterable[Path],
) -> tuple[Dict[str, OfficialAnswer], List[Path]]:
    answers: Dict[str, OfficialAnswer] = {}
    used_files: List[Path] = []
    result_files = [
        path for path in inspected_files
        if path.suffix.lower() == ".json" and path.parent.name == "Result"
    ]

    for path in sorted(result_files):
        try:
            payload = _load_structured_file(path)
        except Exception:
            continue
        records = payload.get("data") if isinstance(payload, dict) else payload
        if not isinstance(records, list):
            continue

        file_used = False
        for index, record in enumerate(records):
            if not isinstance(record, dict):
                continue
            question = normalize_text(record.get("question"))
            short_answer = normalize_text(record.get("short_ground_truth"))
            long_answer = normalize_text(record.get("long_ground_truth"))
            answer = long_answer or short_answer
            if not question or not answer:
                continue
            if question in answers:
                continue
            answers[question] = OfficialAnswer(
                question=question,
                answer=answer,
                source_file=safe_relpath(path, repo_path),
                record_index=index,
                short_ground_truth=short_answer or None,
                long_ground_truth=long_answer or None,
            )
            file_used = True
        if file_used:
            used_files.append(path)
    return answers, used_files


def _discover_abstract_files(repo_path: Path, inspected_files: Iterable[Path]) -> List[Path]:
    primary = repo_path / "Datasets" / "Contradiction"
    abstract_files = []
    if primary.exists():
        abstract_files = sorted(primary.glob("*/*_abstract.json"))
    if abstract_files:
        return [path.resolve() for path in abstract_files if _looks_like_abstract_file(path)]

    candidates: List[Path] = []
    for path in inspected_files:
        if path.suffix.lower() != ".json":
            continue
        if _looks_like_abstract_file(path):
            candidates.append(path)
    return sorted(candidates)


def _looks_like_abstract_file(path: Path) -> bool:
    try:
        payload = _load_structured_file(path)
    except Exception:
        return False
    return (
        isinstance(payload, dict)
        and isinstance(payload.get("retrieved_ranked_docs"), list)
    )


def _index_remedy_files(repo_path: Path, abstract_files: Iterable[Path]) -> List[tuple[str, Path]]:
    remedy_files: List[tuple[str, Path]] = []
    for path in abstract_files:
        try:
            payload = _load_structured_file(path)
        except Exception:
            continue
        remedy = normalize_text(payload.get("medicine")) or path.parent.name
        remedy_files.append((remedy, path))
    remedy_files.sort(key=lambda item: item[0].lower())
    if not remedy_files:
        raise DataDiscoveryError(
            f"No remedies could be read from abstract files under {repo_path / 'Datasets' / 'Contradiction'}."
        )
    return remedy_files


def _select_remedy_files(
    remedy_files: List[tuple[str, Path]],
    selection: SelectionConfig,
) -> List[tuple[str, Path]]:
    if selection.num_remedies > len(remedy_files):
        raise DataDiscoveryError(
            f"Requested {selection.num_remedies} remedies, but only {len(remedy_files)} are available."
        )
    if selection.remedy_selection_strategy == "sequential":
        return remedy_files[: selection.num_remedies]
    rng = random.Random(selection.random_seed)
    return sorted(
        rng.sample(remedy_files, selection.num_remedies),
        key=lambda item: item[0].lower(),
    )


def _load_examples_for_remedy(
    repo_path: Path,
    remedy: str,
    abstract_file: Path,
    requested_question_indices: range,
    ground_truths: Dict[str, OfficialAnswer],
    skipped_selected: List[str],
) -> List[RationaleExample]:
    payload = _load_structured_file(abstract_file)
    query_blocks = payload.get("retrieved_ranked_docs")
    if not isinstance(query_blocks, list):
        skipped_selected.append(f"{safe_relpath(abstract_file, repo_path)} missing retrieved_ranked_docs list.")
        return []

    ranked_metadata, ranked_metadata_file = _load_ranked_metadata(repo_path, abstract_file, remedy)
    contradiction_file = _find_named_file(abstract_file.parent, remedy, "contradiction")
    examples: List[RationaleExample] = []

    for question_index in requested_question_indices:
        if question_index >= len(query_blocks):
            skipped_selected.append(
                f"{remedy} has no question at index {question_index}; "
                f"available indices are 0..{max(len(query_blocks) - 1, 0)}."
            )
            continue
        query_block = query_blocks[question_index]
        if not isinstance(query_block, dict):
            skipped_selected.append(f"{remedy} question index {question_index} is not a JSON object.")
            continue

        question = normalize_text(query_block.get("query"))
        if not question:
            skipped_selected.append(f"{remedy} question index {question_index} missing query.")
            continue

        official = ground_truths.get(question)
        if official is None:
            skipped_selected.append(
                f"{remedy} question index {question_index} has no official answer for query: {question}"
            )
            continue

        raw_docs = query_block.get("retrieved_docs")
        if not isinstance(raw_docs, list):
            skipped_selected.append(f"{remedy} question index {question_index} missing retrieved_docs list.")
            continue

        documents = _extract_documents(raw_docs, ranked_metadata)
        examples.append(
            RationaleExample(
                question_index=question_index,
                question=question,
                remedy=remedy,
                official_answer=official.answer,
                documents=documents,
                metadata={
                    "abstract_file": safe_relpath(abstract_file, repo_path),
                    "ranked_metadata_file": safe_relpath(ranked_metadata_file, repo_path)
                    if ranked_metadata_file else None,
                    "contradiction_file": safe_relpath(contradiction_file, repo_path)
                    if contradiction_file else None,
                    "ground_truth_source_file": official.source_file,
                    "ground_truth_record_index": official.record_index,
                    "short_ground_truth": official.short_ground_truth,
                    "long_ground_truth": official.long_ground_truth,
                },
            )
        )
    return examples


def _load_ranked_metadata(
    repo_path: Path,
    abstract_file: Path,
    remedy: str,
) -> tuple[Dict[str, Dict[str, Any]], Optional[Path]]:
    candidates = [
        abstract_file.parent / f"{remedy}_ranked_file.json",
        abstract_file.parent / "ranked_file.json",
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            payload = _load_structured_file(path)
        except Exception:
            continue
        if not isinstance(payload, dict) or "retrieved_ranked_docs" in payload:
            continue
        if all(isinstance(value, dict) for value in payload.values()):
            return {str(key): value for key, value in payload.items()}, path
    return {}, None


def _find_named_file(directory: Path, remedy: str, suffix_name: str) -> Optional[Path]:
    candidates = [
        directory / f"{remedy}_{suffix_name}.json",
        directory / f"{directory.name}_{suffix_name}.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    matches = sorted(directory.glob(f"*_{suffix_name}.json"))
    return matches[0] if matches else None


def _extract_documents(
    raw_docs: List[Any],
    ranked_metadata: Dict[str, Dict[str, Any]],
) -> List[RetrievedDocument]:
    documents: List[RetrievedDocument] = []
    for position, raw_doc in enumerate(raw_docs, start=1):
        if not isinstance(raw_doc, dict):
            text = normalize_text(raw_doc)
            documents.append(
                RetrievedDocument(
                    doc_id=None,
                    text=text,
                    score=None,
                    metadata={"rank_position": position},
                )
            )
            continue

        doc_id = _first_present(raw_doc, ("pmid", "doc_id", "id", "document_id"))
        doc_id = str(doc_id) if doc_id not in (None, "") else None
        text = normalize_text(
            _first_present(raw_doc, ("abstract", "text", "contents", "content", "document"))
        )
        score_value = _first_present(raw_doc, ("score", "retrieval_score", "similarity", "final_score"))
        score = _coerce_float(score_value)

        metadata = {
            key: value
            for key, value in raw_doc.items()
            if key not in {"abstract", "text", "contents", "content", "document", "score", "retrieval_score"}
        }
        metadata["rank_position"] = position
        if doc_id and doc_id in ranked_metadata:
            article_metadata = {
                key: value for key, value in ranked_metadata[doc_id].items()
                if key != "abstract"
            }
            metadata.update(article_metadata)
            if not text:
                text = normalize_text(ranked_metadata[doc_id].get("abstract"))

        documents.append(RetrievedDocument(doc_id=doc_id, text=text, score=score, metadata=metadata))
    return documents


def _first_present(mapping: Dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in mapping and mapping[key] not in (None, ""):
            return mapping[key]
    return None


def _coerce_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _attach_contradiction_metadata(repo_path: Path, examples: List[RationaleExample]) -> None:
    cache: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for example in examples:
        rel_path = example.metadata.get("contradiction_file")
        if not rel_path:
            example.metadata["contradiction_metadata"] = []
            continue
        if rel_path not in cache:
            path = repo_path / rel_path
            cache[rel_path] = _load_contradiction_file(path)
        example.metadata["contradiction_metadata"] = cache[rel_path].get(example.question, [])


def _load_contradiction_file(path: Path) -> Dict[str, List[Dict[str, Any]]]:
    if not path.exists():
        return {}
    try:
        payload = _load_structured_file(path)
    except Exception:
        return {}
    if not isinstance(payload, list):
        return {}
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for record in payload:
        if not isinstance(record, dict):
            continue
        query = normalize_text(record.get("query"))
        if not query:
            continue
        grouped.setdefault(query, []).append(record)
    return grouped


def _format_discovery_error(repo_path: Path, report: DataDiscoveryReport) -> str:
    inspected_preview = report.inspected_files[:200]
    omitted = max(len(report.inspected_files) - len(inspected_preview), 0)
    lines = [
        "Could not locate all required contradiction/RAG data.",
        f"Repository: {repo_path}",
        "Searched directories:",
        *[f"  - {directory}" for directory in report.searched_dirs],
        "Inspected files:",
        *[f"  - {path}" for path in inspected_preview],
    ]
    if omitted:
        lines.append(f"  - ... {omitted} more files omitted from this error message")
    lines.extend([
        "Required fields/files missing:",
        *[f"  - {field}" for field in report.missing_fields],
    ])
    if report.skipped_examples:
        lines.extend([
            "Skipped selected examples:",
            *[f"  - {example}" for example in report.skipped_examples],
        ])
    return "\n".join(lines)
