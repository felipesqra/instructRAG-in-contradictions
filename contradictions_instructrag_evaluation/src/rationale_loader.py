from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from config import RationalesConfig, SelectionConfig
from data_loader import RetrievedDocument
from utils import contains_template, normalize_text, read_jsonl


class RationaleLoaderError(RuntimeError):
    pass


@dataclass
class RationaleExample:
    rationale_id: str
    question_index: int
    question: str
    remedy: str
    official_answer: str
    documents: list[RetrievedDocument]
    rationale: str
    rationale_type: str | None
    negative_template_appended: bool
    document_ids_used: list[str]
    provider: str | None
    model_name: str | None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RationaleBundle:
    run_dir: Path
    rationale_file: Path
    examples: list[RationaleExample]


def load_rationales(rationales_dir: Path, config: RationalesConfig) -> RationaleBundle:
    run_dir = config.rationale_run_dir or _latest_run_dir(Path(rationales_dir), config.rationale_file)
    rationale_file = run_dir / config.rationale_file
    if not rationale_file.exists():
        raise RationaleLoaderError(f"Phase 1 rationale file not found: {rationale_file}")

    document_lookup = _load_document_lookup(run_dir / "documents.jsonl")
    examples: list[RationaleExample] = []
    for index, row in enumerate(read_jsonl(rationale_file)):
        question = normalize_text(row.get("question"))
        remedy = normalize_text(row.get("remedy"))
        question_index = int(row.get("question_index", -1))
        docs = _documents_from_rationale_row(row)
        if not docs:
            docs = document_lookup.get((question_index, remedy, question), [])
        rationale = normalize_text(row.get("rationale"))
        if not question or not remedy or not rationale:
            continue
        examples.append(
            RationaleExample(
                rationale_id=str(row.get("rationale_id") or f"{run_dir.name}:{index}"),
                question_index=question_index,
                question=question,
                remedy=remedy,
                official_answer=normalize_text(row.get("official_answer")),
                documents=docs,
                rationale=rationale,
                rationale_type=row.get("rationale_type"),
                negative_template_appended=bool(row.get("negative_template_appended", False)),
                document_ids_used=[str(value) for value in row.get("document_ids_used", [])],
                provider=row.get("provider"),
                model_name=row.get("model_name"),
                metadata={"source_row_index": index},
            )
        )
    if not examples:
        raise RationaleLoaderError(f"No usable rationales found in {rationale_file}")
    return RationaleBundle(run_dir=run_dir, rationale_file=rationale_file, examples=examples)


def select_icl_examples(
    bundle: RationaleBundle,
    target_question_index: int,
    target_question: str,
    target_remedy: str,
    config: RationalesConfig,
    selection: SelectionConfig,
) -> list[RationaleExample]:
    candidates = [
        example for example in bundle.examples
        if not (
            example.question_index == target_question_index
            and example.remedy == target_remedy
            and example.question == target_question
        )
    ]
    if not config.include_negative_rationales_as_icl:
        candidates = [
            example for example in candidates
            if not example.negative_template_appended
            and not contains_template(example.rationale, config.negative_template)
        ]

    if config.icl_selection_strategy == "same_question_first":
        matching = [example for example in candidates if example.question_index == target_question_index]
        others = [example for example in candidates if example.question_index != target_question_index]
        ordered = matching + others
    elif config.icl_selection_strategy == "sequential":
        ordered = candidates
    else:
        rng = random.Random(selection.random_seed + target_question_index)
        ordered = candidates[:]
        rng.shuffle(ordered)
    return ordered[: config.max_icl_examples]


def _latest_run_dir(rationales_dir: Path, rationale_file: str) -> Path:
    if not rationales_dir.exists():
        raise RationaleLoaderError(f"Phase 1 rationales directory does not exist: {rationales_dir}")
    candidates = sorted([path for path in rationales_dir.iterdir() if path.is_dir()])
    if not candidates:
        raise RationaleLoaderError(f"No Phase 1 run directories found under: {rationales_dir}")
    usable = [path for path in candidates if (path / rationale_file).exists()]
    if not usable:
        inspected = "\n".join(f"  - {path}" for path in candidates)
        raise RationaleLoaderError(
            f"No Phase 1 run directory under {rationales_dir} contains {rationale_file}.\n"
            f"Inspected run directories:\n{inspected}"
        )
    return usable[-1]


def _load_document_lookup(path: Path) -> dict[tuple[int, str, str], list[RetrievedDocument]]:
    if not path.exists():
        return {}
    lookup: dict[tuple[int, str, str], list[RetrievedDocument]] = {}
    for row in read_jsonl(path):
        key = (
            int(row.get("question_index", -1)),
            normalize_text(row.get("remedy")),
            normalize_text(row.get("question")),
        )
        docs = []
        for doc in row.get("documents", []) or []:
            docs.append(
                RetrievedDocument(
                    doc_id=str(doc.get("doc_id")) if doc.get("doc_id") else None,
                    text=normalize_text(doc.get("text")),
                    score=doc.get("score"),
                    metadata=doc.get("metadata") or {},
                )
            )
        lookup[key] = docs
    return lookup


def _documents_from_rationale_row(row: dict[str, Any]) -> list[RetrievedDocument]:
    docs = []
    for doc in row.get("documents", []) or []:
        if not isinstance(doc, dict):
            continue
        docs.append(
            RetrievedDocument(
                doc_id=str(doc.get("doc_id")) if doc.get("doc_id") else None,
                text=normalize_text(doc.get("text")),
                score=doc.get("score"),
                metadata=doc.get("metadata") or {},
            )
        )
    return docs
