from __future__ import annotations

import csv
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List

from config import OutputsConfig
from utils import append_jsonl, ensure_dir, json_safe, write_json


class OutputWriter:
    def __init__(self, run_dir: Path, outputs_config: OutputsConfig) -> None:
        self.run_dir = ensure_dir(run_dir)
        self.outputs_config = outputs_config

    def copy_config(self, config_path: Path) -> None:
        shutil.copyfile(config_path, self.run_dir / "config_used.yaml")

    def write_rationales(self, records: Iterable[Dict[str, Any]]) -> None:
        if not self.outputs_config.save_jsonl:
            return
        with (self.run_dir / "rationales.jsonl").open("w", encoding="utf-8") as handle:
            for record in records:
                append_jsonl(handle, record)

    def write_prompts(self, records: Iterable[Dict[str, Any]]) -> None:
        if not self.outputs_config.save_prompt:
            return
        with (self.run_dir / "prompts.jsonl").open("w", encoding="utf-8") as handle:
            for record in records:
                append_jsonl(handle, {
                    "run_id": record["run_id"],
                    "question_index": record["question_index"],
                    "remedy": record["remedy"],
                    "question": record["question"],
                    "prompt": record["prompt"],
                    "status": record["status"],
                })

    def write_documents(self, document_records: Iterable[Dict[str, Any]]) -> None:
        if not self.outputs_config.save_documents:
            return
        with (self.run_dir / "documents.jsonl").open("w", encoding="utf-8") as handle:
            for record in document_records:
                append_jsonl(handle, record)

    def write_csv_summary(self, records: List[Dict[str, Any]]) -> None:
        if not self.outputs_config.save_csv_summary:
            return
        fieldnames = [
            "question_index",
            "remedy",
            "provider",
            "model_name",
            "temperature",
            "num_documents_used",
            "status",
            "rationale_type",
            "negative_template_appended",
            "rationale",
        ]
        with (self.run_dir / "rationales.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for record in records:
                writer.writerow({field: json_safe(record.get(field)) for field in fieldnames})

    def write_summary(self, summary: Dict[str, Any]) -> None:
        write_json(self.run_dir / "run_summary.json", summary)
