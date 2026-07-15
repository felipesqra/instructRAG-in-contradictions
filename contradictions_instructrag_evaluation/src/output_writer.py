from __future__ import annotations

import csv
import shutil
from pathlib import Path
from typing import Any

from utils import append_jsonl, ensure_dir, json_safe, write_json


class OutputWriter:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = ensure_dir(run_dir)

    def copy_config(self, config_path: Path) -> None:
        shutil.copyfile(config_path, self.run_dir / "config_used.yaml")

    def write_jsonl(self, filename: str, rows: list[dict[str, Any]]) -> None:
        with (self.run_dir / filename).open("w", encoding="utf-8") as handle:
            for row in rows:
                append_jsonl(handle, row)

    def write_csv(self, filename: str, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
        if fieldnames is None:
            fieldnames = sorted({key for row in rows for key in row.keys()})
        with (self.run_dir / filename).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: json_safe(row.get(key)) for key in fieldnames})

    def write_summary(self, summary: dict[str, Any]) -> None:
        write_json(self.run_dir / "run_summary.json", summary)
