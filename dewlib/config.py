from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Paths:
    data_dir: Path
    manifest_path: Path
    health_report_path: Path
    pages_dir: Path
    index_dir: Path

    @classmethod
    def from_args(
        cls,
        data_dir: str | Path = "data",
        manifest_path: str | Path | None = None,
    ) -> "Paths":
        data = Path(data_dir)
        manifest = Path(manifest_path) if manifest_path else data / "manifest.jsonl"
        return cls(
            data_dir=data,
            manifest_path=manifest,
            health_report_path=data / "health_report.json",
            pages_dir=data / "pages",
            index_dir=data / "index",
        )
