from __future__ import annotations

from pathlib import Path

from .util import atomic_write_jsonl, posix_rel, read_jsonl, sha256_file


def load_manifest(manifest_path: Path) -> list[dict]:
    return read_jsonl(manifest_path)


def _iter_pdf_paths(library_root: Path) -> list[Path]:
    return sorted(
        [path for path in library_root.rglob("*.pdf") if path.is_file()],
        key=lambda path: path.as_posix(),
    )


def build_manifest(library_root: Path, manifest_path: Path) -> list[dict]:
    if not library_root.exists():
        raise FileNotFoundError(f"Library root not found: {library_root}")

    existing_by_id = {row["doc_id"]: row for row in load_manifest(manifest_path)}
    rows: list[dict] = []

    for pdf_path in _iter_pdf_paths(library_root):
        source_rel = pdf_path.relative_to(library_root)
        source_path = posix_rel(source_rel)
        source_sha256 = sha256_file(pdf_path)
        doc_id = source_sha256[:16]
        old = existing_by_id.get(doc_id, {})

        theorist = source_rel.parts[0] if len(source_rel.parts) > 1 else pdf_path.parent.name
        row = {
            "doc_id": doc_id,
            "source_sha256": source_sha256,
            "source_path": source_path,
            "ocr_path": source_path,
            "theorist": theorist,
            "title": pdf_path.stem,
            "mtime": pdf_path.stat().st_mtime,
            "page_count": old.get("page_count"),
            "nonempty_pages": old.get("nonempty_pages"),
            "avg_chars_per_page": old.get("avg_chars_per_page"),
        }
        rows.append(row)

    rows.sort(key=lambda row: row["source_path"])
    atomic_write_jsonl(manifest_path, rows)
    return rows


def update_manifest_stats(manifest_path: Path, stats_by_doc_id: dict[str, dict]) -> list[dict]:
    rows = load_manifest(manifest_path)
    updated: list[dict] = []
    for row in rows:
        stats = stats_by_doc_id.get(row["doc_id"])
        if stats:
            row["page_count"] = int(stats["page_count"])
            row["nonempty_pages"] = int(stats["nonempty_pages"])
            row["avg_chars_per_page"] = float(stats["avg_chars_per_page"])
        updated.append(row)

    atomic_write_jsonl(manifest_path, updated)
    return updated
