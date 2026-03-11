from __future__ import annotations

import hashlib
import json
import os
import re
import unicodedata
import uuid
from pathlib import Path
from typing import Iterable


TOKEN_RE = re.compile(r"[a-z0-9]+")
SENTENCE_BREAK_RE = re.compile(r"(?<=[.!?])\s+")
CLAUSE_BREAK_RE = re.compile(r"(?<=[,;:])\s+")
WHITESPACE_BREAK_RE = re.compile(r"\s+")

UNICODE_TRANSLATIONS = str.maketrans(
    {
        "\u00ad": "",
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2015": "-",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\ufb00": "ff",
        "\ufb01": "fi",
        "\ufb02": "fl",
        "\ufb03": "ffi",
        "\ufb04": "ffl",
        "\ufb05": "ft",
        "\ufb06": "st",
    }
)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def posix_rel(path: Path) -> str:
    return path.as_posix()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text_16(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def atomic_write_text(path: Path, content: str) -> None:
    ensure_dir(path.parent)
    tmp = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def atomic_write_json(path: Path, payload: dict) -> None:
    atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def atomic_write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    lines = [json.dumps(row, ensure_ascii=False) for row in rows]
    atomic_write_text(path, "\n".join(lines) + ("\n" if lines else ""))


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def normalize_unicode_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text or "")
    normalized = normalized.translate(UNICODE_TRANSLATIONS)
    return normalized.replace("\x00", "")


def normalize_page_text(text: str) -> str:
    cleaned = normalize_unicode_text(text)
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = re.sub(r"(?<=\w)-\n(?=\w)", "", cleaned)
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)

    blocks = []
    for raw_block in re.split(r"\n\s*\n+", cleaned):
        lines = []
        for raw_line in raw_block.split("\n"):
            line = re.sub(r"\s+", " ", raw_line).strip()
            if line:
                lines.append(line)
        if not lines:
            continue
        block = " ".join(lines)
        block = re.sub(r"\s+([,.;:!?])", r"\1", block)
        blocks.append(block)
    return "\n\n".join(blocks)


def tokenize(text: str) -> list[str]:
    normalized = normalize_unicode_text(text).lower()
    return TOKEN_RE.findall(normalized)


def normalize_match_text(text: str) -> str:
    return " ".join(tokenize(text))


def _preferred_split_index(segment: str, start: int, target_chars: int, window: int = 160) -> int:
    ideal = min(len(segment), start + target_chars)
    if ideal >= len(segment):
        return len(segment)

    minimum = min(len(segment), start + max(target_chars // 2, 120))
    left = max(minimum, ideal - window)
    right = min(len(segment), ideal + window)

    for pattern in (SENTENCE_BREAK_RE, CLAUSE_BREAK_RE, WHITESPACE_BREAK_RE):
        positions = []
        for match in pattern.finditer(segment[left:right]):
            split_at = left + match.end()
            if split_at > minimum:
                positions.append(split_at)
        if positions:
            return min(
                positions,
                key=lambda pos: (abs(pos - ideal), 0 if pos <= ideal else 1, pos),
            )

    return ideal


def _chunk_units(text: str, target_chars: int) -> list[str]:
    units: list[str] = []
    paragraphs = [part.strip() for part in re.split(r"\n{2,}", text) if part.strip()]
    if not paragraphs:
        paragraphs = [text.strip()] if text.strip() else []

    for paragraph in paragraphs:
        if len(paragraph) <= target_chars:
            units.append(paragraph)
            continue

        start = 0
        while start < len(paragraph):
            end = _preferred_split_index(paragraph, start=start, target_chars=target_chars)
            piece = paragraph[start:end].strip()
            if piece:
                units.append(piece)
            if end >= len(paragraph):
                break
            start = end
    return units


def chunk_text_with_overlap(text: str, target_chars: int = 1200, overlap: int = 200) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= target_chars:
        return [text]

    units = _chunk_units(text, target_chars=target_chars)
    if not units:
        return []

    chunks: list[str] = []
    current: list[str] = []

    for unit in units:
        if not current:
            current = [unit]
            continue

        proposed = "\n\n".join(current + [unit])
        if len(proposed) <= target_chars:
            current.append(unit)
            continue

        chunk = "\n\n".join(current).strip()
        if chunk:
            chunks.append(chunk)

        overlap_units: list[str] = []
        overlap_len = 0
        for previous in reversed(current):
            overlap_units.insert(0, previous)
            overlap_len = len("\n\n".join(overlap_units))
            if overlap_len >= overlap:
                break

        if overlap_units:
            proposed = "\n\n".join(overlap_units + [unit])
            if len(proposed) <= target_chars:
                current = overlap_units + [unit]
                continue

        current = [unit]

    final_chunk = "\n\n".join(current).strip()
    if final_chunk:
        chunks.append(final_chunk)
    return chunks
