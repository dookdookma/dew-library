from __future__ import annotations

from pathlib import Path

import numpy as np

from .embed import build_embedder, normalize_vectors
from .util import atomic_write_json, atomic_write_jsonl, read_json, read_jsonl, tokenize


def _faiss():
    try:
        import faiss
    except ImportError as exc:
        raise RuntimeError("faiss-cpu is required. Install dependency 'faiss-cpu'.") from exc
    return faiss


def _index_paths(index_dir: Path) -> dict[str, Path]:
    return {
        "meta": index_dir / "meta.jsonl",
        "faiss": index_dir / "faiss.index",
        "bm25": index_dir / "bm25_tokens.json",
        "embedder": index_dir / "embedder.json",
    }


def _is_fresh(chunks_path: Path, files: dict[str, Path], expected_embedder: dict) -> bool:
    if not chunks_path.exists():
        return False
    if not all(path.exists() for path in files.values()):
        return False

    current = read_json(files["embedder"])
    if current != expected_embedder:
        return False

    oldest_output = min(path.stat().st_mtime for path in files.values())
    return oldest_output >= chunks_path.stat().st_mtime


def build_hybrid_index(
    chunks_path: Path,
    index_dir: Path,
    embedder: str = "auto",
    dim: int = 384,
    model_name: str = "all-MiniLM-L6-v2",
    allow_download: bool = False,
    force: bool = False,
) -> dict:
    if not chunks_path.exists():
        raise FileNotFoundError(f"Chunks file not found: {chunks_path}")

    index_dir.mkdir(parents=True, exist_ok=True)
    files = _index_paths(index_dir)
    embedder_impl, spec = build_embedder(
        backend=embedder,
        dim=dim,
        model_name=model_name,
        allow_download=allow_download,
        fallback_to_hash=False,
    )
    embedder_json = {"backend": spec.backend, "dim": spec.dim, "model_name": spec.model_name}

    if (not force) and _is_fresh(chunks_path, files, embedder_json):
        return {"status": "skipped", "chunks": 0, "backend": embedder_json["backend"]}

    chunks = read_jsonl(chunks_path)
    tokenized = [tokenize(row.get("text", "")) for row in chunks]
    embed_texts = [
        "\n".join(part for part in (row.get("theorist", ""), row.get("title", ""), row.get("text", "")) if part).strip()
        for row in chunks
    ]

    meta_rows = []
    for row in chunks:
        meta_rows.append(
            {
                "chunk_id": row["chunk_id"],
                "doc_id": row["doc_id"],
                "theorist": row["theorist"],
                "title": row["title"],
                "source_path": row["source_path"],
                "ocr_path": row["ocr_path"],
                "page_start": row["page_start"],
                "page_end": row["page_end"],
                "text_hash": row["text_hash"],
                "text_len": len(row.get("text", "")),
                "text": row.get("text", ""),
            }
        )

    vectors = embedder_impl.encode(embed_texts) if embed_texts else np.zeros((0, spec.dim), dtype=np.float32)
    vectors = normalize_vectors(vectors.astype(np.float32))
    faiss = _faiss()
    dim_for_index = int(vectors.shape[1]) if vectors.size else int(spec.dim)
    faiss_index = faiss.IndexFlatIP(dim_for_index)
    if vectors.size:
        faiss_index.add(vectors)
    faiss.write_index(faiss_index, str(files["faiss"]))

    atomic_write_jsonl(files["meta"], meta_rows)
    atomic_write_json(files["bm25"], {"tokenized": tokenized})
    atomic_write_json(files["embedder"], embedder_json)
    return {"status": "built", "chunks": len(chunks), "backend": embedder_json["backend"]}


def load_index_artifacts(index_dir: Path) -> dict:
    files = _index_paths(index_dir)
    faiss = _faiss()
    meta_rows = read_jsonl(files["meta"])
    if files["bm25"].exists():
        tokenized = read_json(files["bm25"])["tokenized"]
    else:
        tokenized = [tokenize(row.get("text", "")) for row in meta_rows]
        try:
            atomic_write_json(files["bm25"], {"tokenized": tokenized})
        except OSError:
            pass
    return {
        "meta": meta_rows,
        "tokenized": tokenized,
        "embedder": read_json(files["embedder"]),
        "faiss_index": faiss.read_index(str(files["faiss"])),
    }
