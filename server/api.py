from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from dewlib.config import Paths
from dewlib.manifest import load_manifest
from dewlib.search import SearchService
from dewlib.util import read_json

app = FastAPI(title="DEW Library API", version="1.0")
_startup_error: str | None = None


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    theorist: str | None = None
    top_k: int = Field(default=8, ge=1, le=50)


def _paths() -> Paths:
    return Paths.from_args(
        data_dir=os.getenv("DEW_DATA_DIR", "data"),
        manifest_path=os.getenv("DEW_MANIFEST_PATH"),
    )


@lru_cache(maxsize=1)
def _search_service() -> SearchService:
    cfg = _paths()
    return SearchService(data_dir=cfg.data_dir)


@lru_cache(maxsize=1)
def _manifest_by_doc() -> dict[str, dict]:
    cfg = _paths()
    rows = load_manifest(cfg.manifest_path)
    return {row["doc_id"]: row for row in rows}


@lru_cache(maxsize=1)
def _health_flags_by_doc() -> dict[str, list[str]]:
    cfg = _paths()
    report_path = cfg.health_report_path
    if not report_path.exists():
        return {}
    report = read_json(report_path)
    mapping = {}
    for doc in report.get("docs", []):
        mapping[doc["doc_id"]] = doc.get("flags", [])
    return mapping


@app.post("/search")
def search(request: SearchRequest) -> list[dict]:
    try:
        service = _search_service()
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=503, detail=f"Search index unavailable: {exc}") from exc
    return service.search(query=request.query, theorist=request.theorist, top_k=request.top_k)


@app.get("/chunk/{chunk_id}")
def get_chunk(chunk_id: str) -> dict:
    try:
        service = _search_service()
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=503, detail=f"Search index unavailable: {exc}") from exc

    row = service.get_chunk(chunk_id)
    if not row:
        raise HTTPException(status_code=404, detail=f"Chunk not found: {chunk_id}")
    return row


@app.get("/doc/{doc_id}")
def get_doc(doc_id: str) -> dict:
    row = _manifest_by_doc().get(doc_id)
    if not row:
        raise HTTPException(status_code=404, detail=f"Document not found: {doc_id}")
    return {
        **row,
        "health_flags": _health_flags_by_doc().get(doc_id, []),
    }


@app.get("/doc/{doc_id}/pages")
def get_doc_pages(
    doc_id: str,
    start: int = Query(1, ge=1),
    end: int | None = Query(None, ge=1),
) -> dict:
    try:
        service = _search_service()
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=503, detail=f"Search index unavailable: {exc}") from exc

    if end is None:
        end = start
    if end < start:
        raise HTTPException(status_code=400, detail="Invalid range: end must be >= start")

    manifest = _manifest_by_doc()
    if doc_id not in manifest:
        raise HTTPException(status_code=404, detail=f"Document not found: {doc_id}")
    if not (_paths().pages_dir / f"{doc_id}.json").exists():
        raise HTTPException(status_code=404, detail="Page texts are not bundled in this deployment.")
    pages = service.get_doc_pages(doc_id=doc_id, start=start, end=end)
    return {"doc_id": doc_id, "start": start, "end": end, "pages": pages}


@app.on_event("startup")
def _warm_runtime() -> None:
    global _startup_error
    try:
        _manifest_by_doc()
        _health_flags_by_doc()
        _search_service()
        _startup_error = None
    except Exception as exc:
        _startup_error = f"{type(exc).__name__}: {exc}"
        raise


@app.get("/health/index")
def health_index() -> dict:
    cfg = _paths()
    checks = {
        "data_dir": str(cfg.data_dir),
        "manifest_path": str(cfg.manifest_path),
        "exists_manifest": cfg.manifest_path.exists(),
        "exists_meta": (cfg.index_dir / "meta.jsonl").exists(),
        "exists_faiss": (cfg.index_dir / "faiss.index").exists(),
        "exists_embedder": (cfg.index_dir / "embedder.json").exists(),
        "exists_bm25_cache": (cfg.index_dir / "bm25_tokens.json").exists(),
        "ready": _startup_error is None,
        "startup_error": _startup_error,
    }
    if not checks["ready"]:
        raise HTTPException(status_code=503, detail=checks)
    return checks


@app.get("/health/stats")
def health_stats() -> dict:
    cfg = _paths()
    embedder_cfg = read_json(cfg.index_dir / "embedder.json") if (cfg.index_dir / "embedder.json").exists() else {}
    return {
        "manifest_size": cfg.manifest_path.stat().st_size if cfg.manifest_path.exists() else 0,
        "health_report_size": cfg.health_report_path.stat().st_size if cfg.health_report_path.exists() else 0,
        "meta_size": (cfg.index_dir / "meta.jsonl").stat().st_size if (cfg.index_dir / "meta.jsonl").exists() else 0,
        "faiss_size": (cfg.index_dir / "faiss.index").stat().st_size if (cfg.index_dir / "faiss.index").exists() else 0,
        "bm25_cache_size": (cfg.index_dir / "bm25_tokens.json").stat().st_size if (cfg.index_dir / "bm25_tokens.json").exists() else 0,
        "pages_dir_exists": cfg.pages_dir.exists(),
        "embedder_backend": embedder_cfg.get("backend"),
        "embedder_model_name": embedder_cfg.get("model_name"),
        "doc_count": len(_manifest_by_doc()),
        "chunk_count": len(_search_service().meta),
    }
