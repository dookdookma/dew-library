from __future__ import annotations

import os
import threading
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np

from .embed import build_embedder, normalize_vectors
from .index import load_index_artifacts
from .util import normalize_match_text, read_json, tokenize


def _coverage(query_tokens: set[str], doc_tokens: set[str]) -> float:
    if not query_tokens:
        return 0.0
    return float(len(query_tokens.intersection(doc_tokens)) / len(query_tokens))


def _fuzzy_coverage(query_tokens: set[str], doc_tokens: set[str], threshold: float = 0.88) -> float:
    if not query_tokens or not doc_tokens:
        return 0.0

    matched = 0.0
    for query_token in query_tokens:
        if query_token in doc_tokens:
            matched += 1.0
            continue

        best = max(
            (SequenceMatcher(None, query_token, doc_token).ratio() for doc_token in doc_tokens),
            default=0.0,
        )
        if best >= threshold:
            matched += best
    return float(matched / len(query_tokens))


class SearchService:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        loaded = load_index_artifacts(data_dir / "index")
        self.meta: list[dict] = loaded["meta"]
        self.faiss_index = loaded["faiss_index"]
        self.embedder_info: dict = loaded["embedder"]
        self.bm25 = loaded.get("bm25")
        self.embedder = None
        self._embedder_lock = threading.Lock()
        self._embedder_loading = False
        self._embedder_error: str | None = None
        self._vector_query_enabled = (
            os.getenv("DEWLIB_ENABLE_VECTOR_QUERY", "1").strip().lower() not in {"0", "false", "no"}
        )

        if self.embedder_info["backend"] == "hash":
            self.embedder, _spec = build_embedder(
                backend=self.embedder_info["backend"],
                dim=int(self.embedder_info["dim"]),
                model_name=self.embedder_info.get("model_name") or "all-MiniLM-L6-v2",
                allow_download=False,
                fallback_to_hash=False,
            )
        elif self._vector_query_enabled:
            self._start_embedder_load()
        self._chunk_map = {row["chunk_id"]: row for row in self.meta}
        self.doc_first_chunk_page: dict[str, int] = {}
        self.theorist_to_indices: dict[str, list[int]] = {}
        for idx, row in enumerate(self.meta):
            doc_id = row["doc_id"]
            self.doc_first_chunk_page[doc_id] = min(
                row["page_start"],
                self.doc_first_chunk_page.get(doc_id, row["page_start"]),
            )
            self.theorist_to_indices.setdefault(row["theorist"], []).append(idx)
        self._row_feature_cache: list[dict | None] = [None] * len(self.meta)

    def _build_embedder(self):
        embedder, _spec = build_embedder(
            backend=self.embedder_info["backend"],
            dim=int(self.embedder_info["dim"]),
            model_name=self.embedder_info.get("model_name") or "all-MiniLM-L6-v2",
            allow_download=False,
            fallback_to_hash=False,
        )
        return embedder

    def _load_embedder_background(self) -> None:
        try:
            embedder = self._build_embedder()
        except Exception as exc:
            with self._embedder_lock:
                self._embedder_error = f"{type(exc).__name__}: {exc}"
                self._embedder_loading = False
            return
        with self._embedder_lock:
            self.embedder = embedder
            self._embedder_error = None
            self._embedder_loading = False

    def _start_embedder_load(self) -> None:
        if not self._vector_query_enabled or self.embedder_info["backend"] != "sentence_transformers":
            return
        with self._embedder_lock:
            if self.embedder is not None or self._embedder_loading:
                return
            self._embedder_loading = True
        thread = threading.Thread(target=self._load_embedder_background, daemon=True)
        thread.start()

    def semantic_status(self) -> dict[str, str | bool | None]:
        with self._embedder_lock:
            return {
                "vector_query_enabled": self._vector_query_enabled,
                "semantic_backend": self.embedder_info["backend"],
                "semantic_ready": self.embedder is not None,
                "semantic_loading": self._embedder_loading,
                "semantic_error": self._embedder_error,
            }

    def get_chunk(self, chunk_id: str) -> dict | None:
        return self._chunk_map.get(chunk_id)

    def get_doc_pages(self, doc_id: str, start: int, end: int) -> list[dict]:
        pages_file = self.data_dir / "pages" / f"{doc_id}.json"
        if not pages_file.exists():
            return []
        payload = read_json(pages_file)
        rows = []
        for page in payload.get("pages", []):
            page_num = int(page["page"])
            if start <= page_num <= end:
                rows.append({"page": page_num, "text": page.get("text", "")})
        return rows

    def _candidate_indices(self, theorist: str | None) -> list[int]:
        if theorist is None:
            return list(range(len(self.meta)))
        return list(self.theorist_to_indices.get(theorist, []))

    def _row_features(self, idx: int) -> dict:
        cached = self._row_feature_cache[idx]
        if cached is not None:
            return cached

        row = self.meta[idx]
        title_token_set = set(tokenize(row.get("title", "")))
        theorist_token_set = set(tokenize(row.get("theorist", "")))
        cached = {
            "text_token_set": set(tokenize(row.get("text", ""))),
            "title_token_set": title_token_set,
            "theorist_token_set": theorist_token_set,
            "metadata_token_set": title_token_set.union(theorist_token_set),
            "text_match_text": normalize_match_text(row.get("text", "")),
            "title_match_text": normalize_match_text(row.get("title", "")),
            "theorist_match_text": normalize_match_text(row.get("theorist", "")),
            "metadata_match_text": normalize_match_text(f"{row.get('theorist', '')} {row.get('title', '')}"),
        }
        self._row_feature_cache[idx] = cached
        return cached

    def _vector_candidate_pairs(
        self,
        query_vector: np.ndarray,
        candidate_set: set[int] | None,
        total: int,
        vector_k: int,
    ) -> list[tuple[int, float]]:
        fetch_k = min(total, max(vector_k * 4, 200))
        if candidate_set is not None:
            fetch_k = total

        scores, indices = self.faiss_index.search(query_vector, fetch_k)
        pairs: list[tuple[int, float]] = []
        seen: set[int] = set()
        for idx, score in zip(indices[0].tolist(), scores[0].tolist()):
            if idx < 0 or idx in seen:
                continue
            if candidate_set is not None and idx not in candidate_set:
                continue
            seen.add(idx)
            pairs.append((idx, float(score)))
            if len(pairs) >= vector_k:
                break
        return pairs

    def search(
        self,
        query: str,
        theorist: str | None = None,
        top_k: int = 8,
        bm25_k: int = 200,
        vector_k: int = 200,
    ) -> list[dict]:
        if not query.strip() or not self.meta:
            return []

        candidate_indices = self._candidate_indices(theorist)
        if not candidate_indices:
            return []

        total = len(self.meta)
        candidate_set = None if theorist is None else set(candidate_indices)
        bm25_k = min(len(candidate_indices), max(top_k * 12, bm25_k))
        vector_k = min(len(candidate_indices), max(top_k * 12, vector_k))

        query_tokens = tokenize(query)
        query_token_set = set(query_tokens)
        query_match_text = normalize_match_text(query)
        title_like_query = bool(query_token_set) and len(query_token_set) <= 8

        bm25_scores = (
            self.bm25.get_scores(query_tokens)
            if (self.bm25 is not None and query_tokens)
            else np.zeros(total, dtype=float)
        )
        bm25_order = sorted(
            candidate_indices,
            key=lambda idx: (-bm25_scores[idx], self.meta[idx]["chunk_id"]),
        )[:bm25_k]

        vector_pairs: list[tuple[int, float]] = []
        if self.embedder is not None:
            query_vector = self.embedder.encode([query])
            query_vector = normalize_vectors(np.asarray(query_vector, dtype=np.float32))
            vector_pairs = self._vector_candidate_pairs(
                query_vector=query_vector,
                candidate_set=candidate_set,
                total=total,
                vector_k=vector_k,
            )
        elif self.embedder_info["backend"] == "sentence_transformers":
            self._start_embedder_load()

        candidate_pool = set(bm25_order)
        candidate_pool.update(idx for idx, _score in vector_pairs)
        if not candidate_pool:
            candidate_pool.update(candidate_indices[:top_k])

        bm25_rank = {idx: rank for rank, idx in enumerate(bm25_order, start=1)}
        vector_rank = {idx: rank for rank, (idx, _score) in enumerate(vector_pairs, start=1)}
        vector_score_map = {idx: score for idx, score in vector_pairs}

        max_bm25 = max((bm25_scores[idx] for idx in candidate_pool), default=0.0)
        vector_values = [vector_score_map[idx] for idx in candidate_pool if idx in vector_score_map]
        max_vector = max(vector_values, default=0.0)
        min_vector = min(vector_values, default=0.0)

        scored: list[tuple[int, float, float, float]] = []
        for idx in candidate_pool:
            row = self.meta[idx]
            features = self._row_features(idx)
            text_token_set = features["text_token_set"]
            title_token_set = features["title_token_set"]
            theorist_token_set = features["theorist_token_set"]
            metadata_token_set = features["metadata_token_set"]
            text_len = int(row.get("text_len", len(row.get("text", ""))))
            author_mentioned = bool(theorist_token_set.intersection(text_token_set))
            text_cov = _coverage(query_token_set, text_token_set)
            title_cov = _coverage(query_token_set, title_token_set)
            theorist_cov = _coverage(query_token_set, theorist_token_set)
            metadata_cov = _coverage(query_token_set, metadata_token_set)
            fuzzy_text_cov = _fuzzy_coverage(query_token_set, text_token_set)
            fuzzy_metadata_cov = _fuzzy_coverage(query_token_set, metadata_token_set)

            exact_phrase_text = bool(query_match_text and query_match_text in features["text_match_text"])
            exact_phrase_title = bool(query_match_text and query_match_text in features["title_match_text"])
            exact_phrase_theorist = bool(query_match_text and query_match_text in features["theorist_match_text"])
            exact_phrase_metadata = bool(query_match_text and query_match_text in features["metadata_match_text"])

            bm25_norm = (float(bm25_scores[idx]) / max_bm25) if max_bm25 > 0 else 0.0
            if idx in vector_score_map:
                if max_vector > min_vector:
                    vector_norm = (vector_score_map[idx] - min_vector) / (max_vector - min_vector)
                else:
                    vector_norm = 1.0
            else:
                vector_norm = 0.0

            score = 0.0
            if idx in bm25_rank:
                score += 1.0 / (60.0 + bm25_rank[idx])
            if idx in vector_rank:
                score += 1.0 / (60.0 + vector_rank[idx])

            score += 0.45 * bm25_norm
            score += 0.20 * vector_norm
            score += 0.35 * text_cov
            score += 0.22 * title_cov
            score += 0.12 * theorist_cov
            score += 0.18 * metadata_cov
            score += 0.22 * fuzzy_text_cov
            score += 0.10 * fuzzy_metadata_cov

            if exact_phrase_text:
                score += 1.35
                if title_like_query:
                    score += 0.55
                if text_len <= 350:
                    score += 0.35
            if exact_phrase_title:
                score += 1.35
            if exact_phrase_theorist:
                score += 0.35
            if exact_phrase_metadata:
                score += 0.45

            first_chunk_page = self.doc_first_chunk_page.get(row["doc_id"], row["page_start"])
            if title_like_query and fuzzy_text_cov >= 0.95 and text_len <= 350:
                score += 0.95
                if author_mentioned:
                    score += 0.95
            if title_like_query and fuzzy_text_cov >= 0.9 and row["page_start"] <= first_chunk_page + 20:
                score += 0.24
            if title_like_query and text_cov >= 0.75 and row["page_start"] <= first_chunk_page + 20:
                score += 0.22
            if title_like_query and exact_phrase_text and row["page_start"] <= first_chunk_page + 20:
                score += 0.30
            if title_like_query and metadata_cov >= 0.5 and row["page_start"] == first_chunk_page:
                score += 0.12

            scored.append(
                (
                    idx,
                    score,
                    float(bm25_rank.get(idx, 10**9)),
                    float(vector_rank.get(idx, 10**9)),
                )
            )

        ranked = sorted(
            scored,
            key=lambda item: (-item[1], item[2], item[3], self.meta[item[0]]["chunk_id"]),
        )

        results: list[dict] = []
        for idx, score, _bm25_rank_value, _vector_rank_value in ranked[:top_k]:
            row = self.meta[idx]
            results.append(
                {
                    "score": float(score),
                    "chunk_id": row["chunk_id"],
                    "doc_id": row["doc_id"],
                    "theorist": row["theorist"],
                    "title": row["title"],
                    "source_path": row["source_path"],
                    "ocr_path": row["ocr_path"],
                    "page_start": row["page_start"],
                    "page_end": row["page_end"],
                    "excerpt": row.get("text", "")[:600],
                    "embedder_backend": self.embedder_info["backend"],
                    "semantic_ready": self.embedder is not None,
                }
            )

        return results



