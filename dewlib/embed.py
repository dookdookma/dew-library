from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass

import numpy as np

from .util import tokenize


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    if vectors.size == 0:
        return vectors.astype(np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (vectors / norms).astype(np.float32)


@dataclass
class EmbedderSpec:
    backend: str
    dim: int
    model_name: str | None


class HashEmbedder:
    backend = "hash"

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim
        self.model_name = None

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row_i, text in enumerate(texts):
            for token in tokenize(text):
                digest = hashlib.blake2b(token.encode("utf-8"), digest_size=16).digest()
                idx = int.from_bytes(digest[:8], "big") % self.dim
                sign = 1.0 if digest[8] % 2 == 0 else -1.0
                vectors[row_i, idx] += sign
        return normalize_vectors(vectors)


def _env_int(name: str) -> int | None:
    value = os.getenv(name)
    if value is None or not value.strip():
        return None
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an integer; got {value!r}") from exc
    if parsed < 1:
        raise ValueError(f"Environment variable {name} must be >= 1; got {parsed}")
    return parsed


def _default_st_threads() -> int:
    return _env_int("DEWLIB_ST_THREADS") or max(1, min(os.cpu_count() or 1, 8))


def _default_st_batch_size(device: str) -> int:
    configured = _env_int("DEWLIB_ST_BATCH_SIZE")
    if configured is not None:
        return configured
    return 256 if device.startswith("cpu") else 64


class SentenceTransformersEmbedder:
    backend = "sentence_transformers"

    def __init__(self, model_name: str, allow_download: bool = False) -> None:
        from sentence_transformers import SentenceTransformer
        import torch

        self.model_name = model_name
        self.model = SentenceTransformer(model_name, local_files_only=not allow_download)
        self.dim = int(self.model.get_sentence_embedding_dimension())
        self.device = str(self.model.device)
        self.batch_size = _default_st_batch_size(self.device)

        if self.device.startswith("cpu"):
            torch.set_num_threads(_default_st_threads())

    def encode(self, texts: list[str]) -> np.ndarray:
        arr = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return np.asarray(arr, dtype=np.float32)


def _hash_embedder(dim: int) -> tuple[HashEmbedder, EmbedderSpec]:
    embedder = HashEmbedder(dim=dim)
    return embedder, EmbedderSpec(backend="hash", dim=dim, model_name=None)


def _sentence_transformers_embedder(
    model_name: str,
    allow_download: bool,
) -> tuple[SentenceTransformersEmbedder, EmbedderSpec]:
    embedder = SentenceTransformersEmbedder(
        model_name=model_name,
        allow_download=allow_download,
    )
    return embedder, EmbedderSpec(
        backend="sentence_transformers",
        dim=embedder.dim,
        model_name=model_name,
    )


def build_embedder(
    backend: str = "auto",
    dim: int = 384,
    model_name: str = "all-MiniLM-L6-v2",
    allow_download: bool = False,
    fallback_to_hash: bool = False,
):
    normalized_backend = {
        "auto": "auto",
        "hash": "hash",
        "st": "sentence_transformers",
        "sentence_transformers": "sentence_transformers",
    }.get(backend, backend)

    if normalized_backend == "hash":
        return _hash_embedder(dim=dim)

    if normalized_backend == "auto":
        try:
            return _sentence_transformers_embedder(
                model_name=model_name,
                allow_download=allow_download,
            )
        except Exception:
            return _hash_embedder(dim=dim)

    if normalized_backend == "sentence_transformers":
        try:
            return _sentence_transformers_embedder(
                model_name=model_name,
                allow_download=allow_download,
            )
        except Exception:
            if fallback_to_hash:
                return _hash_embedder(dim=dim)
            raise

    raise ValueError(f"Unsupported embedder backend: {backend}")
