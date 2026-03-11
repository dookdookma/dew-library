# dew-library

Minimal, Railway-ready DEW theory library API.

This repo is runtime-only:
- no PDFs
- no OCR outputs
- no ETL pipeline
- no duplicate app/frontend code
- no committed `bm25_tokens.json`

Bundled runtime artifacts:
- `data/manifest.jsonl`
- `data/health_report.json`
- `data/pages/`
- `data/index/meta.jsonl`
- `data/index/faiss.index`
- `data/index/embedder.json`

Built into the Docker image at build time:
- `data/index/bm25_tokens.json`

## API surface

- `POST /search`
- `GET /chunk/{chunk_id}`
- `GET /doc/{doc_id}`
- `GET /doc/{doc_id}/pages`
- `GET /health/index`
- `GET /health/stats`

## Local run

```bash
python -m pip install -e .
python scripts/serve.py --host 127.0.0.1 --port 8787 --data-dir data
```

## Railway

Deploy from repo root.

Recommended Railway settings:
- Start command: use the Dockerfile
- Healthcheck path: `/health/index`
- Volume: optional, mount at `/data`

Behavior:
- the container seeds `/data` from bundled artifacts if the volume is empty
- the app preloads the search stack on startup
- `bm25_tokens.json` is generated during Docker build, so Railway does not need to derive it on first boot
- the sentence-transformers model `all-MiniLM-L6-v2` is downloaded during Docker build, not at request time
