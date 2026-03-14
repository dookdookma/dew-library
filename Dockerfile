FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1
ENV DEW_DATA_DIR=/app/data
ENV HF_HOME=/models/huggingface
ENV SENTENCE_TRANSFORMERS_HOME=/models/sentence-transformers
ENV DEW_SENTENCE_MODEL=all-MiniLM-L6-v2

COPY pyproject.toml README.md ./
COPY dewlib ./dewlib
COPY server ./server
COPY start.sh ./start.sh
COPY data ./data

RUN mkdir -p "$HF_HOME" "$SENTENCE_TRANSFORMERS_HOME" && \
    chmod +x /app/start.sh && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir . && \
    python -c "import os; from sentence_transformers import SentenceTransformer; SentenceTransformer(os.environ['DEW_SENTENCE_MODEL'], cache_folder=os.environ['SENTENCE_TRANSFORMERS_HOME'])" && \
    python -c "from pathlib import Path; from dewlib.index import build_runtime_caches; from dewlib.util import atomic_write_json, read_jsonl, tokenize; meta = read_jsonl(Path('/app/data/index/meta.jsonl')); atomic_write_json(Path('/app/data/index/bm25_tokens.json'), {'tokenized': [tokenize(row.get('text', '')) for row in meta]}); build_runtime_caches(Path('/app/data/index'))"

EXPOSE 8080
CMD ["./start.sh"]
