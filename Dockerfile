FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1
ENV DEW_DATA_DIR=/data
ENV HF_HOME=/models/huggingface

COPY pyproject.toml README.md ./
COPY dewlib ./dewlib
COPY server ./server
COPY scripts ./scripts
COPY start.sh ./start.sh
COPY data ./data

RUN chmod +x /app/start.sh &&     pip install --no-cache-dir --upgrade pip &&     pip install --no-cache-dir . &&     python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')" &&     python -c "from pathlib import Path; from dewlib.util import atomic_write_json, read_jsonl, tokenize; meta = read_jsonl(Path('/app/data/index/meta.jsonl')); atomic_write_json(Path('/app/data/index/bm25_tokens.json'), {'tokenized': [tokenize(row.get('text', '')) for row in meta]})"

EXPOSE 8080
CMD ["./start.sh"]
