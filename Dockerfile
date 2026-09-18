FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# build-essential covers platforms where chromadb's native dependencies
# (e.g. hnswlib) have no prebuilt wheel and must compile from source.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Bake the knowledge base into the image at build time (chunks, embeds with
# paraphrase-MiniLM-L3-v2, and stores everything in knowledge/chroma_db) so
# the container starts ready to serve recommendations - no separate init
# step needed at runtime. Re-run automatically on every build, so changes to
# knowledge/sources/ take effect on the next deploy.
RUN python -m knowledge.ingestion.ingest

EXPOSE 8000

# app/main.py's __main__ block binds 0.0.0.0 and reads $PORT (Render/Railway
# inject their own; defaults to 8000 otherwise) - see app/main.py.
CMD ["python", "-m", "app.main"]
