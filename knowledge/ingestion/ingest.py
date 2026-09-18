"""Ingest .txt/.md source documents from knowledge/sources into ChromaDB.

Each source file may start with a YAML frontmatter block declaring metadata:

    ---
    title: Some Title
    year: 2021
    topic_tags: [soil, biodiversity]
    ---

    Body text...

Documents are chunked (~500 words with overlap, as a token-count proxy),
embedded with the sentence-transformers "all-MiniLM-L6-v2" model, and stored
in the ChromaDB collection "biodiversity_knowledge".
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import chromadb
import yaml
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent.parent
SOURCES_DIR = BASE_DIR / "sources"
# Override with CHROMA_DB_DIR to point at a persistent volume in production.
CHROMA_DIR = Path(os.environ.get("CHROMA_DB_DIR", str(BASE_DIR / "chroma_db")))
COLLECTION_NAME = "biodiversity_knowledge"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

ALLOWED_TAGS = ["soil", "water", "land_use", "biodiversity", "climate", "human_impact"]

# Chunk size/overlap are expressed in words as a simple proxy for tokens,
# avoiding an extra tokenizer dependency.
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)$", re.DOTALL)

_embedder: SentenceTransformer | None = None


def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedder


def get_client(persist_directory: str | Path = CHROMA_DIR) -> ClientAPI:
    return chromadb.PersistentClient(path=str(persist_directory))


def get_collection(client: ClientAPI | None = None) -> Collection:
    client = client or get_client()
    return client.get_or_create_collection(COLLECTION_NAME)


def parse_document(path: Path) -> tuple[dict[str, Any], str]:
    """Split a source file into (frontmatter metadata, body text)."""
    raw = path.read_text(encoding="utf-8")
    match = _FRONTMATTER_RE.match(raw)
    if not match:
        return {}, raw.strip()
    frontmatter, body = match.groups()
    metadata = yaml.safe_load(frontmatter) or {}
    return metadata, body.strip()


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into ~chunk_size-word chunks with overlap between consecutive chunks."""
    words = text.split()
    if not words:
        return []
    step = max(chunk_size - overlap, 1)
    chunks = []
    for start in range(0, len(words), step):
        chunk_words = words[start : start + chunk_size]
        chunks.append(" ".join(chunk_words))
        if start + chunk_size >= len(words):
            break
    return chunks


def build_metadata(doc_metadata: dict[str, Any], source: str) -> dict[str, Any]:
    tags = [t for t in doc_metadata.get("topic_tags", []) if t in ALLOWED_TAGS]
    metadata: dict[str, Any] = {
        "source": source,
        "year": int(doc_metadata.get("year", 0)),
        "topic_tags": ",".join(tags),
    }
    # Chroma metadata values must be scalars, so each allowed tag is stored
    # as its own boolean field to support "where" filtering by tag.
    for tag in ALLOWED_TAGS:
        metadata[f"tag_{tag}"] = tag in tags
    return metadata


def ingest_file(path: Path, collection: Collection) -> int:
    doc_metadata, body = parse_document(path)
    chunks = chunk_text(body)
    if not chunks:
        return 0

    embedder = get_embedder()
    embeddings = embedder.encode(chunks).tolist()
    metadatas = [build_metadata(doc_metadata, path.name) for _ in chunks]
    ids = [f"{path.stem}-{i}" for i in range(len(chunks))]

    collection.upsert(ids=ids, embeddings=embeddings, documents=chunks, metadatas=metadatas)
    return len(chunks)


def ingest_all(source_dir: Path = SOURCES_DIR, client: ClientAPI | None = None) -> int:
    """Ingest every .txt/.md file in source_dir. Returns the number of chunks stored."""
    collection = get_collection(client)
    total = 0
    for path in sorted(Path(source_dir).glob("*")):
        if path.suffix.lower() in {".txt", ".md"}:
            total += ingest_file(path, collection)
    return total


def retrieve(
    query: str,
    tags: list[str] | None = None,
    k: int = 4,
    client: ClientAPI | None = None,
) -> list[dict[str, Any]]:
    """Retrieve the top-k chunks most relevant to `query`.

    If `tags` is given, results are restricted to chunks whose source document
    was tagged with at least one of the given topic tags.
    """
    collection = get_collection(client)
    embedder = get_embedder()
    query_embedding = embedder.encode([query]).tolist()

    where = None
    if tags:
        conditions = [{f"tag_{t}": True} for t in tags if t in ALLOWED_TAGS]
        if len(conditions) == 1:
            where = conditions[0]
        elif len(conditions) > 1:
            where = {"$or": conditions}

    results = collection.query(query_embeddings=query_embedding, n_results=k, where=where)

    ids = results.get("ids", [[]])[0]
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    return [
        {"id": id_, "text": doc, "metadata": meta, "distance": dist}
        for id_, doc, meta, dist in zip(ids, documents, metadatas, distances)
    ]


def main() -> None:
    count = ingest_all()
    print(f"Ingested {count} chunks into '{COLLECTION_NAME}'.")


if __name__ == "__main__":
    main()
