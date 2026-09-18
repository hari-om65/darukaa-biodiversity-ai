import subprocess
import sys
from pathlib import Path

import chromadb
import pytest

from knowledge.ingestion.ingest import SOURCES_DIR, ingest_all, retrieve

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_import_does_not_eagerly_load_embedding_model():
    """Importing knowledge.ingestion.ingest must not load the embedding
    model's weights - only get_embedder() (called from ingest_file()/
    retrieve()) should, so app startup stays cheap on memory-constrained
    deployments.

    Run in a fresh subprocess: within this pytest session, other tests will
    already have triggered the lazy load and populated the module-level
    cache, which would make an in-process check depend on test order.
    """
    script = (
        "import knowledge.ingestion.ingest as m; "
        "assert m._embedder is None, 'model was loaded eagerly at import time'; "
        "print('OK')"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


@pytest.fixture(scope="module")
def chroma_client(tmp_path_factory):
    tmp_dir = tmp_path_factory.mktemp("chroma_test")
    client = chromadb.PersistentClient(path=str(tmp_dir))
    ingested = ingest_all(source_dir=SOURCES_DIR, client=client)
    assert ingested > 0
    return client


def test_retrieve_returns_relevant_chunks(chroma_client):
    results = retrieve("soil organic carbon and microbial life", k=2, client=chroma_client)

    assert len(results) > 0
    assert results[0]["metadata"]["source"] == "soil_organic_carbon.md"


def test_retrieve_with_tag_filter(chroma_client):
    results = retrieve("ecosystem patterns", tags=["water"], k=4, client=chroma_client)

    assert len(results) > 0
    assert all(r["metadata"]["tag_water"] for r in results)
