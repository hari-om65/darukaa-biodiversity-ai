import chromadb
import pytest

from knowledge.ingestion.ingest import SOURCES_DIR, ingest_all, retrieve


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
