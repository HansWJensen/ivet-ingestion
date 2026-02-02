# tests/test_integration_ingest_upsert.py
"""
Integration-style test using in-process mocks:
- chunk a sample document
- embed chunks with the EmbeddingService (mock embedder)
- upsert into MockQdrantClient via QdrantWriter
- verify points exist
"""
from src.chunker import chunk_document
from src.embedding_service import EmbeddingService, MockEmbedder
from src.qdrant_writer import QdrantWriter, MockQdrantClient

def test_end_to_end_ingest_upsert():
    doc = {
        "doc_id": "doc-e2e",
        "title": "Simple Doc",
        "clinical": {"species": ["dog"]},
        "sections": [
            {"id": "s1", "heading": "One", "ordinal": 0, "units": ["A paragraph about risk.", "Another sentence."]}
        ],
        "doc_version": "1.0",
        "source_system": "test",
        "ingested_at": "2026-02-02T00:00:00Z",
        "doc_hash": "abc",
        "pipeline_version": "0.1.0",
        "chunker_version": "0.1.0",
        "embedding_model_id": "mock-embedder-v1",
    }
    emb_service = EmbeddingService(embedder=MockEmbedder(dim=8))
    chunks = chunk_document(doc, emb_service)
    client = MockQdrantClient()
    writer = QdrantWriter(client=client, collection_name="integration_test")
    points = [c.to_point(emb_service.embed(c.body)) for c in chunks]
    resp = writer.upsert_points(points)
    assert resp["status"] == "ok"
    # verify stored
    for p in points:
        stored = client.get_point("integration_test", p["id"])
        assert stored is not None
