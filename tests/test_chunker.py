# tests/test_chunker.py
import pytest
from src.chunker import chunk_document, compute_text_hash, Chunk
from src.embedding_service import EmbeddingService, MockEmbedder

def make_sample_doc():
    return {
        "doc_id": "doc-1",
        "title": "Tox Protocol",
        "clinical": {"species": ["dog"]},
        "sections": [
            {
                "id": "s1",
                "heading": "Signs",
                "ordinal": 0,
                "units": [
                    "Vomiting is common. Animal may be lethargic.",
                    "Profuse salivation may occur.",
                ],
            },
            {
                "id": "s2",
                "heading": "Treatment",
                "ordinal": 1,
                "units": [
                    "If ingestion within 1 hour, induce emesis.",
                    "Administer activated charcoal if indicated.",
                    "Monitor vitals hourly."
                ],
            }
        ],
        "doc_version": "1.0",
        "source_system": "test",
        "ingested_at": "2026-02-02T00:00:00Z",
        "doc_hash": "deadbeef",
        "pipeline_version": "0.1.0",
        "chunker_version": "0.1.0",
        "embedding_model_id": "mock-embedder-v1",
    }

def test_chunk_counts_and_headers():
    emb = EmbeddingService(embedder=MockEmbedder(dim=8))
    doc = make_sample_doc()
    chunks = chunk_document(doc, emb)
    # Expect chunks (adaptive) not to split section boundaries; ensure at least 1 chunk per section
    section_ids = set(c.parent_id for c in chunks)
    assert "s1" in section_ids
    assert "s2" in section_ids
    # micro_header present and contains H:, K:, S:
    for c in chunks:
        assert isinstance(c.micro_header, str)
        assert "H:" in c.micro_header and "K:" in c.micro_header and "S:" in c.micro_header

def test_point_id_stability():
    emb = EmbeddingService(embedder=MockEmbedder(dim=8))
    doc = make_sample_doc()
    chunks_a = chunk_document(doc, emb)
    chunks_b = chunk_document(doc, emb)
    ids_a = [c.point_id for c in chunks_a]
    ids_b = [c.point_id for c in chunks_b]
    assert ids_a == ids_b  # stable across runs for same doc and config
