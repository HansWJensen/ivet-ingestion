# tests/test_embedding_cache.py
from src.embedding_service import EmbeddingService, InMemoryCache, MockEmbedder

def test_cache_hits_and_misses():
    cache = InMemoryCache()
    emb = EmbeddingService(model_id="m1", cache=cache, embedder=MockEmbedder(dim=4))
    v1 = emb.embed("Hello world")
    v2 = emb.embed("Hello world ")
    # normalization should make these equal -> cache hit
    assert v1 == v2
    # different text => different vector (very likely)
    v3 = emb.embed("Completely different")
    assert v3 != v1
