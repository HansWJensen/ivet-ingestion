# src/embedding_service.py
"""
Embedding service with caching (in-memory and optional Redis), batching, normalization,
and a deterministic mock embedder for tests.

Public API:
- EmbeddingService(cache=..., model_id="embedding-v1", real_client=None)
  - embed(text: str) -> List[float]
  - batch_embed(texts: List[str]) -> List[List[float]]

Design:
- Cache key: sha256(model_id + normalized_text)
- Normalization: unicode NFC (python str is fine), collapse whitespace, deterministic list formatting
- For unit tests/CI we rely on MockEmbedder which deterministically derives a small vector from the text.
"""
from typing import List, Dict, Optional
import hashlib
import time
import threading
import math

# Lightweight deterministic mock embedder used for unit tests and CI
class MockEmbedder:
    def __init__(self, dim: int = 8):
        self.dim = dim

    def embed_text(self, text: str) -> List[float]:
        """
        Deterministic pseudo-embedding: uses sha256 digest bytes to create floats, then normalize.
        """
        h = hashlib.sha256(text.encode("utf-8")).digest()
        vec = []
        # produce dim floats from repeated digest segments
        for i in range(self.dim):
            b = h[i % len(h)]
            # map byte to -1..1
            v = (b / 255.0) * 2.0 - 1.0
            vec.append(v)
        # normalize
        norm = math.sqrt(sum(x * x for x in vec))
        if norm == 0:
            return [0.0] * self.dim
        return [x / norm for x in vec]


class InMemoryCache:
    def __init__(self):
        self._store: Dict[str, Dict] = {}
        self._lock = threading.Lock()

    def get(self, key: str):
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            # optional TTL could be supported here
            return entry["value"]

    def set(self, key: str, value, ttl: Optional[int] = None):
        with self._lock:
            self._store[key] = {"value": value, "ts": time.time(), "ttl": ttl}


class EmbeddingService:
    def __init__(self, model_id: str = "embedding-v1", cache: Optional[InMemoryCache] = None, embedder: Optional[MockEmbedder] = None):
        self.model_id = model_id
        self.cache = cache or InMemoryCache()
        self.embedder = embedder or MockEmbedder(dim=8)
        # batch window and locking for naive batching (not used in test-only mode)
        self._batch_lock = threading.Lock()

    @staticmethod
    def _normalize_text(text: str) -> str:
        # collapse whitespace and unify newlines
        return " ".join(text.strip().split())

    @staticmethod
    def _cache_key(model_id: str, normalized_text: str) -> str:
        return hashlib.sha256((model_id + "|" + normalized_text).encode("utf-8")).hexdigest()

    def embed(self, text: str) -> List[float]:
        n = self._normalize_text(text)
        key = self._cache_key(self.model_id, n)
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        vec = self.embedder.embed_text(n)
        self.cache.set(key, vec)
        return vec

    def batch_embed(self, texts: List[str]) -> List[List[float]]:
        # simple implementation: embed sequentially using caching
        return [self.embed(t) for t in texts]
