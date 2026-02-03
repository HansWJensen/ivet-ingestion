# src/chunker.py
"""
Structure-first chunker with adaptive semantic refinement and deterministic micro-headers.

Public API:
- chunk_document(doc: dict, embedding_service) -> List[Chunk]

DocumentCIR input example:
{
  "doc_id": "doc-123",
  "title": "Triage Guidelines",
  "clinical": {"species": ["dog"]},
  "sections": [
    {"id": "sec-1", "heading": "Overview", "ordinal": 0,
     "units": ["First sentence. Second sentence.", "Another unit."]},
    ...
  ],
  "doc_version": "1.0",
  "source_system": "importer",
  ...
}

Notes:
- Uses an embedding_service with method embed(text: str) -> List[float].
- Designed for unit testing with a deterministic mock embedder.
"""
from typing import List, Dict, Any
import hashlib
import re
import numpy as np

MAX_TOKENS_LEAF = 400
MIN_TOKENS_LEAF = 80
TAU_CONTINUE = 0.80

STOPWORDS = set([
    "the","and","is","in","of","to","a","for","with","on","that","it","as","are","this",
])


def _token_count(text: str) -> int:
    # coarse token estimate using whitespace splitting
    return max(1, len(text.split()))


def _first_sentence(text: str) -> str:
    m = re.search(r"([A-Z][^.?!]+[.?!])", text)
    if m:
        return m.group(1).strip()
    return text.strip().split("\n")[0][:120].strip()


def _keywords_from_text(text: str, k: int = 5) -> List[str]:
    words = re.findall(r"\w+", text.lower())
    freq: Dict[str, int] = {}
    for w in words:
        if w in STOPWORDS or len(w) < 3:
            continue
        freq[w] = freq.get(w, 0) + 1
    items = sorted(freq.items(), key=lambda x: (-x[1], x[0]))
    return [w for w, _ in items[:k]]


class Chunk:
    def __init__(self, point_id: str, parent_id: str, ordinal: int, micro_header: str, body: str, metadata: Dict[str, Any]):
        self.point_id = point_id
        self.parent_id = parent_id
        self.ordinal = ordinal
        self.micro_header = micro_header
        self.body = body
        self.metadata = metadata

    def to_point(self, embedding: List[float]) -> Dict[str, Any]:
        # Qdrant-like point representation for upsert
        return {
            "id": self.point_id,
            "vector": embedding,
            "payload": self.metadata
        }


def compute_text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def compute_point_id(root_id: str, doc_version: str, level: str, section_path: List[str], ordinal_in_parent: int, text_hash: str) -> str:
    serial = "|".join([root_id or "", doc_version or "", level or "", "/".join(section_path or []), str(ordinal_in_parent), text_hash])
    return hashlib.sha256(serial.encode("utf-8")).hexdigest()


def structure_first_split(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Return list of unit dicts:
    { section_id, section_path, ordinal_in_parent, text, section_heading, section_ordinal }
    """
    units = []
    for section in doc.get("sections", []):
        section_id = section.get("id")
        section_path = [doc.get("title", ""), section.get("heading", "")]
        ordinal = section.get("ordinal", 0)
        for idx, unit in enumerate(section.get("units", [])):
            # atomic_units are already pre-split (implementation assumes importer splits lists/tables)
            units.append({
                "section_id": section_id,
                "section_path": section_path,
                "ordinal_in_parent": idx,
                "text": unit,
                "section_heading": section.get("heading", ""),
                "section_ordinal": ordinal,
            })
    return units


def adaptive_refine(units: List[Dict[str, Any]], embed_fn, doc_meta: Dict[str, Any]) -> List[Chunk]:
    """
    Adaptive semantic refinement:
    - start from structure-first units
    - iteratively extend chunk while cosine similarity between successive unit embeddings >= TAU_CONTINUE
    - enforce MIN_TOKENS_LEAF and MAX_TOKENS_LEAF caps
    """
    refined: List[Chunk] = []
    i = 0
    chunk_ordinal = 0
    root_id = doc_meta.get("doc_id", "")
    doc_version = doc_meta.get("doc_version", "")
    species = (doc_meta.get("clinical", {}).get("species") or [None])[0]

    while i < len(units):
        current = units[i]
        chunk_units = [current]
        curr_emb = embed_fn(current["text"])
        j = i + 1
        while j < len(units) and _token_count(" ".join(u["text"] for u in chunk_units)) < MAX_TOKENS_LEAF:
            next_unit = units[j]
            next_emb = embed_fn(next_unit["text"])
            sim = cosine_sim(curr_emb, next_emb)
            if sim >= TAU_CONTINUE or _token_count(" ".join(u["text"] for u in chunk_units)) < MIN_TOKENS_LEAF:
                chunk_units.append(next_unit)
                # running average embedding
                curr_emb = [(a + b) / 2.0 for a, b in zip(curr_emb, next_emb)]
                j += 1
            else:
                break
        body = "\n".join(u["text"] for u in chunk_units)
        text_hash = compute_text_hash(body)
        section_path = chunk_units[0]["section_path"]
        point_id = compute_point_id(root_id, doc_version, "leaf", section_path, chunk_ordinal, text_hash)
        micro_header = generate_micro_header(section_path, chunk_units, species)
        metadata = {
            "ids": {
                "point_id": point_id,
                "root_id": root_id,
                "parent_id": chunk_units[0]["section_id"],
                "level": "leaf",
                "ordinal_in_parent": chunk_ordinal
            },
            "provenance": {
                "source_system": doc_meta.get("source_system", "unknown"),
                "source_uri": doc_meta.get("source_uri", ""),
                "ingested_at": doc_meta.get("ingested_at", ""),
                "doc_version": doc_meta.get("doc_version", ""),
                "doc_hash": doc_meta.get("doc_hash", ""),
                "pipeline_version": doc_meta.get("pipeline_version", "0.0.0"),
                "chunker_version": doc_meta.get("chunker_version", "0.0.0"),
                "embedding_model_id": doc_meta.get("embedding_model_id", "embedding-v1")
            },
            "security": {
                "tenant_id": doc_meta.get("tenant_id", "global"),
                "data_classification": doc_meta.get("data_classification", "public"),
                "session_id": doc_meta.get("session_id", None),
                "access_scope": doc_meta.get("access_scope", "shared_kb")
            },
            "clinical": doc_meta.get("clinical", {}),
            "text": {
                "section_path": section_path,
                "title": doc_meta.get("title", ""),
                "micro_header": micro_header,
                "chunk_body": body,
                "source_offsets": {
                    "start_char": 0,
                    "end_char": len(body)
                }
            }
        }
        refined.append(Chunk(point_id, chunk_units[0]["section_id"], chunk_ordinal, micro_header, body, metadata))
        chunk_ordinal += 1
        i = j
    return refined


def generate_micro_header(section_path: List[str], chunk_units: List[Dict[str, Any]], species: str = None) -> str:
    """
    Deterministic micro-header format:
    H: <section_path_short> | <topic label> | <species constraints if any>
    K: <3–8 keywords>
    S: <one-sentence gist, <= 25 words (approx)>
    """
    section_short = "/".join([p for p in section_path if p])[:40]
    heading = chunk_units[0].get("section_heading", "")
    first_def = _first_sentence(chunk_units[0]["text"])
    topic = (heading or first_def)[:40]
    keywords = _keywords_from_text(" ".join(u["text"] for u in chunk_units), k=6)
    gist = _first_sentence(" ".join(u["text"] for u in chunk_units))[:200]
    species_part = species or ""
    micro = f"H: {section_short} | {topic} | {species_part}\nK: {', '.join(keywords)}\nS: {gist}"
    return micro


def cosine_sim(a: List[float], b: List[float]) -> float:
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def chunk_document(doc: Dict[str, Any], embedding_service, params: Dict[str, Any] = None) -> List[Chunk]:
    """
    High-level pipeline: structure-first split -> adaptive refine -> return chunks
    """
    units = structure_first_split(doc)
    def embed_fn(text: str):
        return embedding_service.embed(text)
    chunks = adaptive_refine(units, embed_fn, doc_meta=doc)
    return chunks
