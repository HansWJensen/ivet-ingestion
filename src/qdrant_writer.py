# src/qdrant_writer.py
"""
Qdrant writer skeleton:
- compute_point_id for stable ids
- validate metadata using jsonschema (metadata.schema.json expected in repo root)
- idempotent upsert semantics: we expose upsert_points which accepts list of Chunk.to_point-like dicts
- uses a pluggable client; for tests a MockQdrantClient is provided

Note: This module avoids a hard dependency on qdrant-client for unit tests; if you have qdrant-client
and a running Qdrant instance, you can plug the real client.
"""
from typing import List, Dict, Any, Optional
import json
import hashlib
import os

try:
    import jsonschema
except Exception:
    jsonschema = None  # tests will mock validation where needed


def compute_point_id(root_id: str, doc_version: str, level: str, section_path: List[str], ordinal_in_parent: int, text_hash: str) -> str:
    serial = "|".join([root_id or "", doc_version or "", level or "", "/".join(section_path or []), str(ordinal_in_parent), text_hash])
    return hashlib.sha256(serial.encode("utf-8")).hexdigest()


class ValidationError(Exception):
    pass


class MockQdrantClient:
    """
    Simple in-memory client used for unit tests.
    Stores points per collection in a dict.
    """
    def __init__(self):
        self.storage: Dict[str, Dict[str, Dict[str, Any]]] = {}

    def upsert(self, collection_name: str, points: List[Dict[str, Any]]):
        coll = self.storage.setdefault(collection_name, {})
        for p in points:
            coll[p["id"]] = p
        return {"status": "ok", "upserted": len(points)}

    def get_point(self, collection_name: str, point_id: str) -> Optional[Dict[str, Any]]:
        return self.storage.get(collection_name, {}).get(point_id)


class QdrantWriter:
    def __init__(self, client=None, collection_name: str = "vet_kb_dense_v1", schema_path: Optional[str] = "metadata.schema.json"):
        self.client = client or MockQdrantClient()
        self.collection_name = collection_name
        self.schema_path = schema_path
        self._schema = None
        if jsonschema and os.path.exists(schema_path):
            with open(schema_path, "r", encoding="utf-8") as f:
                self._schema = json.load(f)

    def validate_metadata(self, metadata: Dict[str, Any]):
        if jsonschema and self._schema:
            try:
                jsonschema.validate(instance=metadata, schema=self._schema)
            except Exception as e:
                raise ValidationError(str(e))
        # if jsonschema not installed or schema missing, skip strict validation (unit tests rely on this)
        return True

    def prepare_point(self, chunk_point: Dict[str, Any]) -> Dict[str, Any]:
        """
        Accepts point dicts with keys: id, vector, payload
        Ensures payload contains top-level filter fields
        """
        payload = chunk_point.get("payload", {})
        # flatten important fields to top-level keys in payload for easier indexing
        payload_flat = payload.copy()
        # Copy notable fields from payload->security/clinical/text
        try:
            payload_flat["tenant_id"] = payload["security"]["tenant_id"]
        except Exception:
            payload_flat["tenant_id"] = payload_flat.get("security", {}).get("tenant_id", "global")
        try:
            payload_flat["access_scope"] = payload["security"]["access_scope"]
        except Exception:
            payload_flat["access_scope"] = payload_flat.get("security", {}).get("access_scope", "shared_kb")
        try:
            payload_flat["species"] = payload["clinical"]["species"]
        except Exception:
            payload_flat["species"] = payload_flat.get("clinical", {}).get("species", [])
        try:
            payload_flat["level"] = payload["ids"]["level"]
        except Exception:
            payload_flat["level"] = payload_flat.get("ids", {}).get("level", "")
        prepared = {
            "id": chunk_point["id"],
            "vector": chunk_point.get("vector"),
            "payload": payload_flat
        }
        return prepared

    def upsert_points(self, points: List[Dict[str, Any]]):
        """
        points: list of {id, vector, payload}
        Validates payloads then upserts to client
        """
        prepared = []
        for p in points:
            # validate payload (best-effort)
            try:
                self.validate_metadata(p.get("payload", {}))
            except ValidationError as e:
                # in real pipeline: move to quarantine or log error. For tests, raise.
                raise
            prepared.append(self.prepare_point(p))
        return self.client.upsert(self.collection_name, prepared)
