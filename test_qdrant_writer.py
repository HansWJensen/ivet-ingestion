# tests/test_qdrant_writer.py
from src.qdrant_writer import QdrantWriter, MockQdrantClient
from src.embedding_service import MockEmbedder
from src.chunker import compute_text_hash, compute_point_id

def test_upsert_and_get_point():
    client = MockQdrantClient()
    writer = QdrantWriter(client=client, collection_name="test_coll", schema_path="metadata.schema.json")
    # prepare a fake chunk point
    body = "This is a test chunk body."
    tid = "root"
    text_hash = compute_text_hash(body)
    pid = compute_point_id(tid, "v1", "leaf", ["Doc","Sec"], 0, text_hash)
    point = {"id": pid, "vector": [0.1]*8, "payload": {"ids": {"point_id": pid, "level": "leaf"}, "security": {"tenant_id": "t1", "access_scope": "shared_kb"}, "clinical": {"species": ["dog"]}, "text": {"section_path": ["Doc","Sec"]}}}
    resp = writer.upsert_points([point])
    assert resp["status"] == "ok"
    stored = client.get_point("test_coll", pid)
    assert stored is not None
    assert stored["id"] == pid
    assert "payload" in stored
    assert stored["payload"]["tenant_id"] == "t1"
