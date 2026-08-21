from types import SimpleNamespace

import pytest
from pymilvus.exceptions import MilvusException

import vector_store


def test_get_vector_store_connects_to_milvus(monkeypatch):
    monkeypatch.setattr(vector_store, "_vector_store", None)
    monkeypatch.setattr(vector_store, "get_embedding_model", lambda: object())
    captured_kwargs = {}

    class FakeMilvus:
        def __init__(self, **kwargs):
            captured_kwargs.update(kwargs)
            self.collection_name = kwargs["collection_name"]

    monkeypatch.setattr(vector_store, "Milvus", FakeMilvus)

    store = vector_store.get_vector_store()

    assert store.collection_name == vector_store.COLLECTION_NAME
    assert captured_kwargs["auto_id"] is True
    assert captured_kwargs["consistency_level"] == "Bounded"


def test_init_collection_uses_generic_rag_schema(monkeypatch):
    field_definitions = []

    class FakeCollection:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def create_index(self, field_name, index_params):
            pass

        def load(self):
            pass

    def fake_field_schema(**kwargs):
        field_definitions.append(kwargs)
        return kwargs

    monkeypatch.setattr(vector_store.connections, "connect", lambda **kwargs: None)
    monkeypatch.setattr(vector_store, "FieldSchema", fake_field_schema)
    monkeypatch.setattr(vector_store, "CollectionSchema", lambda fields, description: fields)
    monkeypatch.setattr(vector_store, "Collection", FakeCollection)

    vector_store.init_collection()

    field_by_name = {field["name"]: field for field in field_definitions}

    assert vector_store.COLLECTION_NAME == "general_rag"
    assert set(field_by_name) == {
        "pk",
        "text",
        "vector",
        "document_id",
        "source",
        "section",
        "parent_id",
        "chunk_index",
        "chunk_count",
        "metadata",
    }
    assert "year" not in field_by_name
    assert field_by_name["source"]["max_length"] == 1024
    assert field_by_name["section"]["max_length"] == 2048
    assert field_by_name["document_id"]["max_length"] == 64
    assert field_by_name["parent_id"]["max_length"] == 64
    assert field_by_name["chunk_index"]["dtype"] == vector_store.DataType.INT32
    assert field_by_name["chunk_count"]["dtype"] == vector_store.DataType.INT32
    assert field_by_name["metadata"]["dtype"] == vector_store.DataType.JSON


def test_init_collection_creates_and_loads_ip_hnsw_index(monkeypatch):
    calls = []

    class FakeCollection:
        def __init__(self, **kwargs):
            calls.append(("create_collection", kwargs))

        def create_index(self, field_name, index_params):
            calls.append(("create_index", field_name, index_params))

        def load(self):
            calls.append(("load",))

    monkeypatch.setattr(vector_store.connections, "connect", lambda **kwargs: None)
    monkeypatch.setattr(vector_store, "FieldSchema", lambda **kwargs: kwargs)
    monkeypatch.setattr(vector_store, "CollectionSchema", lambda fields, description: fields)
    monkeypatch.setattr(vector_store, "Collection", FakeCollection)

    vector_store.init_collection()

    assert (
        "create_index",
        "vector",
        {
            "index_type": "HNSW",
            "metric_type": "IP",
            "params": {"M": 8, "efConstruction": 64},
        },
    ) in calls
    assert ("load",) in calls


@pytest.mark.parametrize(
    ("metadata_key", "value"),
    [
        ("source", "中" * 342),
        ("section", "中" * 683),
        ("document_id", "a" * 65),
        ("parent_id", "a" * 65),
    ],
)
def test_add_documents_skips_chunk_with_oversized_required_metadata(
    monkeypatch, metadata_key, value
):
    stored_batches = []
    valid_chunk = SimpleNamespace(
        metadata={
            "source": "report.md",
            "section": "财务报告",
            "document_id": "document-1",
            "parent_id": "parent-1",
        }
    )
    invalid_metadata = valid_chunk.metadata | {metadata_key: value}
    oversized_chunk = SimpleNamespace(metadata=invalid_metadata)

    class FakeVectorStore:
        def add_documents(self, chunks):
            stored_batches.append(chunks)

    monkeypatch.setattr(vector_store.connections, "connect", lambda **kwargs: None)
    monkeypatch.setattr(vector_store.utility, "has_collection", lambda name: True)
    monkeypatch.setattr(vector_store, "get_vector_store", lambda: FakeVectorStore())

    vector_store.add_documents_to_milvus([valid_chunk, oversized_chunk])

    assert stored_batches == [[valid_chunk]]


def test_add_documents_retries_failed_batch_one_chunk_at_a_time(monkeypatch):
    stored_chunks = []
    valid_chunk = SimpleNamespace(metadata={"section": "财务报告"})
    invalid_chunk = SimpleNamespace(metadata={"section": "异常记录"})

    class FakeVectorStore:
        def add_documents(self, chunks):
            if len(chunks) > 1 or chunks[0] is invalid_chunk:
                raise MilvusException(code=1100, message="invalid field")
            stored_chunks.extend(chunks)

    monkeypatch.setattr(vector_store.connections, "connect", lambda **kwargs: None)
    monkeypatch.setattr(vector_store.utility, "has_collection", lambda name: True)
    monkeypatch.setattr(vector_store, "get_vector_store", lambda: FakeVectorStore())

    vector_store.add_documents_to_milvus([valid_chunk, invalid_chunk])

    assert stored_chunks == [valid_chunk]
