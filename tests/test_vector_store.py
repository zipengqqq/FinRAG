from types import SimpleNamespace

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


def test_init_collection_allows_long_utf8_section_metadata(monkeypatch):
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

    section_field = next(field for field in field_definitions if field["name"] == "section")
    assert section_field["max_length"] == 1024


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


def test_add_documents_skips_chunk_with_oversized_section(monkeypatch):
    stored_batches = []
    valid_chunk = SimpleNamespace(metadata={"section": "财务报告"})
    oversized_chunk = SimpleNamespace(metadata={"section": "中" * 342})

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
