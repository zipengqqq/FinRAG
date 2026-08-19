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

    def fake_field_schema(**kwargs):
        field_definitions.append(kwargs)
        return kwargs

    monkeypatch.setattr(vector_store.connections, "connect", lambda **kwargs: None)
    monkeypatch.setattr(vector_store, "FieldSchema", fake_field_schema)
    monkeypatch.setattr(vector_store, "CollectionSchema", lambda fields, description: fields)
    monkeypatch.setattr(vector_store, "Collection", lambda **kwargs: kwargs)

    vector_store.init_collection()

    section_field = next(field for field in field_definitions if field["name"] == "section")
    assert section_field["max_length"] == 1024
