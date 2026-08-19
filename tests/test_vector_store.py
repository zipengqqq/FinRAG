import vector_store


def test_get_vector_store_connects_to_milvus(monkeypatch):
    monkeypatch.setattr(vector_store, "_vector_store", None)
    monkeypatch.setattr(vector_store, "get_embedding_model", lambda: object())

    store = vector_store.get_vector_store()

    assert store.collection_name == vector_store.COLLECTION_NAME
