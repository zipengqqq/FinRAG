# Milvus IP Index Initialization Design

## Goal

Ensure a newly created `financial_rag` collection uses an IP vector index before
documents are inserted, so retrieval with normalized BGE-M3 embeddings cannot
conflict with a LangChain-created L2 index.

## Scope

- Keep `normalize_embeddings=True` and the retriever's `metric_type="IP"`.
- Create and load the collection's IP HNSW index as part of initial collection
  setup.
- Keep index creation idempotent for normal application startup and ingestion.
- Add a unit test that verifies the collection setup sends `HNSW` and `IP` to
  Milvus.

## Data Flow

On the first ingestion, `add_documents_to_milvus()` detects that the collection
does not exist and calls `init_collection()`. That function creates the schema,
then creates and loads an HNSW index with `metric_type="IP"`. The LangChain
vector store then writes normalized embedding vectors. Search continues to pass
`metric_type="IP"`, matching the index.

## Error Handling

Milvus errors during collection or index creation propagate through the existing
upload failure path. The change does not drop, rebuild, or modify an existing
collection/index automatically.

## Testing

Extend `tests/test_vector_store.py` with a mocked `Collection` to assert that
new collection initialization creates an index on `vector` using HNSW and IP,
then loads the collection. Existing vector-store tests remain unchanged.
