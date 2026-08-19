# General RAG Answering Design

## Goal

Make `rag_graph.py` usable with any domain knowledge base rather than only financial documents. The answer path must depend on whether the current query retrieves relevant documents.

## Answering Behavior

- When retrieval returns one or more documents, the generator uses those documents as its knowledge-base context and identifies the supporting sources in the response.
- When retrieval returns no documents, the generator does not include an empty knowledge-base context and does not claim to have sources. It answers the user's question directly with the model's general knowledge.
- The fallback criterion is the retrieval result for the current query. It is not a check of whether the vector collection has ever contained documents.

## Scope

- Keep query rewriting, vector retrieval, reranking, and the LangGraph topology unchanged.
- Replace finance-specific prompts with domain-neutral prompts.
- Add focused tests for the knowledge-base and no-result generation paths.

## Error Handling

This change does not alter retrieval failures. A retrieval exception remains an operational failure rather than being silently treated as an empty result.
