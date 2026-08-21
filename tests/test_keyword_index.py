from langchain_core.documents import Document

from keyword_index import KeywordIndex


def _doc(text, parent_id, chunk_index, source="manual.md", metadata=None):
    return Document(
        page_content=text,
        metadata={
            "source": source,
            "section": "安装",
            "document_id": "document-1",
            "parent_id": parent_id,
            "chunk_index": chunk_index,
            "chunk_count": 1,
            "metadata": metadata or {},
        },
    )


def test_search_returns_exact_match_not_dependent_on_vector_candidates(tmp_path):
    index = KeywordIndex(tmp_path / "keyword.db")
    target = _doc("身份验证令牌配置", "parent-1", 0)
    index.upsert_documents([target])

    assert index.search("身份验证令牌", top_k=10) == [target]


def test_search_applies_source_and_metadata_filters(tmp_path):
    index = KeywordIndex(tmp_path / "keyword.db")
    guide = _doc("安装令牌", "parent-guide", 0, metadata={"category": "guide"})
    other = _doc("安装令牌", "parent-other", 0, source="other.md")
    index.upsert_documents([guide, other])

    assert index.search("安装令牌", source="manual.md", filters={"category": "guide"}) == [guide]


def test_upsert_replaces_document_with_same_stable_key(tmp_path):
    index = KeywordIndex(tmp_path / "keyword.db")
    index.upsert_documents([_doc("旧令牌", "parent-1", 0)])
    replacement = _doc("新令牌", "parent-1", 0)
    index.upsert_documents([replacement])

    assert index.search("旧令牌", top_k=10) == []
    assert index.search("新令牌", top_k=10) == [replacement]
