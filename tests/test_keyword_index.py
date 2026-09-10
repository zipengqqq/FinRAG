import pytest
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

    assert index.search("令牌", top_k=10) == [replacement]


def test_search_matches_natural_language_query_without_requiring_every_token(tmp_path):
    """二元切分会产生跨词边界的噪声词元，要求全部命中会让长查询恒定落空。"""
    index = KeywordIndex(tmp_path / "keyword.db")
    target = _doc("刀片电池采用无模组设计提升体积利用率", "parent-1", 0)
    index.upsert_documents([target])

    assert index.search("刀片电池的技术优势体现在哪些方面", top_k=10) == [target]


def test_search_applies_filters_before_limiting_results(tmp_path):
    """过滤必须先于 LIMIT 生效，否则截断会把符合条件的结果挤掉。"""
    index = KeywordIndex(tmp_path / "keyword.db")
    noise = [
        _doc("安装令牌", f"parent-noise-{index_number}", 0, source="other.md")
        for index_number in range(5)
    ]
    target = _doc("安装令牌", "parent-target", 0, source="manual.md")
    index.upsert_documents(noise + [target])

    assert index.search("安装令牌", source="manual.md", top_k=1) == [target]


def test_search_rejects_non_identifier_filter_keys(tmp_path):
    index = KeywordIndex(tmp_path / "keyword.db")
    index.upsert_documents([_doc("安装令牌", "parent-1", 0)])

    with pytest.raises(ValueError):
        index.search("安装令牌", filters={"$.category": "guide"})
