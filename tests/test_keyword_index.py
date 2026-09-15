import sqlite3

import pytest
from langchain_core.documents import Document

from keyword_index import KeywordIndex
from utils.logger_util import logger


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


def _noise_documents(count):
    """构造一批只含高频词元“情况具体”的子块，用于把剪枝阈值顶到生效区间。"""
    return [_doc("情况具体", f"parent-noise-{number}", 0) for number in range(count)]


def test_search_prunes_tokens_that_appear_in_almost_every_chunk(tmp_path):
    """几乎无处不在的词元区分度极低，会在 BM25 累加打分中淹没稀有的信号词元。"""
    target = _doc("云辇系统", "parent-target", 0)
    documents = _noise_documents(11) + [target]

    pruned = KeywordIndex(tmp_path / "pruned.db")
    pruned.upsert_documents(documents)
    assert pruned.search("云辇系统情况", top_k=40) == [target]

    unpruned = KeywordIndex(tmp_path / "unpruned.db", max_document_ratio=None)
    unpruned.upsert_documents(documents)
    assert len(unpruned.search("云辇系统情况", top_k=40)) == 12


def test_search_keeps_results_when_every_query_token_is_frequent(tmp_path):
    """整条查询都由高频词元组成时不能剪空，否则召回会退化成恒为零。"""
    index = KeywordIndex(tmp_path / "keyword.db")
    index.upsert_documents(_noise_documents(11))

    assert len(index.search("情况具体", top_k=40)) == 11


def test_repeated_overwrite_does_not_inflate_document_frequency(tmp_path):
    """覆盖写入必须先撤销旧文本的文档频率，否则反复覆盖会把稀有词元挤成高频词元。"""
    index = KeywordIndex(tmp_path / "keyword.db")
    index.upsert_documents(_noise_documents(11))
    target = _doc("云辇系统", "parent-target", 0)
    for _ in range(11):
        index.upsert_documents([_doc("情况具体", "parent-target", 0)])
        index.upsert_documents([target])

    assert index.search("云辇系统情况", top_k=40) == [target]


def test_search_warns_when_document_frequency_statistics_are_missing(tmp_path):
    """索引若建于引入剪枝之前，统计表为空会让剪枝静默失效，必须告警而不是默默降级。"""
    path = tmp_path / "keyword.db"
    index = KeywordIndex(path)
    index.upsert_documents(_noise_documents(11) + [_doc("云辇系统", "parent-target", 0)])

    connection = sqlite3.connect(path)
    with connection:
        connection.execute("DELETE FROM keyword_token_stats")
    connection.close()

    messages = []
    sink = logger.add(
        lambda message: messages.append(message.record["message"]), level="WARNING"
    )
    try:
        reopened = KeywordIndex(path)
        assert len(reopened.search("云辇系统情况", top_k=40)) == 12
    finally:
        logger.remove(sink)

    assert any("剪枝" in message for message in messages)


def test_rebuild_token_statistics_restores_pruning(tmp_path):
    """补齐统计后，历史索引上的剪枝应当重新生效。"""
    path = tmp_path / "keyword.db"
    target = _doc("云辇系统", "parent-target", 0)
    KeywordIndex(path).upsert_documents(_noise_documents(11) + [target])

    connection = sqlite3.connect(path)
    with connection:
        connection.execute("DELETE FROM keyword_token_stats")
    connection.close()

    reopened = KeywordIndex(path)
    assert len(reopened.search("云辇系统情况", top_k=40)) == 12

    assert reopened.rebuild_token_statistics() > 0
    assert reopened.search("云辇系统情况", top_k=40) == [target]
