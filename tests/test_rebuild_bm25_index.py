from keyword_index import KeywordIndex
from scripts import rebuild_bm25_index


def test_iter_milvus_records_uses_query_iterator(monkeypatch):
    pages = [[{"pk": 1}], [{"pk": 2}], []]
    calls = {}

    class FakeIterator:
        def next(self):
            return pages.pop(0)

        def close(self):
            calls["closed"] = True

    class FakeCollection:
        def __init__(self, name):
            calls["name"] = name

        def load(self):
            calls["loaded"] = True

        def query_iterator(self, **kwargs):
            calls["query_iterator"] = kwargs
            return FakeIterator()

    monkeypatch.setattr(rebuild_bm25_index.connections, "connect", lambda **kwargs: None)
    monkeypatch.setattr(rebuild_bm25_index, "Collection", FakeCollection)

    assert list(rebuild_bm25_index.iter_milvus_records(page_size=1)) == [
        [{"pk": 1}],
        [{"pk": 2}],
    ]
    assert calls["query_iterator"] == {
        "batch_size": 1,
        "expr": "",
        "output_fields": rebuild_bm25_index.OUTPUT_FIELDS,
    }
    assert calls["closed"] is True


def test_rebuild_reads_all_milvus_pages_and_publishes_index(monkeypatch, tmp_path):
    pages = [
        [
            {
                "text": "第一条",
                "source": "a.md",
                "section": "正文",
                "document_id": "doc",
                "parent_id": "p1",
                "chunk_index": 0,
                "chunk_count": 1,
                "metadata": {},
            }
        ],
        [
            {
                "text": "第二条",
                "source": "a.md",
                "section": "正文",
                "document_id": "doc",
                "parent_id": "p2",
                "chunk_index": 0,
                "chunk_count": 1,
                "metadata": {},
            }
        ],
        [],
    ]
    monkeypatch.setattr(
        rebuild_bm25_index, "iter_milvus_records", lambda page_size: iter(pages)
    )
    target = tmp_path / "keyword.db"

    assert rebuild_bm25_index.rebuild_index(target, page_size=1) == 2
    assert KeywordIndex(target).search("第一条", top_k=10)[0].page_content == "第一条"
    assert KeywordIndex(target).search("第二条", top_k=10)[0].page_content == "第二条"


def test_rebuild_keeps_existing_index_when_reading_milvus_fails(monkeypatch, tmp_path):
    target = tmp_path / "keyword.db"
    index = KeywordIndex(target)
    index.upsert_documents(
        [
            rebuild_bm25_index.record_to_document(
                {
                    "text": "已有内容",
                    "source": "a.md",
                    "section": "正文",
                    "document_id": "doc",
                    "parent_id": "p1",
                    "chunk_index": 0,
                    "chunk_count": 1,
                    "metadata": {},
                }
            )
        ]
    )

    def failing_pages(page_size):
        raise RuntimeError("Milvus 不可用")
        yield

    monkeypatch.setattr(rebuild_bm25_index, "iter_milvus_records", failing_pages)

    try:
        rebuild_bm25_index.rebuild_index(target)
    except RuntimeError:
        pass
    else:
        raise AssertionError("重建失败必须重新抛出异常")

    assert KeywordIndex(target).search("已有内容", top_k=10)[0].page_content == "已有内容"
