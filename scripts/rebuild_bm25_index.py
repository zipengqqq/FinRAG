"""从 Milvus 重建 SQLite FTS5 关键词索引。"""

import argparse
import os
import sys
from pathlib import Path

from langchain_core.documents import Document
from pymilvus import Collection, connections


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from keyword_index import DEFAULT_DATABASE_PATH, KeywordIndex
from utils.settings import settings
from vector_store import COLLECTION_NAME


OUTPUT_FIELDS = [
    "pk",
    "text",
    "source",
    "section",
    "document_id",
    "parent_id",
    "chunk_index",
    "chunk_count",
    "metadata",
]


def record_to_document(record):
    """将 Milvus 记录转换为关键词索引使用的 Document。"""
    return Document(
        page_content=str(record["text"]),
        metadata={
            "source": record.get("source"),
            "section": record.get("section"),
            "document_id": record["document_id"],
            "parent_id": record["parent_id"],
            "chunk_index": record["chunk_index"],
            "chunk_count": record.get("chunk_count"),
            "metadata": record.get("metadata") or {},
        },
    )


def iter_milvus_records(page_size):
    """使用查询迭代器分批读取 Milvus 全部子块。"""
    connections.connect(host=settings.milvus_host, port=settings.milvus_port)
    collection = Collection(COLLECTION_NAME)
    collection.load()
    iterator = collection.query_iterator(
        batch_size=page_size,
        expr="",
        output_fields=OUTPUT_FIELDS,
    )
    try:
        while records := iterator.next():
            yield records
    finally:
        iterator.close()


def rebuild_index(target_path=DEFAULT_DATABASE_PATH, page_size=500):
    """重建临时索引并在全部成功后原子替换正式索引。"""
    if page_size <= 0:
        raise ValueError("page_size 必须大于零")

    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = target_path.with_name(f"{target_path.name}.rebuild")
    if temporary_path.exists():
        temporary_path.unlink()

    count = 0
    try:
        index = KeywordIndex(temporary_path)
        for records in iter_milvus_records(page_size):
            documents = [record_to_document(record) for record in records]
            index.upsert_documents(documents)
            count += len(documents)
        os.replace(temporary_path, target_path)
    except Exception:
        if temporary_path.exists():
            temporary_path.unlink()
        raise
    return count


def main():
    """解析命令行参数并执行索引重建。"""
    parser = argparse.ArgumentParser(description="从 Milvus 重建关键词索引")
    parser.add_argument("--database-path", type=Path, default=DEFAULT_DATABASE_PATH)
    parser.add_argument("--page-size", type=int, default=500)
    args = parser.parse_args()
    count = rebuild_index(args.database_path, args.page_size)
    print(f"重建完成：{count} 条子块")


if __name__ == "__main__":
    main()
