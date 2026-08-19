from pathlib import Path

import torch

COLLECTION_NAME = 'financial_rag'
DIMENSION = 1024
SECTION_MAX_LENGTH = 1024
EMBEDDING_MODEL_CACHE_DIR = Path(__file__).resolve().parent / 'models' / 'bge-m3'

"""
经过验证，该文件代码对hnsw的使用不存在问题，可放心使用，不会出现随着文档数量增多，插入向量数据库越来越慢的问题
"""

from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection
)
from pymilvus.exceptions import MilvusException

from langchain_milvus import Milvus
from langchain_community.embeddings import HuggingFaceEmbeddings
from modelscope import snapshot_download

from decorator.time_consume import time_consume
from utils.logger_util import logger
from utils.model_paths import resolve_model_path
from utils.settings import settings


# Embedding 单例
_embedding_model = None
# VectorStore 单例
_vector_store = None

def get_embedding_model():
    """Embedding 模型单例（只加载一次）"""
    global _embedding_model
    if _embedding_model is not None:
        return _embedding_model

    if not EMBEDDING_MODEL_CACHE_DIR.exists():
        logger.info("🚀 本地未检测到模型，开始下载 bge-m3")
        snapshot_download('Xorbits/bge-m3', cache_dir=str(EMBEDDING_MODEL_CACHE_DIR))

    real_model_path = resolve_model_path(EMBEDDING_MODEL_CACHE_DIR, 'Xorbits/bge-m3')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    _embedding_model = HuggingFaceEmbeddings(
        model_name=str(real_model_path),
        model_kwargs={'device': device},
        encode_kwargs={'normalize_embeddings': True},  # 对应 IP
    )
    logger.info("✅ Embedding 模型加载完成")
    return _embedding_model


def init_collection():
    """Collection 初始化（不建索引）"""
    connections.connect(host=settings.milvus_host, port=settings.milvus_port)

    fields = [
        FieldSchema(name='pk', dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name='text', dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name='vector', dtype=DataType.FLOAT_VECTOR, dim=DIMENSION),
        FieldSchema(name='source', dtype=DataType.VARCHAR, max_length=200),
        FieldSchema(name='year', dtype=DataType.INT16),
        FieldSchema(name='section', dtype=DataType.VARCHAR, max_length=SECTION_MAX_LENGTH),
    ]

    schema = CollectionSchema(fields, description="金融财报 RAG 知识库")

    collection = Collection(
        name=COLLECTION_NAME,
        schema=schema
    )

    collection.create_index(
        field_name="vector",
        index_params={
            "index_type": "HNSW",
            "metric_type": "IP",
            "params": {
                "M": 8,
                "efConstruction": 64,
            },
        },
    )
    collection.load()

    logger.info(f"✅ Collection {COLLECTION_NAME} 创建完成（HNSW/IP 索引已加载）")
    return collection


def get_vector_store():
    """获取 VectorStore"""
    global _vector_store
    if _vector_store is not None:
        return _vector_store

    embedding = get_embedding_model()

    # 使用 uri 方式连接远程 Milvus
    uri = settings.milvus_uri

    _vector_store = Milvus(
        embedding_function=embedding,
        collection_name=COLLECTION_NAME,
        connection_args={"uri": uri},  # ✅ 正确方式
        auto_id=True,
        text_field='text',
        vector_field='vector',
    )
    return _vector_store


def _is_section_within_limit(chunk) -> bool:
    section = str(chunk.metadata.get('section', ''))
    return len(section.encode('utf-8')) <= SECTION_MAX_LENGTH


def _insert_batch_with_fallback(vector_store, batch, start: int) -> None:
    try:
        vector_store.add_documents(batch)
        return
    except MilvusException as exc:
        logger.warning(
            f"Batch insert failed {start} - {start + len(batch)}; retrying individually: {exc}"
        )

    for offset, chunk in enumerate(batch):
        try:
            vector_store.add_documents([chunk])
        except MilvusException as exc:
            section = str(chunk.metadata.get('section', ''))
            logger.error(
                f"Skipping invalid chunk {start + offset}, section={section[:100]!r}: {exc}"
            )


@time_consume
def add_documents_to_milvus(chunks, batch_size=256):
    """文档入库"""
    if not chunks:
        logger.info("⚠️ chunks 为空，跳过")
        return

    connections.connect(host=settings.milvus_host, port=settings.milvus_port)

    if not utility.has_collection(COLLECTION_NAME):
        init_collection()

    valid_chunks = [chunk for chunk in chunks if _is_section_within_limit(chunk)]
    skipped_count = len(chunks) - len(valid_chunks)
    if skipped_count:
        logger.warning(
            f"Skipping {skipped_count} chunks with section longer than {SECTION_MAX_LENGTH} bytes"
        )

    if not valid_chunks:
        logger.warning("No valid chunks to insert")
        return

    vector_store = get_vector_store()
    total = len(valid_chunks)

    logger.info(f"🚀 开始入库，共 {total} 条")

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch = valid_chunks[start:end]
        _insert_batch_with_fallback(vector_store, batch, start)
        logger.info(f"✅ 已入库 {start} - {end}")

    logger.info("🎉 所有块入库完成")
