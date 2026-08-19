from pathlib import Path

import torch

COLLECTION_NAME = 'financial_rag'
DIMENSION = 1024
EMBEDDING_MODEL_CACHE_DIR = Path(__file__).resolve().parent / 'models' / 'bge-m3'


from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection
)

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
        FieldSchema(name='section', dtype=DataType.VARCHAR, max_length=1024),
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


@time_consume
def add_documents_to_milvus(chunks, batch_size=256):
    """文档入库（高速写入）"""
    if not chunks:
        logger.info("⚠️ chunks 为空，跳过")
        return

    connections.connect(host=settings.milvus_host, port=settings.milvus_port)

    if not utility.has_collection(COLLECTION_NAME):
        init_collection()

    vector_store = get_vector_store()
    total = len(chunks)

    logger.info(f"🚀 开始入库，共 {total} 条")

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch = chunks[start:end]
        vector_store.add_documents(batch)
        logger.info(f"✅ 已入库 {start} - {end}")

    logger.info("🎉 所有文档入库完成")


@time_consume
def build_hnsw_index():
    """建索引（所有数据完成后执行）"""
    connections.connect(host=settings.milvus_host, port=settings.milvus_port)
    collection = Collection(COLLECTION_NAME)

    # 🔥 关键：先判断 & 删除已有索引
    if collection.indexes:
        logger.info("⚠️ 检测到已有索引，先释放后删除")
        collection.release()
        collection.drop_index()
        logger.info("🗑 原有索引已删除")

    logger.info("开始创建 HNSW 索引（一次性）")

    index_params = {
        "index_type": "HNSW",
        "metric_type": "IP",  # normalize_embeddings=True
        "params": {
            "M": 8,
            "efConstruction": 64
        }
    }

    collection.create_index(
        field_name="vector",
        index_params=index_params
    )

    collection.load()
    logger.info("✅ HNSW 索引创建并加载完成")


def clear_financial_rag(recreate=True):
    """清空 & 重建collections"""
    connections.connect(host=settings.milvus_host, port=settings.milvus_port)

    if utility.has_collection(COLLECTION_NAME):
        utility.drop_collection(COLLECTION_NAME)
        logger.info(f"🗑 Collection {COLLECTION_NAME} 已删除")

    if recreate:
        init_collection()
