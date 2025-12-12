from decorator.time_consume import time_consume
from model_loader import get_embedding_model
from langchain_milvus import Milvus
from pymilvus import connections, utility, FieldSchema, CollectionSchema, DataType, Collection

MILVUS_HOST = '127.0.0.1'
MILVUS_PORT = '19530'
COLLECTION_NAME = 'financial_rag'
DIMENSION = 1024

def init_collection():
    """手动定义 Schema 和 Collection"""
    # 数据库连接
    connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)

    # 定义字段
    fields = [
        # 主键 ID，自动增长
        FieldSchema(name='pk', dtype=DataType.INT64, is_primary=True, auto_id=True),
        # 文本内容
        FieldSchema(name='text', dtype=DataType.VARCHAR, max_length=65535),
        # 向量字段（核心）
        FieldSchema(name='vector', dtype=DataType.FLOAT_VECTOR, dim=DIMENSION),
        # -- 下面是元数据字段，用于过滤 --
        FieldSchema(name='source', dtype=DataType.VARCHAR, max_length=200),
        # 年份
        FieldSchema(name='year', dtype=DataType.INT16),
        # 章节标题
        FieldSchema(name='section', dtype=DataType.VARCHAR, max_length=200),
    ]

    schema = CollectionSchema(fields, description="金融财报知识库")

    # 创建集合
    collection = Collection(name=COLLECTION_NAME, schema=schema)

    # 创建索引 -- 必须做，否则无法实现搜索
    index_params = {
        'metric_type': 'L2', # 欧式距离
        'index_type': 'HNSW', # HNSW 是目前最快的向量索引算法之一
        'params': {'M': 8, 'efConstruction': 64}
    }
    collection.create_index(field_name='vector', index_params=index_params)

    # 加载到内存 -- Milvus 的特性，必须 load 才能搜
    collection.load()

    print(f"集合 {COLLECTION_NAME} 创建并加载成功！")
    return collection

def get_vector_store():
    """获取 milvus 向量数据库实例"""
    embedding_model = get_embedding_model()
    vector_store = Milvus(
        embedding_function=embedding_model,
        collection_name=COLLECTION_NAME,
        connection_args={'host': MILVUS_HOST, 'port': MILVUS_PORT},
        # --告诉langchain 文本存在 'text'字段，向量存在 'vector' 字段
        text_field='text',
        vector_field='vector',
    )
    return vector_store

@time_consume
def add_documents_to_milvus(chunks):
    """将切分好的文档存入 Milvus"""
    if not chunks:
        print(f"块为空，不需要入库")
        return

    connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)

    # 先确认表存在
    if not utility.has_collection(COLLECTION_NAME):
        init_collection()

    vector_store = get_vector_store()

    # langchain 会自动把 chunks 的 metadata 映射到 schema 里同名的字段
    # 所以切分的时候，chunks 里的 metadata 必须包含 'source' 和 'section' 字段
    vector_store.add_documents(chunks)

    print(f"入库成功，collection：{COLLECTION_NAME}")


