import os.path

from langchain_huggingface import HuggingFaceEmbeddings
from modelscope import snapshot_download

MODEL_PATH = './models/bge-m3'


def get_embedding_model():
    """本地没有模型，则会先下载"""
    # 1) 检测模型是否存在
    if not os.path.exists(MODEL_PATH):
        print(f"🚀 本地未检测到模型，正在从 ModelScope 下载 BGE-M3...")
        snapshot_download('Xorbits/bge-m3', cache_dir='./models/bge-m3')
    else:
        print("✅ 检测到本地模型，直接加载。")

    # 2) 加载模型
    real_model_path = MODEL_PATH + '/Xorbits/bge-m3'
    embeddings = HuggingFaceEmbeddings(
        model_name=real_model_path,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True},
    )
    return embeddings

if __name__ == "__main__":
    emb = get_embedding_model()
    vec = emb.embed_query('测试文本')
    print(f"{len(vec)}")

