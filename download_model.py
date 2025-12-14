from modelscope import snapshot_download

# 下载重排序模型
model_dir = snapshot_download('Xorbits/bge-reranker-base', cache_dir='./models/bge-reranker-base')
print(f"Reranker 模型已下载到: {model_dir}")

# 下载embeddings模型
model_dir = snapshot_download('Xorbits/bge-m3', cache_dir='./models/bge-m3')
print(f"Embeddings 模型已下载到: {model_dir}")