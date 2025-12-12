from modelscope import snapshot_download

model_dir = snapshot_download('Xorbits/bge-reranker-base', cache_dir='./models/bge-reranker-base')
print(f"Reranker 模型已下载到: {model_dir}")
