from pathlib import Path
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from decorator.time_consume import time_consume
from vector_store import get_vector_store

RERANKER_MODEL_PATH = (Path(__file__).parent / 'models' / 'bge-reranker-base' / 'Xorbits' / 'bge-reranker-base').as_posix()

class AdvancedRetriever:
    def __init__(self):
        print(f"正在加载 Reranker 模型")
        self.tokenizer = AutoTokenizer.from_pretrained(RERANKER_MODEL_PATH, use_fast=False)
        self.model = AutoModelForSequenceClassification.from_pretrained(RERANKER_MODEL_PATH)
        self.model.eval()
        print(f"Reranker 加载完成")

    def rerank(self, query, docs, top_k=5):
        """对 Milvus 召回的文档进行精细打分排序"""
        if not docs:
            return []

        # 构造 input pairs: [[query, doc1], [query, doc2], ...]
        pairs = [[query, doc.page_content] for doc in docs]

        # 计算分数
        with torch.no_grad():
            inputs = self.tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
            scores = self.model(**inputs, return_dict=True).logits.view(-1,).float()

        # 排序
        score_list = scores.tolist()
        doc_score_pairs = list(zip(docs, score_list))

        # 按分数从高到低排序
        sorted_docs = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)

        # 只取前 k 个，并且把分数写入 metadata
        final_docs = []
        for doc, score in sorted_docs[: top_k]:
            doc.metadata['rerank_score'] = score
            final_docs.append(doc)

        return final_docs

    @time_consume
    def search(self, query, year=None, source=None, top_k=5):
        """
            核心检索函数
            1. metadata 过滤
            2. 向量搜索
            3. 重排序
        """
        vector_store = get_vector_store()

        # 1) 构建 Milvus 的过滤表达式
        expr_list = []
        if year:
            expr_list.append(f"year == {year}") # 注意：如果是整数，就不用引号
        if source:
            expr_list.append(f"source == '{source}'")

        filter_expr = " and ".join(expr_list) if expr_list else None

        # 2) 向量检索（召回多一点，比如 20个，让 Reranker 挑）
        initial_docs = vector_store.similarity_search(
            query,
            k=20,
            expr=filter_expr
        )
        final_docs = self.rerank(query, initial_docs, top_k)

        return final_docs




