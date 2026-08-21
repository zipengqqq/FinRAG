import json
import math
from pathlib import Path

import torch
from langchain_core.documents import Document
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from decorator.time_consume import time_consume
from keyword_index import get_keyword_index
from utils.logger_util import logger
from utils.model_paths import resolve_model_path
from vector_store import get_vector_store


RERANKER_MODEL_CACHE_DIR = Path(__file__).resolve().parent / "models" / "bge-reranker-base"
RERANKER_MODEL_PATH = resolve_model_path(
    RERANKER_MODEL_CACHE_DIR, "Xorbits/bge-reranker-base"
)
VECTOR_CANDIDATE_COUNT = 40
KEYWORD_CANDIDATE_COUNT = 40
RERANK_CANDIDATE_COUNT = VECTOR_CANDIDATE_COUNT + KEYWORD_CANDIDATE_COUNT
RRF_K = 60


class AdvancedRetriever:
    def __init__(self):
        logger.info("正在加载 Reranker 模型")
        self.tokenizer = AutoTokenizer.from_pretrained(RERANKER_MODEL_PATH, use_fast=False)
        self.model = AutoModelForSequenceClassification.from_pretrained(RERANKER_MODEL_PATH)
        self.model.eval()
        logger.info("Reranker 加载完成")

    def rerank(self, query, docs, top_k=5):
        """对 Milvus 召回的文档进行精细打分排序。"""
        if not docs:
            return []

        pairs = [[query, doc.page_content] for doc in docs]
        with torch.no_grad():
            inputs = self.tokenizer(
                pairs,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512,
            )
            scores = self.model(**inputs, return_dict=True).logits.view(-1).float()

        doc_score_pairs = list(zip(docs, scores.tolist()))
        sorted_docs = sorted(doc_score_pairs, key=lambda item: item[1], reverse=True)
        final_docs = []
        for doc, score in sorted_docs[:top_k]:
            doc.metadata["rerank_score"] = score
            final_docs.append(doc)
        return final_docs

    @staticmethod
    def _document_key(document):
        """为候选去重生成稳定键，保留同一父记录的不同子块。"""
        parent_id = document.metadata.get("parent_id")
        chunk_index = document.metadata.get("chunk_index")
        if parent_id is not None and chunk_index is not None:
            return f"{parent_id}:{chunk_index}"
        return (
            document.metadata.get("source"),
            document.metadata.get("section"),
            document.page_content,
        )

    @staticmethod
    def _result_key(document, index):
        """为最终结果分组，避免同一文档章节的近重复内容占满结果。"""
        source = document.metadata.get("source")
        section = document.metadata.get("section")
        if source and section:
            return ("section", source, section)

        parent_id = document.metadata.get("parent_id")
        return ("parent", parent_id) if parent_id else ("legacy", index)

    @classmethod
    def _fuse_candidates(cls, vector_docs, keyword_docs):
        """按 RRF 融合独立向量和关键词召回结果。"""
        scores = {}
        documents = {}
        for ranked_documents in (vector_docs, keyword_docs):
            for rank, document in enumerate(ranked_documents, start=1):
                key = cls._document_key(document)
                documents.setdefault(key, document)
                scores[key] = scores.get(key, 0.0) + 1 / (RRF_K + rank)
        return [documents[key] for key in sorted(scores, key=lambda key: -scores[key])]

    @staticmethod
    def _build_filter_expr(source=None, filters=None):
        """构造仅包含固定字段和 JSON 标量等值比较的 Milvus 过滤表达式。"""
        expressions = []
        if source is not None:
            if not isinstance(source, str):
                raise ValueError("source 必须是字符串")
            expressions.append(f"source == {json.dumps(source, ensure_ascii=False)}")

        if filters is not None:
            if not isinstance(filters, dict):
                raise ValueError("filters 必须是字典")
            for key, value in filters.items():
                if not isinstance(key, str) or not key.isidentifier():
                    raise ValueError("filters 的字段名必须是 Python 标识符")
                if isinstance(value, float) and not math.isfinite(value):
                    raise ValueError("filters 不支持非有限浮点数")
                if not isinstance(value, (str, int, float, bool)):
                    raise ValueError("filters 的值只支持字符串、数字和布尔值")
                expressions.append(
                    f'metadata[{json.dumps(key, ensure_ascii=False)}] == '
                    f"{json.dumps(value, ensure_ascii=False)}"
                )

        return " and ".join(expressions) if expressions else None

    @staticmethod
    def _document_from_record(record, rerank_score):
        """将 Milvus 精确查询结果转换为 LangChain Document。"""
        if isinstance(record, Document):
            document = Document(
                page_content=record.page_content,
                metadata=dict(record.metadata),
            )
        else:
            metadata_fields = (
                "source",
                "section",
                "document_id",
                "parent_id",
                "chunk_index",
                "chunk_count",
                "metadata",
            )
            document = Document(
                page_content=str(record.get("text", "")),
                metadata={
                    field_name: record[field_name]
                    for field_name in metadata_fields
                    if field_name in record
                },
            )
        document.metadata["rerank_score"] = rerank_score
        return document

    def _expand_parent_record(self, vector_store, hit, rerank_score):
        """按父记录精确查询全部子块，查询失败时保留命中的原始子块。"""
        parent_id = hit.metadata.get("parent_id")
        collection = getattr(vector_store, "col", None)
        query_records = getattr(collection, "query", None)
        if not parent_id or not callable(query_records):
            return [hit]

        output_fields = [
            "text",
            "source",
            "section",
            "document_id",
            "parent_id",
            "chunk_index",
            "chunk_count",
            "metadata",
        ]
        try:
            records = query_records(
                expr=f"parent_id == {json.dumps(parent_id, ensure_ascii=False)}",
                output_fields=output_fields,
            )
        except Exception as exc:
            logger.warning(f"查询父记录 {parent_id!r} 失败，保留命中子块: {exc}")
            return [hit]

        if not records:
            return [hit]

        documents = [self._document_from_record(record, rerank_score) for record in records]
        return sorted(documents, key=lambda document: document.metadata.get("chunk_index", 0))

    @time_consume
    async def search(self, query, source=None, filters=None, top_k=5):
        """执行向量召回、重排，并返回按父记录补全后的连续内容。"""
        vector_store = get_vector_store()
        filter_expr = self._build_filter_expr(source=source, filters=filters)
        vector_docs = vector_store.similarity_search(
            query,
            k=VECTOR_CANDIDATE_COUNT,
            expr=filter_expr,
            param={
                "metric_type": "IP",
                "params": {"ef": max(64, VECTOR_CANDIDATE_COUNT)},
            },
        )
        try:
            keyword_docs = get_keyword_index().search(
                query=query,
                source=source,
                filters=filters,
                top_k=KEYWORD_CANDIDATE_COUNT,
            )
        except Exception as exc:
            logger.warning(f"关键词检索失败，已降级为仅向量检索: {exc}")
            keyword_docs = []
        rerank_candidates = self._fuse_candidates(vector_docs, keyword_docs)
        reranked_docs = self.rerank(
            query, rerank_candidates, top_k=RERANK_CANDIDATE_COUNT
        )

        parent_hits_by_key = {}
        for index, document in enumerate(reranked_docs):
            parent_key = self._result_key(document, index)
            rerank_score = document.metadata.get("rerank_score", 0.0)
            previous_hit = parent_hits_by_key.get(parent_key)
            if previous_hit is None or rerank_score > previous_hit[1]:
                parent_hits_by_key[parent_key] = (document, rerank_score)

        parent_hits = list(parent_hits_by_key.values())
        parent_hits.sort(key=lambda item: item[1], reverse=True)
        final_docs = []
        for hit, rerank_score in parent_hits[:top_k]:
            # 按照父记录精确查询全部子块
            final_docs.extend(self._expand_parent_record(vector_store, hit, rerank_score))
        return final_docs


retriever = AdvancedRetriever()
