import json
import math
import re
from pathlib import Path

import torch
from langchain_core.documents import Document
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from decorator.time_consume import time_consume
from utils.logger_util import logger
from utils.model_paths import resolve_model_path
from vector_store import get_vector_store


RERANKER_MODEL_CACHE_DIR = Path(__file__).resolve().parent / "models" / "bge-reranker-base"
RERANKER_MODEL_PATH = resolve_model_path(
    RERANKER_MODEL_CACHE_DIR, "Xorbits/bge-reranker-base"
)
SEMANTIC_CANDIDATE_COUNT = 200
SEMANTIC_RERANK_CANDIDATE_COUNT = 40
LEXICAL_RERANK_CANDIDATE_COUNT = 40
RERANK_CANDIDATE_COUNT = (
    SEMANTIC_RERANK_CANDIDATE_COUNT + LEXICAL_RERANK_CANDIDATE_COUNT
)
_WORD_PATTERN = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]+")


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
    def _lexical_terms(text):
        """提取不依赖领域词典的中英文检索词项。"""
        terms = set()
        for token in _WORD_PATTERN.findall(text.lower()):
            if all("\u4e00" <= character <= "\u9fff" for character in token):
                if len(token) == 1:
                    terms.add(token)
                else:
                    terms.update(token[index : index + 2] for index in range(len(token) - 1))
            else:
                terms.add(token)
        return terms

    @classmethod
    def _lexical_score(cls, query, document):
        """计算查询与文档的通用词项重合度。"""
        return len(cls._lexical_terms(query) & cls._lexical_terms(document.page_content))

    @classmethod
    def _lexical_scores(cls, query, documents):
        """按候选集内词项稀有度计算通用词项匹配分数。"""
        query_terms = cls._lexical_terms(query)
        document_terms = [cls._lexical_terms(document.page_content) for document in documents]
        document_frequency = {term: 0 for term in query_terms}
        for terms in document_terms:
            for term in query_terms & terms:
                document_frequency[term] += 1

        document_count = len(documents)
        return [
            sum(
                1 + math.log((document_count + 1) / (document_frequency[term] + 1))
                for term in query_terms & terms
            )
            for terms in document_terms
        ]

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
    def _select_rerank_candidates(cls, query, semantic_docs):
        """合并有限语义候选与词项候选，供重排模型统一排序。"""
        semantic_candidates = semantic_docs[:SEMANTIC_RERANK_CANDIDATE_COUNT]
        lexical_scores = cls._lexical_scores(query, semantic_docs)
        lexical_candidates = sorted(
            (
                (score, index, document)
                for index, (score, document) in enumerate(
                    zip(lexical_scores, semantic_docs)
                )
            ),
            key=lambda item: (-item[0], item[1]),
        )
        lexical_candidates = [
            document
            for score, _, document in lexical_candidates[:LEXICAL_RERANK_CANDIDATE_COUNT]
            if score > 0
        ]

        unique_candidates = []
        candidate_keys = set()
        for document in semantic_candidates + lexical_candidates:
            key = cls._document_key(document)
            if key not in candidate_keys:
                candidate_keys.add(key)
                unique_candidates.append(document)
        return unique_candidates

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
        initial_docs = vector_store.similarity_search(
            query,
            k=SEMANTIC_CANDIDATE_COUNT,
            expr=filter_expr,
            param={
                "metric_type": "IP",
                "params": {"ef": max(64, SEMANTIC_CANDIDATE_COUNT)},
            },
        )
        rerank_candidates = self._select_rerank_candidates(query, initial_docs)
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
