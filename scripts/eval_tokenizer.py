"""对比不同中文切分策略在同一套 BM25 下的关键词召回质量。

实验控制：同一份语料、同一个 SQLite FTS5/BM25 实现、同样的 top_k，只替换分词器，
确保测出的差异完全归因于切分策略，而不是 BM25 实现或语料差异。

评测集不依赖任何领域词表：候选词由语料自动挖掘（jieba 切得开、但共现粘性高的组合），
相关性判定用“原文是否包含该词”的客观标准，无需人工标注。
"""

import argparse
import re
import sqlite3
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jieba
from langchain_core.documents import Document

from keyword_index import KeywordIndex, tokenize

DEFAULT_SOURCE_DATABASE = Path(__file__).resolve().parent.parent / "data" / "keyword_index.db"
CONTENT_PATTERN = re.compile(r"^[\w一-鿿]+$")
QUERY_TEMPLATES = (
    "{term}的具体情况是怎样的",
    "关于{term}的说明有哪些",
    "报告中提到的{term}是什么",
    "{term}相关的数据情况如何",
)


def jieba_tokenize(text):
    """词典分词，对齐 Milvus 内置中文 analyzer 采用的策略。"""
    return [
        token
        for token in jieba.lcut(str(text).lower())
        if CONTENT_PATTERN.match(token)
    ]


def document_key(document):
    """与检索链路一致的稳定标识，用于判定是否命中同一子块。"""
    metadata = document.metadata
    return (
        metadata.get("document_id"),
        metadata.get("parent_id"),
        metadata.get("chunk_index"),
    )


def load_documents(database_path):
    """从既有关键词索引读取语料，两个对照索引共用同一份 chunk。"""
    connection = sqlite3.connect(database_path)
    try:
        rows = connection.execute(
            """
            SELECT text, source, section, document_id, parent_id,
                   chunk_index, chunk_count
            FROM keyword_records
            ORDER BY id
            """
        ).fetchall()
    finally:
        connection.close()

    return [
        Document(
            page_content=row[0],
            metadata={
                "source": row[1],
                "section": row[2],
                "document_id": row[3],
                "parent_id": row[4],
                "chunk_index": row[5],
                "chunk_count": row[6],
                "metadata": {},
            },
        )
        for row in rows
    ]


def mine_candidate_terms(documents, stickiness=0.6, minimum_document_frequency=3):
    """挖掘 jieba 切得开、但左右两半互相强预测的组合词。

    共现比例（粘性）能在不引入停用词表的前提下滤掉“公司的”这类松散搭配：
    “的”随处出现，与任何词的共现比例都极低，会被自动排除。
    """
    token_frequency = Counter()
    pair_frequency = Counter()

    for document in documents:
        tokens = list(jieba.cut(document.page_content.lower()))
        seen_tokens = set()
        seen_pairs = set()
        for index, token in enumerate(tokens):
            if not CONTENT_PATTERN.match(token):
                continue
            seen_tokens.add(token)
            if index + 1 < len(tokens):
                following = tokens[index + 1]
                if CONTENT_PATTERN.match(following):
                    seen_pairs.add((token, following))
        token_frequency.update(seen_tokens)
        pair_frequency.update(seen_pairs)

    candidates = []
    for (left, right), frequency in pair_frequency.items():
        if frequency < minimum_document_frequency:
            continue
        term = left + right
        if not 2 <= len(term) <= 8:
            continue
        # jieba 能一次切出整词的，说明它已在词典里，不属于未登录词
        if len(jieba.lcut(term)) < 2:
            continue
        if min(frequency / token_frequency[left], frequency / token_frequency[right]) < stickiness:
            continue
        candidates.append((term, frequency))

    # 次键必须显式指定：集合迭代顺序随进程的字符串哈希种子变化，只按频次排序会让
    # 同频词条的顺序在两次运行间漂移，评测集随之变化，指标就无法复现。
    candidates.sort(key=lambda item: (-item[1], item[0]))
    return [term for term, _ in candidates]


def build_evaluation_set(documents, terms, maximum_relevant, limit):
    """相关集 = 原文包含该词的全部 chunk，客观可复现。

    限制相关集规模，是为了让 Recall@k 在理论上可以取到 1.0，否则指标不可解释。
    """
    evaluation_set = []
    for term in terms:
        relevant = {
            document_key(document)
            for document in documents
            if term in document.page_content.lower()
        }
        if not 1 <= len(relevant) <= maximum_relevant:
            continue
        template = QUERY_TEMPLATES[len(evaluation_set) % len(QUERY_TEMPLATES)]
        evaluation_set.append(
            {"term": term, "query": template.format(term=term), "relevant": relevant}
        )
        if len(evaluation_set) == limit:
            break
    return evaluation_set


def build_index(database_path, documents, **options):
    """把同一批 chunk 重新建成一个独立索引，options 直接透传给 KeywordIndex。

    剪枝等策略一律走生产代码路径，评测与线上共用同一份实现，避免两处逻辑漂移。
    """
    if database_path.exists():
        database_path.unlink()
    index = KeywordIndex(database_path, **options)
    index.upsert_documents(documents)
    return index


def evaluate(index, evaluation_set, top_k, mrr_depth):
    """统计 Hit@k、Recall@k 与 MRR@depth。"""
    hits = 0
    recall_total = 0.0
    reciprocal_rank_total = 0.0
    per_query = []

    for case in evaluation_set:
        retrieved = [document_key(document) for document in index.search(case["query"], top_k=top_k)]
        relevant = case["relevant"]
        matched = [key for key in retrieved if key in relevant]

        recall = len(set(matched)) / len(relevant)
        reciprocal_rank = 0.0
        for rank, key in enumerate(retrieved[:mrr_depth], start=1):
            if key in relevant:
                reciprocal_rank = 1 / rank
                break

        hits += 1 if matched else 0
        recall_total += recall
        reciprocal_rank_total += reciprocal_rank
        per_query.append({"term": case["term"], "recall": recall, "rr": reciprocal_rank})

    count = len(evaluation_set)
    return {
        "hit_rate": hits / count,
        "recall": recall_total / count,
        "mrr": reciprocal_rank_total / count,
        "per_query": per_query,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-db", type=Path, default=DEFAULT_SOURCE_DATABASE)
    parser.add_argument("--work-dir", type=Path, required=True, help="对照索引的输出目录")
    parser.add_argument("--top-k", type=int, default=40, help="对齐 KEYWORD_CANDIDATE_COUNT")
    parser.add_argument("--mrr-depth", type=int, default=10)
    parser.add_argument("--terms", type=int, default=50, help="评测词条数量")
    parser.add_argument("--max-relevant", type=int, default=40, help="相关集规模上限")
    arguments = parser.parse_args()

    arguments.work_dir.mkdir(parents=True, exist_ok=True)

    documents = load_documents(arguments.source_db)
    print(f"语料: {len(documents)} chunks")

    terms = mine_candidate_terms(documents)
    evaluation_set = build_evaluation_set(
        documents, terms, arguments.max_relevant, arguments.terms
    )
    print(f"候选未登录词: {len(terms)}，入选评测: {len(evaluation_set)}")
    print("评测词样例:", "、".join(case["term"] for case in evaluation_set[:12]))
    print()

    # 每个策略只是 KeywordIndex 的一组构造参数，剪枝阈值直接复用生产默认值。
    strategies = {
        "bigram": {"tokenizer": tokenize, "max_document_ratio": None},
        "jieba": {"tokenizer": jieba_tokenize, "max_document_ratio": None},
        "bigram文档+jieba查询": {"tokenizer": tokenize, "query_tokenizer": jieba_tokenize},
        "bigram+剪枝2%": {"tokenizer": tokenize, "max_document_ratio": 0.02},
        "bigram+剪枝5%": {"tokenizer": tokenize, "max_document_ratio": 0.05},
    }
    results = {}
    for name, options in strategies.items():
        index = build_index(arguments.work_dir / f"{name}.db", documents, **options)
        results[name] = evaluate(index, evaluation_set, arguments.top_k, arguments.mrr_depth)
        print(f"[{name}] 索引构建与评测完成")

    print()
    header = f"{'策略':<10}{f'Hit@{arguments.top_k}':>12}{f'Recall@{arguments.top_k}':>14}{f'MRR@{arguments.mrr_depth}':>12}"
    print(header)
    print("-" * len(header))
    for name, result in results.items():
        print(
            f"{name:<10}{result['hit_rate']:>12.3f}{result['recall']:>14.3f}{result['mrr']:>12.3f}"
        )

    print()
    print("=== bigram 与 jieba 的 MRR 差异最大的词条 ===")
    bigram_by_term = {item["term"]: item for item in results["bigram"]["per_query"]}
    jieba_by_term = {item["term"]: item for item in results["jieba"]["per_query"]}
    differences = sorted(
        bigram_by_term,
        key=lambda term: -abs(bigram_by_term[term]["rr"] - jieba_by_term[term]["rr"]),
    )
    print(f"{'词条':<12}{'bigram RR':>12}{'jieba RR':>12}{'bigram R':>12}{'jieba R':>12}")
    for term in differences[:12]:
        bigram_item, jieba_item = bigram_by_term[term], jieba_by_term[term]
        print(
            f"{term:<12}{bigram_item['rr']:>12.3f}{jieba_item['rr']:>12.3f}"
            f"{bigram_item['recall']:>12.3f}{jieba_item['recall']:>12.3f}"
        )


if __name__ == "__main__":
    main()
