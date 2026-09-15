"""端到端检索评测：承载答案的子块是否进入最终上下文。

此前的分词评测只测了关键词单路（BM25 的 Recall@40 / MRR@10）。真实链路是
「向量召回 + 关键词召回 → RRF 合并去重 → 交叉编码器重排 → 父记录展开」，本脚本
把 `AdvancedRetriever.search` 整体作为被测对象，衡量最终交给大模型的上下文里
是否包含承载答案的子块。

对比四种配置，用于回答「混合检索相对纯向量到底有没有增益」：

    纯向量 / 纯关键词 / 双路无剪枝 / 双路+剪枝

相关性判定沿用主评测的客观标准（正文包含评测词即为相关），不引入人工标注。
"""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import keyword_index
import retriever as retriever_module
from scripts.eval_tokenizer import (
    build_evaluation_set,
    document_key,
    load_documents,
    mine_candidate_terms,
)


class _EmptyKeywordIndex:
    """纯向量配置：关键词一路返回空，模拟关键词索引不可用。"""

    def search(self, **_kwargs):
        return []


class _KeywordOnlyStore:
    """纯关键词配置：向量召回返回空，但保留父记录展开能力。

    父记录展开走 `col.query`，属于结果补全而非召回，两种配置下都应保留，
    否则对比的就不是「召回能力」而是「后处理能力」了。
    """

    def __init__(self, store):
        self._store = store

    @property
    def col(self):
        return self._store.col

    def similarity_search(self, *_args, **_kwargs):
        return []


def apply_configuration(name, real_store, unpruned_index):
    """把指定配置装载到 retriever 模块。

    `retriever.search` 内部通过模块级名字取两路索引，因此替换模块属性即可在
    不改动被测代码的前提下切换配置。
    """
    if name == "纯向量":
        retriever_module.get_keyword_index = lambda: _EmptyKeywordIndex()
        retriever_module.get_vector_store = lambda: real_store
    elif name == "纯关键词":
        retriever_module.get_keyword_index = keyword_index.get_keyword_index
        retriever_module.get_vector_store = lambda: _KeywordOnlyStore(real_store)
    elif name == "双路无剪枝":
        retriever_module.get_keyword_index = lambda: unpruned_index
        retriever_module.get_vector_store = lambda: real_store
    elif name == "双路+剪枝":
        retriever_module.get_keyword_index = keyword_index.get_keyword_index
        retriever_module.get_vector_store = lambda: real_store
    else:
        raise ValueError(f"未知配置: {name}")


async def evaluate(retriever, evaluation_set, top_k):
    """统计最终上下文对相关子块的命中率与召回率。"""
    hits = 0
    recall_total = 0.0
    context_sizes = []

    for order, case in enumerate(evaluation_set, start=1):
        documents = await retriever.search(case["query"], top_k=top_k)
        keys = {document_key(document) for document in documents}
        relevant = case["relevant"]
        matched = keys & relevant

        hits += 1 if matched else 0
        recall_total += len(matched) / len(relevant)
        context_sizes.append(len(documents))
        print(
            f"    [{order}/{len(evaluation_set)}] 命中={'是' if matched else '否'}"
            f" 上下文 {len(documents)} 条  {case['term']}",
            flush=True,
        )

    count = len(evaluation_set)
    return {
        "hit_rate": hits / count,
        "recall": recall_total / count,
        "average_context": sum(context_sizes) / count,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-db", type=Path, default=keyword_index.DEFAULT_DATABASE_PATH)
    parser.add_argument("--top-k", type=int, default=5, help="对齐生产默认的最终父记录条数")
    parser.add_argument("--terms", type=int, default=50, help="评测词条数量")
    parser.add_argument("--max-relevant", type=int, default=40, help="相关集规模上限")
    parser.add_argument(
        "--configurations",
        nargs="+",
        default=["纯向量", "纯关键词", "双路无剪枝", "双路+剪枝"],
    )
    arguments = parser.parse_args()

    documents = load_documents(arguments.source_db)
    terms = mine_candidate_terms(documents)
    evaluation_set = build_evaluation_set(
        documents, terms, arguments.max_relevant, arguments.terms
    )
    print(f"语料 {len(documents)} chunks，评测集 {len(evaluation_set)} 条", flush=True)
    print("评测词样例:", "、".join(case["term"] for case in evaluation_set[:12]), flush=True)
    print()

    # 直接用模块级单例，避免重复加载一次 Reranker。
    retriever = retriever_module.retriever
    real_store = retriever_module.get_vector_store()
    unpruned_index = keyword_index.KeywordIndex(
        arguments.source_db, max_document_ratio=None
    )

    results = {}
    for name in arguments.configurations:
        print(f"[{name}] 开始", flush=True)
        apply_configuration(name, real_store, unpruned_index)
        results[name] = asyncio.run(evaluate(retriever, evaluation_set, arguments.top_k))
        print(f"[{name}] 完成", flush=True)

    print()
    header = f"{'配置':<12}{'最终命中率':>12}{'最终召回率':>12}{'平均上下文':>12}"
    print(header)
    print("-" * len(header))
    for name, result in results.items():
        print(
            f"{name:<12}{result['hit_rate']:>12.3f}{result['recall']:>12.3f}"
            f"{result['average_context']:>12.1f}"
        )


if __name__ == "__main__":
    main()
