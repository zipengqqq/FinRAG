"""留出集验证：在未参与词条挖掘与阈值调参的年报上复现分词对比结论。

主评测（eval_tokenizer.py）在 2022/2023 年报上挖词条、调剪枝阈值；本脚本用
2024 年报作为留出集，阈值固定为训练集选定的 2%，验证“bigram + 查询侧剪枝”
相对 jieba 的优势是否可迁移，排除过拟合。
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chunker import split_md_content
from scripts.eval_tokenizer import (
    build_evaluation_set,
    build_index,
    evaluate,
    jieba_tokenize,
    mine_candidate_terms,
)
from keyword_index import tokenize


DEFAULT_MARKDOWN = Path(__file__).resolve().parent.parent / "data" / "docs" / "2024-比亚迪-年报.md"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--work-dir", type=Path, required=True, help="留出集索引输出目录")
    parser.add_argument("--source-name", type=str, default="2024-比亚迪-年报.pdf")
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--mrr-depth", type=int, default=10)
    parser.add_argument("--terms", type=int, default=50)
    parser.add_argument("--max-relevant", type=int, default=40)
    arguments = parser.parse_args()

    arguments.work_dir.mkdir(parents=True, exist_ok=True)

    markdown_text = arguments.markdown.read_text(encoding="utf-8")
    documents = split_md_content(markdown_text, arguments.source_name)
    print(f"留出集语料: {len(documents)} chunks（{arguments.source_name}）")

    terms = mine_candidate_terms(documents)
    evaluation_set = build_evaluation_set(
        documents, terms, arguments.max_relevant, arguments.terms
    )
    print(f"候选未登录词: {len(terms)}，入选评测: {len(evaluation_set)}")
    print("评测词样例:", "、".join(case["term"] for case in evaluation_set[:12]))
    print()

    strategies = {
        "bigram": {"tokenizer": tokenize, "max_document_ratio": None},
        "jieba": {"tokenizer": jieba_tokenize, "max_document_ratio": None},
        # 阈值固定为生产默认值 2%，与训练集一致，不做任何重新调参
        "bigram+剪枝2%": {"tokenizer": tokenize, "max_document_ratio": 0.02},
    }

    results = {}
    for name, options in strategies.items():
        index = build_index(arguments.work_dir / f"{name}.db", documents, **options)
        results[name] = evaluate(index, evaluation_set, arguments.top_k, arguments.mrr_depth)
        print(f"[{name}] 索引构建与评测完成")

    print()
    header = f"{'策略':<14}{f'Hit@{arguments.top_k}':>12}{f'Recall@{arguments.top_k}':>14}{f'MRR@{arguments.mrr_depth}':>12}"
    print(header)
    print("-" * len(header))
    for name, result in results.items():
        print(
            f"{name:<14}{result['hit_rate']:>12.3f}{result['recall']:>14.3f}{result['mrr']:>12.3f}"
        )


if __name__ == "__main__":
    main()
