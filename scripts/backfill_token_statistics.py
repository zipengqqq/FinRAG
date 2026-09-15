"""就地补齐关键词索引的词元文档频率统计。

引入查询侧剪枝之前建立的索引没有 keyword_token_stats 表，剪枝会在其上静默失效
（检索仍然可用，只是退化成朴素二元切分）。统计量完全由索引里已存的子块文本推导，
因此补齐无需访问 Milvus，秒级完成。
"""

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from keyword_index import DEFAULT_DATABASE_PATH, KeywordIndex


def main():
    """解析命令行参数并补齐统计。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-path", type=Path, default=DEFAULT_DATABASE_PATH)
    arguments = parser.parse_args()

    count = KeywordIndex(arguments.database_path).rebuild_token_statistics()
    print(f"统计补齐完成：{count} 个词元")


if __name__ == "__main__":
    main()
