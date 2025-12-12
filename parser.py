import os
from pathlib import Path

from llama_parse import LlamaParse

from dotenv import load_dotenv

from decorator.time_consume import time_consume

load_dotenv()
llama_cloud_api_key = os.getenv('LLAMA_CLOUD_API_KEY')

@time_consume
def parser_financial_report(pdf_path):
    parser = LlamaParse(
        api_key=llama_cloud_api_key,
        result_type="markdown",
        parsing_instruction="""
        这是一个上市公司的年度财报。
        请注意表格的处理：
        1. 对于跨页表格，请尝试合并上下文。
        2. 保留表格中的所有数字和单位。
        3. 不要遗漏'合并资产负债表'、'合并利润表'中的任何一行。
        """,
        language='ch_sim',
        verbose=True
    )

    documents = parser.load_data(pdf_path)
    return documents[0].text

if __name__ == "__main__":
    text = parser_financial_report('data/docs/营业专业用户诉求分析报告1.pdf')
    Path('test/营业报告2.txt').write_text(text, encoding='utf-8')
