import os
import unittest
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from chunker import split_md_content
from decorator.time_consume import time_consume
from graph import app
from retriever import AdvancedRetriever
from vector_store import add_documents_to_milvus

retriever = AdvancedRetriever()
class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)  # add assertion here

    def test1_add(self):
        text = Path('test/营业报告1.txt').read_text(encoding='utf-8')
        chunks = split_md_content(text, source_filename='营业报告1.txt', year=2025)
        add_documents_to_milvus(chunks)

    def test2_retrive(self):

        results = retriever.search("业扩咨询有多少条？")

        print(results)


    def test_deepseek(self):
        load_dotenv()

        client = OpenAI(
            api_key=os.getenv('DEEPSEEK_API_KEY'),
            base_url=os.getenv('DEEPSEEK_BASE_URL'))

        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": "Hello"},
            ],
            stream=False
        )

        print(response.choices[0].message.content)

    def test_env(self):
        print("API Key:", repr(os.getenv("DEEPSEEK_API_KEY")))

    @time_consume
    def test_graph(self):
        response = app.invoke({"query": "业务进度有多少条数据？", "year": 2025})
        print(response['answer'])



if __name__ == '__main__':
    unittest.main()
