import os
import unittest
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from chunker import split_md_content
from decorator.time_consume import time_consume
from entity.file_model import FileModel
from rag_graph import app
from retriever import AdvancedRetriever
from utils.db_util import db_manager, create_session
from utils.id_util import id_worker
from utils.logger_util import logger
from vector_store import add_documents_to_milvus, clear_financial_rag, build_hnsw_index, init_collection


class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)  # add assertion here

    def test1_add(self):
        text = Path('output/营业报告2.txt').read_text(encoding='utf-8')
        chunks = split_md_content(text, source_filename='营业报告2.txt', year=2025)
        add_documents_to_milvus(chunks)

    def test2_retrive(self):
        retriever = AdvancedRetriever()
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

    def test_db(self):
        with db_manager.get_session() as session:
            records = session.query(FileModel).all()
            for record in records:
                print(record)

    def test_id(self):
        for i in range(3):
            print(id_worker.get_id())


    def test_insert_file(self):
        with create_session() as session:
            record = FileModel(
                id=id_worker.get_id(),
                minio_url='fin-rag/263230283145281536/2022-比亚迪-年报.pdf',
                status=2,
                file_name='2022-比亚迪-年报.pdf',
                size='7.4',
                type=0,
                create_time=datetime.now(),
                delete_flag=0
            )
            session.add(record)
            record = FileModel(
                id=id_worker.get_id(),
                minio_url='fin-rag/263230283145281537/2023-比亚迪-年报.pdf',
                status=2,
                file_name='2023-比亚迪-年报.pdf',
                size='9.2',
                type=0,
                create_time=datetime.now(),
                delete_flag=0
            )
            session.add(record)
            record = FileModel(
                id=id_worker.get_id(),
                minio_url='fin-rag/263230283145281538/2024-比亚迪-年报.pdf',
                status=2,
                file_name='2024-比亚迪-年报.pdf',
                size='9.6',
                type=0,
                create_time=datetime.now(),
                delete_flag=0
            )
            session.add(record)

    @time_consume
    def test_insert_milus(self):
        tuples = [
            ('2022-比亚迪-年报.pdf', 2022),
            ('2023-比亚迪-年报.pdf', 2023),
            ('2024-比亚迪-年报.pdf', 2024)
        ]
        for i, file in enumerate(Path('data/docs').glob('*.md')):
            begin = datetime.now()
            content = file.read_text(encoding='utf-8')
            logger.info(f"md文件长度为：{len(content)}")
            logger.info(f"文件内容为：{content[:30]}")
            filename, year = tuples[i]
            chunks = split_md_content(content, source_filename=filename, year=year)
            add_documents_to_milvus(chunks)
            end = datetime.now()
            logger.info(f"{filename}文件插入到milvus，耗时{end - begin}")

    def test_clear_collections(self):
        clear_financial_rag()

    def test_merge(self):
        clear_financial_rag()
        self.test_insert_milus()

        # 所有文件插完后，再建索引
        build_hnsw_index()

    def test_init(self):
        init_collection()




if __name__ == '__main__':
    unittest.main()
