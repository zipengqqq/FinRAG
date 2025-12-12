from pathlib import Path

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter


def split_md_content(md_text, source_filename, year) -> list:
    """
    对markdown文本进行切分

    return: 切分好的 Document 对象列表（包含 content 和 metadata）
    """
    # 1) 基于markdown标题，进行语义切分
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
        ("####", "Header 4"),
        ("#####", "Header 5"),
    ]

    # 初始化 Markdown 切分器
    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False, # 将 headers 保留，有助于语义完整
    )

    # 执行切分
    # 结果是一个 Document 列表，每个 Document 的 metadata 里面会包含 {'Header 1': '...', 'Header 2': '...'}
    md_header_splits = md_splitter.split_text(md_text)

    print(f"基于标题切分后，共有{len(md_header_splits)}个块")

    # 2) 基于字符长度递归切分（防止某个块内容太长）
    chunk_size = 500 # 每个块大约 500字符
    chunk_overlap = 50 # 每个块重叠部分，防止上下文割裂
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=['\n\n', '\n', ' ', ''] # 优先按段落切，其次按换行切
    )

    splits = text_splitter.split_documents(md_header_splits)

    final_splits = []
    for doc in splits:
        # 1) 提取章节信息
        # 把 Header 1/2/3 拼起来变成 “第七节-财务报告-资产负债表”
        headers = [v for k, v in doc.metadata.items() if k.startswith('Header')]
        section_name = '-'.join(headers) if headers else '正文'

        # 2) 构建 schema 的 metadata
        new_metadata = {
            'source': source_filename,
            'section': section_name,
            'year': year
        }

        doc.metadata = new_metadata
        final_splits.append(doc)


    print(f"递归切分后，共有{len(final_splits)}个块")

    return final_splits

if __name__ == "__main__":
    text = Path('test/营业报告1.txt').read_text(encoding='utf-8')
    split_md_content(text, source_filename='营业报告1.txt', year=2025)