from chunker import split_md_content


TABLE_WITH_BLANK_FIRST_CELL_CONTINUATION = """# 承诺事项
| 承诺方 | 承诺内容 | 履行情况 |
| --- | --- | --- |
| 公司 | 承诺 A | 正在履行 |
|  | 续行条款 | 持续披露 |
"""


def test_split_md_content_assigns_generic_record_metadata():
    chunks = split_md_content("# 安装\n请连接网络。", "guide.md")

    assert len(chunks) == 1
    assert chunks[0].metadata == {
        "source": "guide.md",
        "section": "安装",
        "document_id": chunks[0].metadata["document_id"],
        "parent_id": chunks[0].metadata["parent_id"],
        "chunk_index": 0,
        "chunk_count": 1,
        "metadata": {},
    }
    assert len(chunks[0].metadata["document_id"]) == 64
    assert len(chunks[0].metadata["parent_id"]) == 64


def test_split_md_content_generates_a_stable_document_id():
    first = split_md_content("相同内容。", "guide.md")
    second = split_md_content("相同内容。", "guide.md")

    assert first[0].metadata["document_id"] == second[0].metadata["document_id"]


def test_split_md_content_uses_supplied_document_id_and_metadata():
    chunks = split_md_content(
        "没有标题的正文。",
        "notes.txt",
        metadata={"category": "notes"},
        document_id="document-42",
    )

    assert chunks[0].metadata["section"] == "正文"
    assert chunks[0].metadata["document_id"] == "document-42"
    assert chunks[0].metadata["metadata"] == {"category": "notes"}


def test_split_md_content_numbers_all_children_of_a_long_text_record():
    chunks = split_md_content("a" * 1200, "long.txt")

    assert len(chunks) > 1
    assert {chunk.metadata["parent_id"] for chunk in chunks} == {
        chunks[0].metadata["parent_id"]
    }
    assert [chunk.metadata["chunk_index"] for chunk in chunks] == list(range(len(chunks)))
    assert {chunk.metadata["chunk_count"] for chunk in chunks} == {len(chunks)}


def test_split_md_content_merges_table_continuation_into_one_parent_record():
    chunks = split_md_content(TABLE_WITH_BLANK_FIRST_CELL_CONTINUATION, "report.md")

    row_chunks = [chunk for chunk in chunks if "承诺 A" in chunk.page_content]
    assert len(row_chunks) == 1
    assert "续行条款" in row_chunks[0].page_content
    assert "持续披露" in row_chunks[0].page_content
    assert row_chunks[0].metadata["parent_id"]
    assert row_chunks[0].metadata["chunk_index"] == 0
    assert row_chunks[0].metadata["chunk_count"] == 1


def test_split_md_content_ignores_repeated_table_header_before_continuation():
    markdown = """| 承诺方 | 承诺内容 |
| --- | --- |
| 公司 | 承诺 A |
| 承诺方 | 承诺内容 |
| --- | --- |
|  | 续行条款 |
"""

    chunks = split_md_content(markdown, "report.md")

    row_chunks = [chunk for chunk in chunks if "承诺 A" in chunk.page_content]
    assert len(row_chunks) == 1
    assert "续行条款" in row_chunks[0].page_content


def test_split_md_content_skips_standalone_table_separator_before_continuation():
    markdown = """| A | B |
| --- | --- |
| x | one |
| --- | --- |
|  | two |
"""

    chunks = split_md_content(markdown, "report.md")

    row_chunks = [chunk for chunk in chunks if "one" in chunk.page_content]
    assert len(row_chunks) == 1
    assert "two" in row_chunks[0].page_content


def test_split_md_content_merges_continuation_after_blank_line_and_repeated_header():
    markdown = """| A | B |
| --- | --- |
| x | one |

| A | B |
| --- | --- |
|  | two |
"""

    chunks = split_md_content(markdown, "report.md")

    row_chunks = [chunk for chunk in chunks if "one" in chunk.page_content]
    assert len(row_chunks) == 1
    assert "two" in row_chunks[0].page_content


def test_split_md_content_merges_continuation_after_blank_line_without_repeated_header():
    markdown = """| A | B | C |
| --- | --- | --- |
| x | clause 1 to clause 5 | active |

|  | clause 6 |
| --- | --- | --- |
"""

    chunks = split_md_content(markdown, "report.md")

    row_chunks = [chunk for chunk in chunks if "clause 1" in chunk.page_content]
    assert len(row_chunks) == 1
    assert "clause 6" in row_chunks[0].page_content


def test_split_md_content_does_not_merge_different_table_after_blank_line():
    markdown = """| A | B |
| --- | --- |
| x | one |

| C | D |
| --- | --- |
| y | two |
"""

    chunks = split_md_content(markdown, "report.md")

    first_table_row = next(chunk for chunk in chunks if "one" in chunk.page_content)
    second_table_row = next(chunk for chunk in chunks if "two" in chunk.page_content)
    assert "| C | D |" in second_table_row.page_content
    assert second_table_row.metadata["parent_id"] != first_table_row.metadata["parent_id"]


def test_split_md_content_does_not_merge_same_width_table_after_page_heading():
    markdown = """# 第五节
| A | B |
| --- | --- |
| x | one |
## 第六节 重要事项
|  | 新表第二列 |
| --- | --- |
| y | two |
"""

    chunks = split_md_content(markdown, "report.md")

    previous_row = next(chunk for chunk in chunks if "one" in chunk.page_content)
    new_table_row = next(chunk for chunk in chunks if "two" in chunk.page_content)
    assert "新表第二列" not in previous_row.page_content
    assert "新表第二列" in new_table_row.page_content
    assert new_table_row.metadata["parent_id"] != previous_row.metadata["parent_id"]


def test_split_md_content_does_not_merge_unowned_table_continuation():
    markdown = """| 承诺方 | 承诺内容 |
| --- | --- |
|  | 无归属续行 |
| 公司 | 承诺 B |
"""

    chunks = split_md_content(markdown, "report.md")

    orphan_chunks = [chunk for chunk in chunks if "无归属续行" in chunk.page_content]
    row_chunks = [chunk for chunk in chunks if "承诺 B" in chunk.page_content]
    assert len(orphan_chunks) == 1
    assert len(row_chunks) == 1
    assert orphan_chunks[0].metadata["parent_id"] != row_chunks[0].metadata["parent_id"]
