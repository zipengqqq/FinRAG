import hashlib
import re

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from utils.logger_util import logger


_HEADING_PATTERN = re.compile(r"^(#{1,5})\s+(.+?)\s*$")
_TABLE_SEPARATOR_CELL = re.compile(r"^:?-{3,}:?$")
_CHUNK_SIZE = 500
_CHUNK_OVERLAP = 50


def split_md_content(md_text, source_filename, metadata=None, document_id=None) -> list[Document]:
    """将 Markdown 切分为带稳定记录元数据的可检索子块。"""
    source = str(source_filename)
    record_metadata = dict(metadata or {})
    document_id = document_id or _stable_id(source, md_text)
    records = _markdown_records(md_text)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=_CHUNK_SIZE,
        chunk_overlap=_CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )

    chunks: list[Document] = []
    for record_index, (record_type, section, content) in enumerate(records):
        child_texts = text_splitter.split_text(content)
        if not child_texts:
            continue

        parent_id = _stable_id(
            document_id, record_type, section, str(record_index), content
        )
        chunk_count = len(child_texts)
        for chunk_index, child_text in enumerate(child_texts):
            chunks.append(
                Document(
                    page_content=child_text,
                    metadata={
                        "source": source,
                        "section": section,
                        "document_id": document_id,
                        "parent_id": parent_id,
                        "chunk_index": chunk_index,
                        "chunk_count": chunk_count,
                        "metadata": record_metadata.copy(),
                    },
                )
            )

    logger.info(f"Markdown split into {len(chunks)} retrievable chunks")
    return chunks


def _markdown_records(md_text: str) -> list[tuple[str, str, str]]:
    lines = md_text.splitlines()
    headings: list[str] = []
    text_lines: list[str] = []
    records: list[tuple[str, str, str]] = []

    def flush_text() -> None:
        content = "\n".join(text_lines).strip()
        text_lines.clear()
        if content:
            records.append(("text_block", _section_name(headings), _with_heading(headings, content)))

    index = 0
    while index < len(lines):
        line = lines[index]
        heading = _HEADING_PATTERN.match(line)
        if heading:
            flush_text()
            level = len(heading.group(1))
            headings[level - 1 :] = [heading.group(2)]
            index += 1
            continue

        if _starts_table(lines, index):
            flush_text()
            record_headings = headings.copy()
            table_lines, index = _read_table(lines, index)
            records.extend(_table_records(table_lines, record_headings))
            continue

        text_lines.append(line)
        index += 1

    flush_text()
    return records


def _starts_table(lines: list[str], index: int) -> bool:
    return (
        index + 1 < len(lines)
        and _is_table_row(lines[index])
        and _is_table_separator(_table_cells(lines[index + 1]))
    )


def _read_table(lines: list[str], index: int) -> tuple[list[str], int]:
    table_lines: list[str] = []
    while index < len(lines):
        if _is_table_row(lines[index]):
            table_lines.append(lines[index])
            index += 1
            continue

        next_index = index
        while next_index < len(lines) and not lines[next_index].strip():
            next_index += 1
        if next_index > index and _is_table_continuation_after_blank_line(
            lines, next_index, _table_cells(table_lines[0])
        ):
            index = next_index
            continue
        break
    return table_lines, index


def _is_table_continuation_after_blank_line(
    lines: list[str], table_start: int, previous_header: list[str]
) -> bool:
    if not _is_table_row(lines[table_start]):
        return False

    first_cells = _table_cells(lines[table_start])
    if first_cells == previous_header:
        if table_start + 2 >= len(lines) or not _is_table_row(lines[table_start + 2]):
            return False
        first_data_cells = _table_cells(lines[table_start + 2])
        return not first_data_cells[0] and any(first_data_cells[1:])

    # Marker 在跨页时可能省略重复表头和末尾空列。
    return (
        1 < len(first_cells) <= len(previous_header)
        and not first_cells[0]
        and any(first_cells[1:])
        and table_start + 1 < len(lines)
        and _is_table_separator(_table_cells(lines[table_start + 1]))
    )


def _table_records(
    table_lines: list[str], headings: list[str]
) -> list[tuple[str, str, str]]:
    section = _section_name(headings)
    heading_text = _heading_text(headings)
    header_line, separator_line = table_lines[:2]
    header_cells = _table_cells(header_line)
    column_count = len(header_cells)
    rows: list[list[str]] = []
    records: list[tuple[str, str, str]] = []
    last_data_row: list[str] | None = None

    index = 2
    while index < len(table_lines):
        line = table_lines[index]
        cells = _table_cells(line)
        if _is_table_separator(cells):
            index += 1
            continue
        if (
            cells == header_cells
            and index + 1 < len(table_lines)
            and _is_table_separator(_table_cells(table_lines[index + 1]))
        ):
            index += 2
            continue

        if len(cells) < column_count:
            cells.extend([""] * (column_count - len(cells)))
        elif len(cells) > column_count:
            cells = cells[:column_count]

        is_continuation = not cells[0] and any(cells[1:])
        if is_continuation and last_data_row is not None:
            for column_index, value in enumerate(cells):
                if value:
                    previous = last_data_row[column_index]
                    last_data_row[column_index] = f"{previous}\n{value}" if previous else value
            index += 1
            continue

        if is_continuation:
            content = _join_nonempty([heading_text, line])
            records.append(("text_block", section, content))
            index += 1
            continue

        row = cells
        rows.append(row)
        last_data_row = row
        index += 1

    for row in rows:
        content = _join_nonempty(
            [heading_text, header_line, separator_line, _render_table_row(row)]
        )
        records.append(("table_row", section, content))
    return records


def _is_table_row(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith("|") and stripped.endswith("|") and stripped.count("|") >= 2


def _table_cells(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _is_table_separator(cells: list[str]) -> bool:
    return bool(cells) and all(_TABLE_SEPARATOR_CELL.fullmatch(cell) for cell in cells)


def _render_table_row(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def _section_name(headings: list[str]) -> str:
    return " / ".join(headings) if headings else "正文"


def _heading_text(headings: list[str]) -> str:
    return "\n".join(f"{'#' * (index + 1)} {heading}" for index, heading in enumerate(headings))


def _with_heading(headings: list[str], content: str) -> str:
    return _join_nonempty([_heading_text(headings), content])


def _join_nonempty(parts: list[str]) -> str:
    return "\n".join(part for part in parts if part)


def _stable_id(*values: object) -> str:
    digest_input = "\0".join(str(value) for value in values)
    return hashlib.sha256(digest_input.encode("utf-8")).hexdigest()
