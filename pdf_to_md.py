import os
import sys
from pathlib import Path

import pymupdf4llm
from docling.document_converter import DocumentConverter
from markitdown import MarkItDown


def parse_pdf_docling(pdf_path, output_dir: str = "./output"):
    print(f"🚀 正在使用 Docling 本地解析: {pdf_path} ...")

    converter = DocumentConverter()
    result = converter.convert(pdf_path)

    # 导出为 Markdown
    md_text = result.document.export_to_markdown()

    # 生成输出文件名
    pdf_name = Path(pdf_path).stem
    output_file = os.path.join(output_dir, f"{pdf_name}.md")

    # 保存 Markdown 文件
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(md_text)

    print(f"✅ 转换成功！Markdown 已保存至: {output_file}")

    return md_text


def parse_pdf_local(pdf_path, output_dir: str = "./output"):
    print(f"🚀 正在使用 PyMuPDF4LLM 本地解析: {pdf_path} ...")

    # to_markdown 会自动根据字体大小识别 Headers
    # write_images=False 表示不提取图片，专注文本和表格
    md_text = pymupdf4llm.to_markdown(pdf_path, write_images=False)

    # 生成输出文件名
    pdf_name = Path(pdf_path).stem
    output_file = os.path.join(output_dir, f"{pdf_name}.md")

    # 保存 Markdown 文件
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(md_text)

    print(f"✅ 转换成功！Markdown 已保存至: {output_file}")


def pdf_to_markdown(pdf_path: str, output_dir: str = "./output") -> str:
    """将 PDF 文件转换为 Markdown """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    try:
        print(f"正在解析 PDF: {pdf_path}")
        markitdown = MarkItDown()

        # 自动选择最佳策略（文本 or OCR）
        result = markitdown.convert(
            pdf_path,
            # 可选：强制使用 OCR（适用于扫描件）
            # strategy="ocr",
            # 可选：指定 OCR 语言（支持中英文）
            ocr_languages=["chi_sim", "chi_tra", "eng"]
        )

        markdown_content = result.markdown

        # 生成输出文件名
        pdf_name = Path(pdf_path).stem
        output_file = os.path.join(output_dir, f"{pdf_name}.md")

        # 保存 Markdown 文件
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(markdown_content)

        print(f"✅ 转换成功！Markdown 已保存至: {output_file}")
        return markdown_content

    except Exception as e:
        print(f"❌ 转换失败: {e}", file=sys.stderr)
        raise


if __name__ == "__main__":
    input_pdf = 'data/docs/年报.pdf'
    output_dir = 'data/docs'
    try:
        parse_pdf_docling(input_pdf, output_dir)
    except Exception as e:
        print(f"程序异常退出: {e}")
        sys.exit(1)