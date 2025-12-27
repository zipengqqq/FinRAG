import os
from pathlib import Path
import torch
from decorator.time_consume import time_consume
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered
from marker.config.parser import ConfigParser

def _set_cn_env(cache_dir: str, use_gpu: bool):
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("HF_HOME", cache_dir)
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", cache_dir)
    os.environ.setdefault("TRANSFORMERS_CACHE", cache_dir)
    os.environ.setdefault("TORCH_HOME", str(Path(cache_dir) / "torch"))
    os.environ.setdefault("TORCH_DEVICE", "cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    os.environ.setdefault("MODEL_DTYPE", "float16" if use_gpu and torch.cuda.is_available() else "float32")
    os.environ.setdefault("TORCH_DEVICE_MODEL", "cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if use_gpu and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("medium")
        except Exception:
            pass

@time_consume
def parse_pdf_marker(pdf_path: str, output_dir: str = "output", page_range: str | None = None, force_ocr: bool = False, disable_image_extraction: bool = False) -> str:
    print(f"🚀 正在使用 Marker 解析: {pdf_path} ...")
    cache_dir = str(Path(__file__).parent / ".cache" / "huggingface")
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    _set_cn_env(cache_dir, True)

    config = {"output_format": "markdown"}
    print(f"设备: {os.environ.get('TORCH_DEVICE')}, dtype: {os.environ.get('MODEL_DTYPE')}")
    if page_range:
        config["page_range"] = page_range
    if force_ocr:
        config["force_ocr"] = True
    if disable_image_extraction:
        config["disable_image_extraction"] = True

    config_parser = ConfigParser(config)
    artifact_dict = create_model_dict()
    converter = PdfConverter(
        config=config_parser.generate_config_dict(),
        artifact_dict=artifact_dict,
        processor_list=config_parser.get_processors(),
        renderer=config_parser.get_renderer(),
        llm_service=config_parser.get_llm_service()
    )

    rendered = converter(pdf_path)
    text, _, images = text_from_rendered(rendered)
    out_path = Path(output_dir) / (Path(pdf_path).stem + ".md")
    Path(out_path).write_text(text, encoding="utf-8")
    print(f"✅ 转换完成: {out_path}")
    return str(out_path)

if __name__ == "__main__":
    parse_pdf_marker("data/docs/年报.pdf", output_dir="output")
