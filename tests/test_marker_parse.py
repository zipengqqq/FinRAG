import json

import marker_parse


def test_marker_uses_cuda_for_marker_and_surya_when_available(monkeypatch, tmp_path):
    for name in (
        "TORCH_DEVICE",
        "MODEL_DTYPE",
        "TORCH_DEVICE_MODEL",
        "SURYA_DEVICE",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(marker_parse.torch.cuda, "is_available", lambda: True)

    marker_parse._set_cn_env(str(tmp_path), use_gpu=True)

    assert marker_parse.os.environ["TORCH_DEVICE"] == "cuda"
    assert marker_parse.os.environ["MODEL_DTYPE"] == "float16"
    assert marker_parse.os.environ["TORCH_DEVICE_MODEL"] == "cuda"
    assert marker_parse.os.environ["SURYA_DEVICE"] == "cuda"


def test_parse_pdf_passes_cuda_to_marker_model_factory(monkeypatch, tmp_path):
    captured = {}

    class FakeConfigParser:
        def __init__(self, config):
            pass

        def generate_config_dict(self):
            return {}

        def get_processors(self):
            return []

        def get_renderer(self):
            return "renderer"

        def get_llm_service(self):
            return None

    class FakeConverter:
        def __init__(self, **kwargs):
            pass

        def __call__(self, pdf_path):
            return "rendered"

    def fake_create_model_dict(**kwargs):
        captured.update(kwargs)
        return {}

    monkeypatch.setattr(marker_parse.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(marker_parse, "ConfigParser", FakeConfigParser)
    monkeypatch.setattr(marker_parse, "PdfConverter", FakeConverter)
    monkeypatch.setattr(marker_parse, "create_model_dict", fake_create_model_dict)
    monkeypatch.setattr(marker_parse, "text_from_rendered", lambda rendered: ("text", None, []))

    marker_parse.parse_pdf_marker(str(tmp_path / "report.pdf"), output_dir=str(tmp_path))

    assert captured == {"device": "cuda", "dtype": marker_parse.torch.float16}


def test_worker_entry_writes_markdown_path_on_success(monkeypatch, tmp_path):
    result_path = tmp_path / "result.json"
    markdown_path = tmp_path / "report.md"
    markdown_path.write_text("# report", encoding="utf-8")
    monkeypatch.setattr(
        marker_parse,
        "parse_pdf_marker",
        lambda *args, **kwargs: str(markdown_path),
    )

    exit_code = marker_parse.main(
        [str(tmp_path / "report.pdf"), str(tmp_path), str(result_path)]
    )

    assert exit_code == 0
    assert json.loads(result_path.read_text(encoding="utf-8")) == {
        "ok": True,
        "markdown_path": str(markdown_path),
    }


def test_worker_entry_writes_error_on_python_exception(monkeypatch, tmp_path):
    result_path = tmp_path / "result.json"
    monkeypatch.setattr(
        marker_parse,
        "parse_pdf_marker",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("invalid PDF")),
    )

    exit_code = marker_parse.main(
        [str(tmp_path / "report.pdf"), str(tmp_path), str(result_path)]
    )

    assert exit_code == 1
    assert json.loads(result_path.read_text(encoding="utf-8"))["error"] == "invalid PDF"
