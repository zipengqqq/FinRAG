from pathlib import Path


def test_requirements_include_sentence_transformers_for_embeddings():
    requirements = (Path(__file__).resolve().parents[1] / "requirements.txt").read_text(
        encoding="utf-8"
    )

    assert any(
        line.startswith("sentence-transformers")
        for line in requirements.splitlines()
    )


def test_requirements_pin_cuda_enabled_pytorch_for_marker():
    requirements = (Path(__file__).resolve().parents[1] / "requirements.txt").read_text(
        encoding="utf-8"
    )

    assert "--extra-index-url https://download.pytorch.org/whl/cu128" in requirements
    assert "torch==2.7.1+cu128" in requirements
    assert "torchvision==0.22.1+cu128" in requirements
