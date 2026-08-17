from pathlib import Path

import pytest

from utils.model_paths import resolve_model_path


def test_resolve_model_path_supports_modelscope_snapshot_layout(tmp_path):
    model_path = (
        tmp_path
        / "models"
        / "Xorbits--bge-reranker-base"
        / "snapshots"
        / "master"
    )
    model_path.mkdir(parents=True)
    (model_path / "config.json").write_text("{}", encoding="utf-8")

    assert resolve_model_path(tmp_path, "Xorbits/bge-reranker-base") == model_path


def test_resolve_model_path_supports_legacy_modelscope_layout(tmp_path):
    model_path = tmp_path / "Xorbits" / "bge-m3"
    model_path.mkdir(parents=True)
    (model_path / "config.json").write_text("{}", encoding="utf-8")

    assert resolve_model_path(tmp_path, "Xorbits/bge-m3") == model_path


def test_resolve_model_path_reports_missing_local_model(tmp_path):
    with pytest.raises(FileNotFoundError, match="download_model.py"):
        resolve_model_path(tmp_path, "Xorbits/bge-m3")
