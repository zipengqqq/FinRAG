from pathlib import Path

import pytest

from utils.model_paths import resolve_model_path
import vector_store


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


def test_embedding_model_receives_a_string_model_path(monkeypatch, tmp_path):
    resolved_model_path = tmp_path / "models" / "Xorbits--bge-m3" / "snapshots" / "master"
    captured = {}

    def fake_embeddings(*, model_name, **kwargs):
        captured["model_name"] = model_name
        return object()

    monkeypatch.setattr(vector_store, "_embedding_model", None)
    monkeypatch.setattr(vector_store, "EMBEDDING_MODEL_CACHE_DIR", tmp_path)
    monkeypatch.setattr(vector_store, "resolve_model_path", lambda *_: resolved_model_path)
    monkeypatch.setattr(vector_store, "HuggingFaceEmbeddings", fake_embeddings)

    vector_store.get_embedding_model()

    assert captured["model_name"] == str(resolved_model_path)
