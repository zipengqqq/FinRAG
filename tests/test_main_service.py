import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from service.main_service import Service


def test_step_parse_pdf_returns_worker_markdown_path(monkeypatch, tmp_path):
    markdown_path = tmp_path / "report.md"
    markdown_path.write_text("# report", encoding="utf-8")

    def fake_run(command, check):
        assert check is False
        assert Path(command[1]).name == "marker_parse.py"
        result_path = Path(command[-1])
        result_path.write_text(
            json.dumps({"ok": True, "markdown_path": str(markdown_path)}),
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("service.main_service.subprocess.run", fake_run)

    assert Service()._step_parse_pdf(tmp_path / "report.pdf") == str(markdown_path)


def test_step_parse_pdf_raises_when_worker_crashes(monkeypatch, tmp_path):
    def fake_run(command, check):
        assert check is False
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr("service.main_service.subprocess.run", fake_run)

    with pytest.raises(RuntimeError, match="Marker worker failed"):
        Service()._step_parse_pdf(tmp_path / "report.pdf")


def test_step_parse_pdf_reports_worker_python_error(monkeypatch, tmp_path):
    def fake_run(command, check):
        assert check is False
        Path(command[-1]).write_text(
            json.dumps({"ok": False, "error": "invalid PDF"}),
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr("service.main_service.subprocess.run", fake_run)

    with pytest.raises(RuntimeError, match="invalid PDF"):
        Service()._step_parse_pdf(tmp_path / "report.pdf")
