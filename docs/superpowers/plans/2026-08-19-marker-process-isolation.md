# Marker 解析进程隔离实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 Marker PDF 解析放到独立进程中，使 `pdfium.dll` 崩溃只影响当前文件处理，不终止 FastAPI 服务。

**Architecture:** 新增 `marker_worker.py` 作为只加载 Marker 的命令行 worker。`Service._step_parse_pdf()` 使用 `subprocess.run()` 启动 worker，worker 通过 JSON 结果文件返回 Markdown 路径或 Python 异常信息；非零退出且无结果文件统一转换为解析失败。Windows 不使用 `multiprocessing.spawn`，避免重新导入 `main.py` 和重复加载 Reranker。

**Tech Stack:** Python 3.12、subprocess、JSON、Marker PDF、pytest。

---

### Task 1: 增加 worker 协议和主服务边界测试

**Files:**

- Modify: `tests/test_marker_parse.py`
- Create: `tests/test_marker_worker.py`
- Create: `tests/test_main_service.py`

- [ ] **Step 1: 写入失败测试**

在 `tests/test_main_service.py` 中新增以下测试，模拟 `subprocess.run()`：

```python
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from service.main_service import Service


def test_step_parse_pdf_returns_worker_markdown_path(monkeypatch, tmp_path):
    markdown_path = tmp_path / "report.md"
    markdown_path.write_text("# report", encoding="utf-8")

    def fake_run(command, check, capture_output, text):
        result_path = Path(command[-1])
        result_path.write_text(
            json.dumps({"ok": True, "markdown_path": str(markdown_path)}),
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr("service.main_service.subprocess.run", fake_run)
    service = Service()

    output = service._step_parse_pdf(tmp_path / "report.pdf")

    assert output == str(markdown_path)


def test_step_parse_pdf_raises_when_worker_crashes(monkeypatch, tmp_path):
    def fake_run(command, check, capture_output, text):
        return SimpleNamespace(returncode=1, stderr="worker crashed")

    monkeypatch.setattr("service.main_service.subprocess.run", fake_run)

    with pytest.raises(RuntimeError, match="Marker worker failed"):
        Service()._step_parse_pdf(tmp_path / "report.pdf")


def test_step_parse_pdf_reports_worker_python_error(monkeypatch, tmp_path):
    def fake_run(command, check, capture_output, text):
        Path(command[-1]).write_text(
            '{"ok": false, "error": "invalid PDF"}', encoding="utf-8"
        )
        return SimpleNamespace(returncode=1, stderr="")

    monkeypatch.setattr("service.main_service.subprocess.run", fake_run)

    with pytest.raises(RuntimeError, match="invalid PDF"):
        Service()._step_parse_pdf(tmp_path / "report.pdf")
```

在 `tests/test_marker_worker.py` 中新增 worker 协议测试：

```python
import json

import marker_worker


def test_worker_writes_markdown_path_on_success(monkeypatch, tmp_path):
    result_path = tmp_path / "result.json"
    markdown_path = tmp_path / "report.md"
    markdown_path.write_text("# report", encoding="utf-8")
    monkeypatch.setattr(marker_worker, "parse_pdf_marker", lambda *args, **kwargs: str(markdown_path))

    exit_code = marker_worker.main([str(tmp_path / "report.pdf"), str(tmp_path), str(result_path)])

    assert exit_code == 0
    assert json.loads(result_path.read_text(encoding="utf-8")) == {
        "ok": True,
        "markdown_path": str(markdown_path),
    }


def test_worker_writes_error_on_python_exception(monkeypatch, tmp_path):
    result_path = tmp_path / "result.json"
    monkeypatch.setattr(
        marker_worker,
        "parse_pdf_marker",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("invalid PDF")),
    )

    exit_code = marker_worker.main([str(tmp_path / "report.pdf"), str(tmp_path), str(result_path)])

    assert exit_code == 1
    assert json.loads(result_path.read_text(encoding="utf-8"))["error"] == "invalid PDF"
```

- [ ] **Step 2: 运行测试确认失败**

运行：

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_main_service.py -v
```

预期：测试收集或执行失败，因为 `marker_worker` 模块尚不存在，且当前主服务没有导入 `subprocess` 或 worker 结果协议。

- [ ] **Step 3: 实现 worker 和主服务调用**

创建 `marker_worker.py`，实现：

```python
import json
import sys
import traceback
from pathlib import Path

from marker_parse import parse_pdf_marker


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 3:
        return 2

    pdf_path, output_dir, result_path = map(Path, args)
    try:
        markdown_path = parse_pdf_marker(str(pdf_path), output_dir=str(output_dir))
        result_path.write_text(
            json.dumps({"ok": True, "markdown_path": markdown_path}, ensure_ascii=False),
            encoding="utf-8",
        )
        return 0
    except Exception as exc:
        result_path.write_text(
            json.dumps({"ok": False, "error": str(exc), "traceback": traceback.format_exc()}, ensure_ascii=False),
            encoding="utf-8",
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
```

在 `service/main_service.py` 中导入 `json`、`subprocess` 和 `sys`，将 `_step_parse_pdf()` 改为：

```python
    def _step_parse_pdf(self, local_pdf_path: Path) -> str:
        result_path = local_pdf_path.with_suffix(".marker-result.json")
        if result_path.exists():
            result_path.unlink()

        command = [
            sys.executable,
            str(Path(__file__).resolve().parent.parent / "marker_worker.py"),
            str(local_pdf_path),
            "output",
            str(result_path),
        ]
        completed = subprocess.run(command, check=False, capture_output=True, text=True)

        if not result_path.exists():
            detail = completed.stderr.strip() or f"exit code {completed.returncode}"
            raise RuntimeError(f"Marker worker failed: {detail}")

        result = json.loads(result_path.read_text(encoding="utf-8"))
        if not result.get("ok"):
            raise RuntimeError(f"Marker worker failed: {result.get('error', 'unknown error')}")

        if completed.returncode != 0:
            detail = completed.stderr.strip() or f"exit code {completed.returncode}"
            raise RuntimeError(f"Marker worker failed: {detail}")

        markdown_path = Path(result["markdown_path"])
        if not markdown_path.exists():
            raise RuntimeError(f"Marker worker output does not exist: {markdown_path}")
        return str(markdown_path)
```

- [ ] **Step 4: 运行目标测试确认通过**

运行：

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_marker_parse.py tests\test_marker_worker.py tests\test_main_service.py -v
```

预期：新 worker 成功、DLL 崩溃式非零退出、worker Python 异常测试和既有 Marker 测试全部通过。

- [ ] **Step 5: 提交实现**

运行：

```powershell
git add marker_worker.py service/main_service.py tests/test_main_service.py tests/test_marker_worker.py tests/test_marker_parse.py
git commit -m "fix: 隔离 Marker PDF 解析进程"
```

### Task 2: 验证 API 进程不会被解析崩溃带走

**Files:**

- Modify: 无
- Test: 本地 API 和 Windows 事件记录

- [ ] **Step 1: 运行全量测试**

运行：

```powershell
.\.venv\Scripts\python.exe -m pytest tests -v
```

预期：所有测试通过。

- [ ] **Step 2: 启动 API 并上传同一 PDF**

运行：

```powershell
.\.venv\Scripts\python.exe main.py
```

上传 `2022-比亚迪-年报.pdf`。预期：解析成功则继续向量化；若 PDFium 崩溃，文件标记失败，但 API 进程仍保持运行并可访问 `/document`。

- [ ] **Step 3: 查询进程状态**

运行：

```powershell
Invoke-WebRequest http://127.0.0.1:8288/openapi.json -UseBasicParsing
```

预期：返回 HTTP 200；不再出现主 Python 进程因 `pdfium.dll` 退出。

## 自检

- worker 成功、Python 异常、无结果文件/非零退出均有明确处理步骤。
- Marker 仍由 `parse_pdf_marker()` 使用，未替换解析器。
- 不使用 `multiprocessing.spawn`，避免重复导入主应用。
- 未包含 TBD、TODO 或未定义步骤。
