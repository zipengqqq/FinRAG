# Local Docker Middleware Implementation Plan

> For agentic workers: REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Run MinIO and Milvus locally through Docker Desktop while FinRAG's Python/FastAPI process runs on Windows and connects to the user's existing MySQL container.

**Architecture:** Compose manages etcd, MinIO, and Milvus only, persists their empty development data in named volumes, and exposes them only through 127.0.0.1. A settings module supplies host-side endpoints and an idempotent CLI creates only the fin_rag schema and its file table in external MySQL.

**Tech Stack:** Docker Compose, MinIO, Milvus 2.3, etcd, Python, SQLAlchemy, PyMySQL, pytest, python-dotenv.

---

## File structure

- .env.example: safe local template.
- .gitignore: ignores local credentials.
- docker-compose.yml: non-MySQL middleware stack.
- utils/settings.py: validated settings and Milvus URI.
- utils/db_util.py, utils/minio_util.py, vector_store.py: consume settings.
- scripts/init_local_db.py: safe external MySQL initializer.
- tests: settings, schema-URI, and Compose configuration coverage.
- README.md: Windows lifecycle and verification.

### Task 1: Centralize and test connection settings

**Files:**
- Create: tests/test_settings.py and utils/settings.py
- Modify: utils/db_util.py, utils/minio_util.py, vector_store.py, requirements.txt

- [ ] **Step 1: Write the failing tests**

~~~python
import pytest
from utils.settings import Settings, read_settings

def test_read_settings_builds_local_milvus_uri(monkeypatch):
    values = {
        "DATABASE_URI": "mysql+pymysql://root:password@127.0.0.1:3306/fin_rag",
        "ENDPOINT": "127.0.0.1:9000", "ACCESS_KEY": "minioadmin",
        "SECRET_KEY": "minioadmin", "BUCKET_NAME": "fin-rag",
        "MILVUS_HOST": "127.0.0.1", "MILVUS_PORT": "19530",
    }
    for key, value in values.items():
        monkeypatch.setenv(key, value)
    settings = read_settings()
    assert settings == Settings(
        values["DATABASE_URI"], values["ENDPOINT"], values["ACCESS_KEY"],
        values["SECRET_KEY"], values["BUCKET_NAME"], "127.0.0.1", "19530")
    assert settings.milvus_uri == "http://127.0.0.1:19530"

def test_read_settings_rejects_missing_database_uri(monkeypatch):
    monkeypatch.delenv("DATABASE_URI", raising=False)
    with pytest.raises(RuntimeError, match="DATABASE_URI"):
        read_settings()
~~~

- [ ] **Step 2: Verify failure**

Run: python -m pytest tests/test_settings.py -v

Expected: fail because utils.settings does not exist.

- [ ] **Step 3: Implement settings**

~~~python
# utils/settings.py
import os
from dataclasses import dataclass
from dotenv import load_dotenv

@dataclass(frozen=True)
class Settings:
    database_uri: str
    minio_endpoint: str
    minio_access_key: str
    minio_secret_key: str
    bucket_name: str
    milvus_host: str
    milvus_port: str
    @property
    def milvus_uri(self) -> str:
        return f"http://{self.milvus_host}:{self.milvus_port}"

def _required(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value

def read_settings() -> Settings:
    load_dotenv(override=False)
    return Settings(
        _required("DATABASE_URI"), _required("ENDPOINT"), _required("ACCESS_KEY"),
        _required("SECRET_KEY"), _required("BUCKET_NAME"),
        os.getenv("MILVUS_HOST", "127.0.0.1"), os.getenv("MILVUS_PORT", "19530"))

settings = read_settings()
~~~

Replace local load_dotenv and getenv calls in the database and MinIO utilities with settings fields. In vector_store.py, use settings.milvus_host and settings.milvus_port for PyMilvus and settings.milvus_uri for LangChain Milvus. Append pymysql~=1.1.1, langchain-milvus~=0.1.10, and pytest~=8.3.5 to requirements.txt.

- [ ] **Step 4: Verify success**

Run: python -m pytest tests/test_settings.py -v

Expected: 2 passed.

- [ ] **Step 5: Commit**

~~~powershell
git add tests/test_settings.py utils/settings.py utils/db_util.py utils/minio_util.py vector_store.py requirements.txt
git commit -m "feat: 支持本地中间件配置"
~~~

### Task 2: Create the external MySQL schema idempotently

**Files:**
- Create: scripts/__init__.py, scripts/init_local_db.py, tests/test_init_local_db.py

- [ ] **Step 1: Write failing URI-validation tests**

~~~python
import pytest
from scripts.init_local_db import database_name_from_uri

def test_database_name_from_uri_returns_fin_rag():
    assert database_name_from_uri(
        "mysql+pymysql://root:password@127.0.0.1:3306/fin_rag") == "fin_rag"

@pytest.mark.parametrize("uri", [
    "mysql+pymysql://root:password@127.0.0.1:3306/",
    "mysql+pymysql://root:password@127.0.0.1:3306/not-valid-name",
])
def test_database_name_from_uri_rejects_unsafe_names(uri):
    with pytest.raises(ValueError, match="database name"):
        database_name_from_uri(uri)
~~~

- [ ] **Step 2: Verify failure**

Run: python -m pytest tests/test_init_local_db.py -v

Expected: fail because scripts.init_local_db does not exist.

- [ ] **Step 3: Implement safe initialization**

~~~python
# scripts/init_local_db.py
import re
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url
from entity.file_model import FileModel
from utils.settings import settings

def database_name_from_uri(database_uri: str) -> str:
    name = make_url(database_uri).database
    if not name or not re.fullmatch(r"[A-Za-z0-9_]+", name):
        raise ValueError("DATABASE_URI must contain a safe database name")
    return name

def initialize_database(database_uri: str) -> None:
    url = make_url(database_uri)
    name = database_name_from_uri(database_uri)
    with create_engine(url.set(database=None)).begin() as connection:
        connection.execute(text(
            f"CREATE DATABASE IF NOT EXISTS {name} CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"))
    FileModel.metadata.create_all(bind=create_engine(url))

if __name__ == "__main__":
    initialize_database(settings.database_uri)
    print("Local MySQL database fin_rag is ready.")
~~~

Create an empty scripts/__init__.py. In real code, quote the validated database name with MySQL identifier quoting. This command must never use DROP and must not alter non-fin_rag objects.

- [ ] **Step 4: Verify unit and real-container idempotency**

Run: python -m pytest tests/test_init_local_db.py -v; then run python -m scripts.init_local_db twice against local MySQL.

Expected: 3 passed; both CLI runs print Local MySQL database fin_rag is ready.

- [ ] **Step 5: Commit**

~~~powershell
git add scripts/__init__.py scripts/init_local_db.py tests/test_init_local_db.py
git commit -m "feat: 新增本地数据库初始化命令"
~~~

### Task 3: Configure Docker-only middleware and environment files

**Files:**
- Create: .env.example and tests/test_compose_config.py
- Modify: .gitignore and docker-compose.yml
- Remove from Git index only: .env

- [ ] **Step 1: Write the failing Compose test**

~~~python
from pathlib import Path

def test_compose_keeps_mysql_external_and_uses_named_volumes():
    compose = Path("docker-compose.yml").read_text(encoding="utf-8")
    assert "  mysql:" not in compose
    assert all(name in compose for name in ("etcd_data:", "minio_data:", "milvus_data:"))
    assert "127.0.0.1:9000:9000" in compose
    assert "127.0.0.1:19530:19530" in compose
~~~

- [ ] **Step 2: Verify failure**

Run: python -m pytest tests/test_compose_config.py -v

Expected: fail because the current file has host bind mounts and broad port publishing.

- [ ] **Step 3: Implement Compose and the local template**

Replace the three current host mounts with named volumes etcd_data, minio_data, and milvus_data, declared under root volumes. Bind MinIO as 127.0.0.1:9000:9000 and 127.0.0.1:9001:9001; bind Milvus as 127.0.0.1:19530:19530 and 127.0.0.1:9091:9091. Do not add MySQL. Use the existing ACCESS_KEY and SECRET_KEY Compose substitutions for MINIO_ROOT_USER and MINIO_ROOT_PASSWORD. Set both Milvus dependency conditions to service_healthy.

Create .env.example:
~~~dotenv
DATABASE_URI=mysql+pymysql://<MYSQL_USER>:<MYSQL_PASSWORD>@127.0.0.1:3306/fin_rag
ENDPOINT=127.0.0.1:9000
ACCESS_KEY=minioadmin
SECRET_KEY=minioadmin
BUCKET_NAME=fin-rag
MILVUS_HOST=127.0.0.1
MILVUS_PORT=19530
DEEPSEEK_API_KEY=<REPLACE_WITH_YOUR_KEY>
DEEPSEEK_BASE_URL=https://api.deepseek.com
~~~

Append .env to .gitignore and run git rm --cached .env; retain the developer's working copy.

- [ ] **Step 4: Validate and start**

Run: python -m pytest tests/test_compose_config.py -v; docker compose config; docker compose up -d; docker compose ps; Test-NetConnection 127.0.0.1 -Port 9000; Test-NetConnection 127.0.0.1 -Port 19530.

Expected: test and Compose validation pass; both port checks report TcpTestSucceeded : True.

- [ ] **Step 5: Commit**

~~~powershell
git add .gitignore .env.example docker-compose.yml tests/test_compose_config.py
git rm --cached .env
git commit -m "chore: 配置本地 Docker 中间件"
~~~

### Task 4: Document and smoke-test the Windows workflow

**Files:**
- Modify: README.md

- [ ] **Step 1: Add Windows quick-start**

Add a Windows local-development section:
~~~powershell
Copy-Item .env.example .env
# Edit .env: set existing MySQL credentials and DEEPSEEK_API_KEY.
docker compose up -d
docker compose ps
python -m scripts.init_local_db
python -m uvicorn main:app --host 127.0.0.1 --port 8288
~~~

Explain MySQL is a pre-existing, user-managed container; MinIO console is http://127.0.0.1:9001. Document docker compose down and docker compose down -v, explicitly stating the latter deletes only this stack's MinIO/Milvus/etcd named-volume development data.

- [ ] **Step 2: Verify tests and live connections**

Run: python -m pytest -v.

Then, after services and MySQL are ready:
~~~powershell
python -c "from utils.settings import settings; from utils.minio_util import minio_client; from pymilvus import connections; minio_client.list_buckets(); connections.connect(host=settings.milvus_host, port=settings.milvus_port); print('MinIO and Milvus connected')"
python -c "from sqlalchemy import text; from utils.db_util import engine; with engine.connect() as connection: print(connection.execute(text('SELECT 1')).scalar())"
~~~

Expected: all pytest tests pass, then output MinIO and Milvus connected and 1.

- [ ] **Step 3: Commit**

~~~powershell
git add README.md
git commit -m "docs: 补充 Windows 本地启动说明"
~~~

## Plan self-review

- Spec coverage: Task 1 removes remote literals, Task 2 initializes only external MySQL, Task 3 delivers the named-volume Docker middleware and safe template, and Task 4 documents and proves the Windows workflow.
- Placeholder scan: no unresolved implementation steps remain; MySQL credentials and API keys are intentionally developer-specific .env inputs.
- Consistency: every application client uses utils.settings; MySQL initialization and README use DATABASE_URI; Compose gets the same MinIO credentials from .env.
