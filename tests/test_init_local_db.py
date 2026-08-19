import importlib.util
import sys
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from sqlalchemy import BigInteger

from entity.file_model import FileModel
from scripts import init_local_db
from scripts.init_local_db import database_name_from_uri


def test_directly_loaded_script_adds_repository_root_to_module_path(monkeypatch):
    script_path = Path(init_local_db.__file__).resolve()
    repository_root = script_path.parents[1]
    direct_script_path = [
        entry
        for entry in sys.path
        if Path(entry or ".").resolve() != repository_root
    ]
    monkeypatch.setattr(sys, "path", [str(script_path.parent), *direct_script_path])
    monkeypatch.delitem(sys.modules, "entity", raising=False)
    monkeypatch.delitem(sys.modules, "entity.file_model", raising=False)

    spec = importlib.util.spec_from_file_location("direct_init_local_db", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert str(repository_root) in sys.path


def test_database_name_from_uri_accepts_mysql_database_name():
    uri = "mysql+pymysql://local-user:local-password@127.0.0.1:3306/fin_rag"

    assert database_name_from_uri(uri) == "fin_rag"


@pytest.mark.parametrize("database_name", ["finrag", "otherdb"])
def test_database_name_from_uri_rejects_unexpected_safe_database_name(database_name):
    uri = (
        "mysql+pymysql://local-user:local-password@127.0.0.1:3306/"
        f"{database_name}"
    )

    with pytest.raises(ValueError, match="must use database name 'fin_rag'"):
        database_name_from_uri(uri)


@pytest.mark.parametrize(
    "uri",
    [
        "mysql+pymysql://local-user:local-password@127.0.0.1:3306",
        "mysql+pymysql://local-user:local-password@127.0.0.1:3306/",
    ],
)
def test_database_name_from_uri_rejects_missing_database_name(uri):
    with pytest.raises(ValueError, match="database name"):
        database_name_from_uri(uri)


@pytest.mark.parametrize(
    "uri",
    [
        "mysql+pymysql://local-user:local-password@127.0.0.1:3306/fin-rag",
        "mysql+pymysql://local-user:local-password@127.0.0.1:3306/fin%20rag",
        "mysql+pymysql://local-user:local-password@127.0.0.1:3306/fin_rag;DROP",
    ],
)
def test_database_name_from_uri_rejects_unsafe_database_name(uri):
    with pytest.raises(ValueError, match="safe database name"):
        database_name_from_uri(uri)


def test_initialize_database_creates_database_then_file_schema(monkeypatch):
    uri = "mysql+pymysql://local-user:local-password@127.0.0.1:3306/fin_rag"
    server_engine = _FakeEngine()
    target_engine = _FakeEngine()
    created_engines = []
    schema_engines = []

    def fake_create_engine(engine_uri):
        created_engines.append(str(engine_uri))
        return (server_engine, target_engine)[len(created_engines) - 1]

    monkeypatch.setattr(init_local_db, "create_engine", fake_create_engine)
    monkeypatch.setattr(
        init_local_db.FileModel.metadata,
        "create_all",
        lambda engine: schema_engines.append(engine),
    )

    assert init_local_db.initialize_database(uri) == "fin_rag"

    assert created_engines == [
        "mysql+pymysql://local-user:***@127.0.0.1:3306",
        uri,
    ]
    assert server_engine.statements == [
        "CREATE DATABASE IF NOT EXISTS `fin_rag` "
        "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
    ]
    assert target_engine.statements == [
        "ALTER TABLE `file` MODIFY COLUMN `id` BIGINT NOT NULL COMMENT '文件ID'"
    ]
    assert schema_engines == [target_engine]
    assert server_engine.disposed
    assert target_engine.disposed


def test_file_model_uses_bigint_for_snowflake_ids():
    assert isinstance(FileModel.__table__.c.id.type, BigInteger)
    assert FileModel.__table__.c.id.autoincrement is False


def test_main_uses_configured_uri_and_prints_derived_database_name(monkeypatch, capsys):
    uri = "mysql+pymysql://local-user:local-password@127.0.0.1:3306/fin_rag"
    settings_module = ModuleType("utils.settings")
    settings_module.settings = SimpleNamespace(database_uri=uri)
    initialized_uris = []

    monkeypatch.setitem(sys.modules, "utils.settings", settings_module)
    monkeypatch.setattr(
        init_local_db,
        "initialize_database",
        lambda configured_uri: initialized_uris.append(configured_uri),
    )

    init_local_db.main()

    assert initialized_uris == [uri]
    assert capsys.readouterr().out == "Local MySQL database fin_rag is ready.\n"


class _FakeEngine:
    def __init__(self):
        self.statements = []
        self.disposed = False

    def begin(self):
        return nullcontext(self)

    def exec_driver_sql(self, statement):
        self.statements.append(statement)

    def dispose(self):
        self.disposed = True
