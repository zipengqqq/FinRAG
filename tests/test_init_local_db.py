import sys
from contextlib import nullcontext
from types import ModuleType, SimpleNamespace

import pytest

from scripts import init_local_db
from scripts.init_local_db import database_name_from_uri


def test_database_name_from_uri_accepts_mysql_database_name():
    uri = "mysql+pymysql://local-user:local-password@127.0.0.1:3306/fin_rag"

    assert database_name_from_uri(uri) == "fin_rag"


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
    assert schema_engines == [target_engine]
    assert server_engine.disposed
    assert target_engine.disposed


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
