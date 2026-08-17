import importlib

import pytest


REQUIRED_VALUES = {
    "DATABASE_URI": "mysql+pymysql://env-user:env-password@127.0.0.1:3306/finrag",
    "ENDPOINT": "127.0.0.1:9000",
    "ACCESS_KEY": "env-access-key",
    "SECRET_KEY": "env-secret-key",
    "BUCKET_NAME": "financial-reports",
}


@pytest.fixture
def settings_module(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text(
        "DATABASE_URI=mysql+pymysql://dotenv-user:dotenv-password@localhost/dotenv\n"
        "ENDPOINT=dotenv-endpoint:9000\n"
        "ACCESS_KEY=dotenv-access-key\n"
        "SECRET_KEY=dotenv-secret-key\n"
        "BUCKET_NAME=dotenv-bucket\n",
        encoding="utf-8",
    )
    for name, value in REQUIRED_VALUES.items():
        monkeypatch.setenv(name, value)

    module = importlib.import_module("utils.settings")
    return importlib.reload(module)


def test_read_settings_prefers_existing_environment_and_defaults_milvus(settings_module):
    values = settings_module.read_settings()

    assert values.database_uri == REQUIRED_VALUES["DATABASE_URI"]
    assert values.minio_endpoint == REQUIRED_VALUES["ENDPOINT"]
    assert values.minio_access_key == REQUIRED_VALUES["ACCESS_KEY"]
    assert values.minio_secret_key == REQUIRED_VALUES["SECRET_KEY"]
    assert values.bucket_name == REQUIRED_VALUES["BUCKET_NAME"]
    assert values.milvus_host == "127.0.0.1"
    assert values.milvus_port == "19530"
    assert values.milvus_uri == "http://127.0.0.1:19530"


def test_read_settings_uses_explicit_milvus_values(settings_module, monkeypatch):
    monkeypatch.setenv("MILVUS_HOST", "milvus.local")
    monkeypatch.setenv("MILVUS_PORT", "25000")

    values = settings_module.read_settings()

    assert values.milvus_host == "milvus.local"
    assert values.milvus_port == "25000"
    assert values.milvus_uri == "http://milvus.local:25000"


@pytest.mark.parametrize("missing_name", REQUIRED_VALUES)
def test_read_settings_identifies_missing_required_variable(
    settings_module, monkeypatch, tmp_path, missing_name
):
    empty_directory = tmp_path / "without-dotenv"
    empty_directory.mkdir()
    monkeypatch.chdir(empty_directory)
    monkeypatch.delenv(missing_name)

    with pytest.raises(RuntimeError, match=missing_name):
        settings_module.read_settings()
