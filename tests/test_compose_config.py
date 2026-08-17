from pathlib import Path

import yaml


COMPOSE_FILE = Path(__file__).parents[1] / "docker-compose.yml"


def _load_compose():
    return yaml.safe_load(COMPOSE_FILE.read_text(encoding="utf-8"))


def test_yaml_parser_accepts_comments_and_hyphenated_service_names():
    compose = yaml.safe_load(
        """# Local middleware
services:
  api-gateway: # Public API
    image: example/api
"""
    )

    assert compose["services"] == {"api-gateway": {"image": "example/api"}}


def test_compose_uses_local_only_ports_and_named_volumes_without_mysql():
    compose = _load_compose()
    services = compose["services"]

    assert "version" not in compose
    assert set(services) == {"etcd", "minio", "standalone"}
    assert "mysql" not in services
    assert compose["volumes"] == {
        "etcd_data": None,
        "minio_data": None,
        "milvus_data": None,
    }
    assert services["etcd"]["volumes"] == ["etcd_data:/etcd"]
    assert services["minio"]["volumes"] == ["minio_data:/minio_data"]
    assert services["standalone"]["volumes"] == ["milvus_data:/var/lib/milvus"]
    assert services["minio"]["ports"] == [
        "127.0.0.1:9001:9001",
        "127.0.0.1:9000:9000",
    ]
    assert services["standalone"]["ports"] == [
        "127.0.0.1:19530:19530",
        "127.0.0.1:9091:9091",
    ]
    assert services["minio"]["environment"] == {
        "MINIO_ROOT_USER": "${ACCESS_KEY:-minioadmin}",
        "MINIO_ROOT_PASSWORD": "${SECRET_KEY:-minioadmin}",
    }
    assert services["standalone"]["depends_on"] == {
        "etcd": {"condition": "service_healthy"},
        "minio": {"condition": "service_healthy"},
    }
