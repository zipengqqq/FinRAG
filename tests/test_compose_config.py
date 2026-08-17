import re
from pathlib import Path


COMPOSE_FILE = Path(__file__).parents[1] / "docker-compose.yml"


def _service_names(compose):
    names = []
    in_services = False

    for line in compose.splitlines():
        if line == "services:":
            in_services = True
            continue
        if not in_services:
            continue
        if line and not line.startswith(" "):
            break
        if not line.startswith("  ") or line.startswith("   "):
            continue

        name, separator, _ = line[2:].partition(":")
        if separator and name:
            names.append(name.strip().strip('"\''))

    return names


def test_service_name_extraction_accepts_compose_identifier_characters():
    compose = """services:
  api-gateway:
    image: example/api
  worker.v2:
    image: example/worker
  job_runner:
    image: example/job
volumes:
  local_data:
"""

    assert _service_names(compose) == ["api-gateway", "worker.v2", "job_runner"]


def test_compose_uses_local_only_ports_and_named_volumes_without_mysql():
    compose = COMPOSE_FILE.read_text(encoding="utf-8")

    assert "mysql:" not in compose.lower()
    assert _service_names(compose) == [
        "etcd",
        "minio",
        "standalone",
    ]
    assert "etcd_data:/etcd" in compose
    assert "minio_data:/minio_data" in compose
    assert "milvus_data:/var/lib/milvus" in compose
    assert "etcd_data:" in compose
    assert "minio_data:" in compose
    assert "milvus_data:" in compose
    assert '"127.0.0.1:9000:9000"' in compose
    assert '"127.0.0.1:9001:9001"' in compose
    assert '"127.0.0.1:19530:19530"' in compose
    assert '"127.0.0.1:9091:9091"' in compose
    assert "MINIO_ROOT_USER: ${ACCESS_KEY:-minioadmin}" in compose
    assert "MINIO_ROOT_PASSWORD: ${SECRET_KEY:-minioadmin}" in compose
    assert re.search(
        r"    depends_on:\n"
        r"      etcd:\n"
        r"        condition: service_healthy\n"
        r"      minio:\n"
        r"        condition: service_healthy",
        compose,
    )
