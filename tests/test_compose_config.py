import re
from pathlib import Path


COMPOSE_FILE = Path(__file__).parents[1] / "docker-compose.yml"


def test_compose_uses_local_only_ports_and_named_volumes_without_mysql():
    compose = COMPOSE_FILE.read_text(encoding="utf-8")
    services = compose.split("\nvolumes:\n", maxsplit=1)[0]

    assert "mysql:" not in compose.lower()
    assert re.findall(r"^  ([a-z_]+):$", services, flags=re.MULTILINE) == [
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
