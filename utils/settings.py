import os
from dataclasses import dataclass
from pathlib import Path

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


def read_settings() -> Settings:
    load_dotenv(dotenv_path=Path.cwd() / ".env", override=False)

    required_variables = (
        "DATABASE_URI",
        "ENDPOINT",
        "ACCESS_KEY",
        "SECRET_KEY",
        "BUCKET_NAME",
    )
    values = {name: os.getenv(name) for name in required_variables}
    for name, value in values.items():
        if not value:
            raise RuntimeError(f"Missing required environment variable: {name}")

    return Settings(
        database_uri=values["DATABASE_URI"],
        minio_endpoint=values["ENDPOINT"],
        minio_access_key=values["ACCESS_KEY"],
        minio_secret_key=values["SECRET_KEY"],
        bucket_name=values["BUCKET_NAME"],
        milvus_host=os.getenv("MILVUS_HOST", "127.0.0.1"),
        milvus_port=os.getenv("MILVUS_PORT", "19530"),
    )


settings = read_settings()
