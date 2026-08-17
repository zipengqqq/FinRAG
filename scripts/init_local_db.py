import re

from sqlalchemy import create_engine
from sqlalchemy.engine import make_url

from entity.file_model import FileModel


def database_name_from_uri(uri: str) -> str:
    database_name = make_url(uri).database
    if not database_name:
        raise ValueError("Database URI must include a database name.")
    if re.fullmatch(r"[A-Za-z0-9_]+", database_name) is None:
        raise ValueError("Database URI must include a safe database name.")
    return database_name


def initialize_database(uri: str) -> str:
    database_name = database_name_from_uri(uri)
    server_uri = make_url(uri)._replace(database=None)
    server_engine = create_engine(server_uri)
    try:
        with server_engine.begin() as connection:
            connection.exec_driver_sql(
                f"CREATE DATABASE IF NOT EXISTS `{database_name}` "
                "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
            )
    finally:
        server_engine.dispose()

    target_engine = create_engine(uri)
    try:
        FileModel.metadata.create_all(target_engine)
    finally:
        target_engine.dispose()
    return database_name


def main() -> None:
    from utils.settings import settings

    database_name = database_name_from_uri(settings.database_uri)
    initialize_database(settings.database_uri)
    print(f"Local MySQL database {database_name} is ready.")


if __name__ == "__main__":
    main()
