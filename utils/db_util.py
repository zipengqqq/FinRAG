from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from utils.logger_util import logger
from utils.settings import settings

engine = create_engine(
    settings.database_uri,
    pool_pre_ping=True,
    pool_recycle=3600,
    echo=False
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


class DatabaseManager:
    def __init__(self):
        self.engine = engine
        self.SessionLocal = SessionLocal

    @contextmanager
    def get_session(self):
        """获取数据库会话的上下文管理器"""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Database error: {e}")
            raise
        except Exception as e:
            session.rollback()
            logger.error(f"Unexpected error: {e}")
            raise
        finally:
            session.close()


db_manager = DatabaseManager()

def create_session():
    return db_manager.get_session()
