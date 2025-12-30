from entity.file_model import FileModel
from utils.db_util import create_session


class Service():
    def list(self):
        """查看文档解析情况"""
        with create_session() as session:
            records = session.query(FileModel).all()
            return [record.to_dict() for record in records]