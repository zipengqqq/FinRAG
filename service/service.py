from entity.file_model import FileModel
from request.document_request import DocumentRequest
from utils.db_util import create_session


class Service():
    def list(self, req: DocumentRequest):
        """查看文档解析情况"""
        file_name = req.file_name
        status = req.status
        with create_session() as session:
            query = session.query(FileModel).filter(FileModel.delete_flag == 0)
            if file_name:
                query = query.filter(FileModel.file_name.like(f"%{file_name}%"))
            if status is not None:
                query = query.filter(FileModel.status == status)
            records = query.order_by(FileModel.create_time.desc()).all()
            return [record.to_dict() for record in records]