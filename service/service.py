from entity.file_model import FileModel
from request.document_request import DocumentRequest
from request.file_download_request import FileDownloadRequest
from utils.db_util import create_session
from utils.logger_util import logger
from utils.minio_util import minio_client, BUCKET_NAME


class Service():
    def list(self, req: DocumentRequest):
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

    def file_download(self, req: FileDownloadRequest):
        logger.info(f"文件下载请求, file_id={req.file_id}")
        try:
            file_id = int(req.file_id)
        except ValueError:
            logger.error(f"无效的 file_id: {req.file_id}")
            return None

        with create_session() as session:
            record = session.query(FileModel).filter(
                FileModel.id == file_id,
                FileModel.delete_flag == 0
            ).first()

            if not record:
                logger.error(f"未找到文件记录, file_id={file_id}")
                return None

            minio_url = record.minio_url or ""
            file_name = record.file_name
            file_type = record.type

        if not minio_url:
            logger.error(f"文件 minio_url 为空, file_id={file_id}")
            return None

        bucket = BUCKET_NAME
        if minio_url.startswith(f"{bucket}/"):
            object_name = minio_url[len(bucket) + 1 :]
        else:
            object_name = minio_url

        logger.info(f"文件是 {bucket}/{object_name}")

        response = minio_client.get_object(bucket, object_name)

        def file_iterator():
            try:
                for data in response.stream(32 * 1024):
                    yield data
            finally:
                response.close()
                response.release_conn()

        content_type = "application/pdf" if file_type == 0 else "application/octet-stream"

        return {
            "iterator": file_iterator(),
            "file_name": file_name,
            "content_type": content_type,
        }
