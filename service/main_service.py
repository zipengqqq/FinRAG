import json
import subprocess
import sys
from entity.file_model import FileModel
from request.document_request import DocumentRequest
from request.file_download_request import FileDownloadRequest
from request.file_preview_request import FilePreviewRequest
from utils.db_util import create_session
from utils.id_util import id_worker
from utils.logger_util import logger
from utils.minio_util import minio_client, BUCKET_NAME
from starlette.responses import StreamingResponse
from urllib.parse import quote
from pathlib import Path
from datetime import datetime
from chunker import split_md_content
from vector_store import add_documents_to_milvus


class Service():
    def list(self, req: DocumentRequest):
        file_name = req.file_name
        status = req.status
        page = req.page or 1
        page_size = req.page_size or 10
        if page < 1:
            page = 1
        if page_size < 1:
            page_size = 10
        with create_session() as session:
            base_query = session.query(FileModel).filter(FileModel.delete_flag == 0)
            if file_name:
                base_query = base_query.filter(FileModel.file_name.like(f"%{file_name}%"))
            if status is not None:
                base_query = base_query.filter(FileModel.status == status)
            total = base_query.count()
            query = base_query.order_by(FileModel.create_time.desc())
            records = query.limit(page_size).offset((page - 1) * page_size).all()
            items = [record.to_dict() for record in records]
            return {
                "items": items,
                "total": total,
                "page": page,
                "page_size": page_size
            }

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

        encoded_file_name = quote(file_name, safe='')
        content_disposition = "attachment; filename*=UTF-8''{}".format(encoded_file_name)

        return StreamingResponse(
            file_iterator(),
            media_type=content_type,
            headers={"Content-Disposition": content_disposition}
        )

    def file_preview(self, req: FilePreviewRequest):
        logger.info(f"文件预览请求, file_id={req.file_id}")
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

        encoded_file_name = quote(file_name, safe='')
        content_disposition = "inline; filename*=UTF-8''{}".format(encoded_file_name)

        return StreamingResponse(
            file_iterator(),
            media_type="application/pdf",
            headers={"Content-Disposition": content_disposition}
        )

    def upload_file_async(self, filename: str, file_content: bytes, content_type: str):
        """
        后台异步处理文件上传和解析
        
        处理流程:
        1. 上传文件到 MinIO 存储
        2. 创建数据库记录（状态: 处理中）
        3. 如果是 PDF，进行解析和向量化
        4. 更新数据库状态（成功/失败）
        
        Args:
            filename: 原始文件名
            file_content: 文件二进制内容
            content_type: 文件 MIME 类型
        """
        from io import BytesIO
        uid = None
        try:
            logger.info(f"开始处理文件: {filename}")
            ext = (Path(filename).suffix or "").lower()
            is_pdf = ext == ".pdf"
            uid = id_worker.get_id()
            bucket = BUCKET_NAME
            
            # 确保 bucket 存在
            if not minio_client.bucket_exists(bucket):
                minio_client.make_bucket(bucket)
            
            # 1. 上传到 MinIO
            size_bytes = len(file_content)
            object_name = f"uploads/{uid}/{filename}"
            minio_content_type = "application/pdf" if is_pdf else (content_type or "application/octet-stream")
            minio_client.put_object(bucket, object_name, BytesIO(file_content), length=size_bytes, content_type=minio_content_type)
            minio_url = f"{bucket}/{object_name}"
            logger.info(f"文件已上传到 MinIO: {minio_url}")
            
            # 2. 创建数据库记录
            self._step_create_db_record(uid, filename, size_bytes, is_pdf, minio_url)
            
            # 3. 非 PDF 文件不进行解析
            if not is_pdf:
                self._step_update_status_by_id(uid, 3)
                logger.warning(f"非 PDF 文件，跳过解析: {filename}")
                return
            
            # 4. 保存到本地临时目录
            local_dir = Path("tmp") / str(uid)
            local_dir.mkdir(parents=True, exist_ok=True)
            local_pdf_path = local_dir / filename
            local_pdf_path.write_bytes(file_content)
            
            # 5. PDF 解析转 Markdown
            logger.info(f"开始解析 PDF: {filename}")
            md_path = self._step_parse_pdf(local_pdf_path)
            
            # 6. 向量化并存入 Milvus
            logger.info(f"开始向量化: {filename}")
            self._step_ingest_md(md_path, filename)
            
            # 7. 更新状态为成功
            self._step_update_status_by_id(uid, 2)
            logger.info(f"文件处理完成: {filename}")
            
        except Exception as e:
            logger.error(f"文件处理失败 [{filename}]: {e}")
            if uid:
                try:
                    self._step_update_status_by_id(uid, 3)
                except Exception:
                    pass

    def _step_create_db_record(self, uid: int, filename: str, size_bytes: int, is_pdf: bool, minio_url: str):
        with create_session() as session:
            record = FileModel(
                id=uid,
                file_name=filename,
                status=1,
                size=round(size_bytes / (1024 * 1024), 2),
                type=0 if is_pdf else 1,
                create_time=datetime.now(),
                delete_flag=0,
                minio_url=minio_url
            )
            session.add(record)

    def _step_parse_pdf(self, local_pdf_path: Path) -> str:
        result_path = local_pdf_path.with_suffix(".marker-result.json")
        if result_path.exists():
            result_path.unlink()

        # 调用独立 Marker 解析进程。
        command = [
            sys.executable,
            str(Path(__file__).resolve().parent.parent / "marker_parse.py"),
            str(local_pdf_path),
            "output",
            str(result_path),
        ]
        # 继承标准输出和错误输出，实时显示 Marker 解析日志。
        completed = subprocess.run(command, check=False)

        if not result_path.exists():
            raise RuntimeError(f"Marker worker failed: exit code {completed.returncode}")

        result = json.loads(result_path.read_text(encoding="utf-8"))
        if not result.get("ok"):
            raise RuntimeError(f"Marker worker failed: {result.get('error', 'unknown error')}")

        if completed.returncode != 0:
            raise RuntimeError(f"Marker worker failed: exit code {completed.returncode}")

        markdown_path = Path(result["markdown_path"])
        if not markdown_path.exists():
            raise RuntimeError(f"Marker worker output does not exist: {markdown_path}")
        return str(markdown_path)

    def _step_ingest_md(self, md_path: str, filename: str):
        text = Path(md_path).read_text(encoding="utf-8")
        chunks = split_md_content(text, source_filename=filename, metadata={})
        add_documents_to_milvus(chunks)

    def _step_update_status_by_id(self, uid: int, status: int):
        with create_session() as session:
            session.query(FileModel).filter(FileModel.id == uid).update({"status": status})
