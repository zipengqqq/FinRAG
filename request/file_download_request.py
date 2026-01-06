from pydantic import BaseModel, Field


class FileDownloadRequest(BaseModel):
    file_id: str = Field(..., description="文件 ID")