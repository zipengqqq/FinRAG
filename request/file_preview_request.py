from pydantic import BaseModel, Field


class FilePreviewRequest(BaseModel):
    file_id: str = Field(..., description="文件 ID")

