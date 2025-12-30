from pydantic import BaseModel, Field


class DocumentRequest(BaseModel):
    file_name: str = Field(None, description="文档名称")
    status: int = Field(None, description="文件状态；0文件待解析，1文件正在解析，2文件解析成功，3文件解析失败")