from typing import Optional

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    query: str = Field(..., description="查询内容")
    conversation_id: str = Field(None, description="会话 ID")