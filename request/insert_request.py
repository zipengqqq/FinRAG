from typing import Any

from pydantic import BaseModel, Field


class InsertRequest(BaseModel):
    text: str
    source: str
    metadata: dict[str, Any] = Field(default_factory=dict)
