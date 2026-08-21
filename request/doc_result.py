from typing import Any, Optional

from pydantic import BaseModel, Field


class DocResult(BaseModel):
    content: str
    rerank_score: Optional[float] = None
    source: Optional[str] = None
    section: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)
