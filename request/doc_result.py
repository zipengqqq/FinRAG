from typing import Optional

from pydantic import BaseModel


class DocResult(BaseModel):
    content: str
    rerank_score: Optional[float] = None
    source: Optional[str] = None
    section: Optional[str] = None
    year: Optional[int] = None