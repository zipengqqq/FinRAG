from typing import Optional

from pydantic import BaseModel


class SearchRequest(BaseModel):
    query: str
    year: Optional[int] = None
    source: Optional[str] = None
    top_k: int = 5