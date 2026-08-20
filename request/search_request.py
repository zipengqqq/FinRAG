from typing import Any, Optional

from pydantic import BaseModel


class SearchRequest(BaseModel):
    query: str
    source: Optional[str] = None
    filters: Optional[dict[str, Any]] = None
    top_k: int = 5
