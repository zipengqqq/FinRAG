from pydantic import BaseModel


class InsertRequest(BaseModel):
    text: str
    source: str
    year: int