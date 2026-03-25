from pydantic import BaseModel
from typing import List


class IngestResponse(BaseModel):
    message: str
    documents_count: int
