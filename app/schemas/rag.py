from pydantic import BaseModel, Field
from typing import List, Dict, Optional


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    retrieval_top_k: int = 20
    source: Optional[str] = Field(default=None, description="Source to filter by")



class RetrievedDocument(BaseModel):
    content: str
    score: Optional[float] = None
    metadata: Dict = Field(default_factory=dict)


class QueryResponse(BaseModel):
    query: str
    answer: str
    retrieved_documents: List[RetrievedDocument]