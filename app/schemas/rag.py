from pydantic import BaseModel, Field
from typing import List, Dict, Optional


class QueryRequest(BaseModel):
    query: str
    top_k: int = 3
    retrieval_top_k: int = 30
    source: Optional[str] = Field(default=None, description="Source to filter by")
    user_id: str = Field(default="user_123", description="User ID for chat history")



class RetrievedDocument(BaseModel):
    content: str
    score: Optional[float] = None
    metadata: Dict = Field(default_factory=dict)


class QueryResponse(BaseModel):
    query: str
    answer: str
    retrieved_documents: List[RetrievedDocument]
    chat_history: List[Dict] = Field(default_factory=list, description="Recent chat history")