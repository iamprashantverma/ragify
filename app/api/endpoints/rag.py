from fastapi import APIRouter, HTTPException
from app.schemas.rag import QueryRequest, QueryResponse
from app.services.hybrid_rag_service import retrieve_and_generate_hybrid

router = APIRouter()


@router.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    result = retrieve_and_generate_hybrid(
        request.query,
        request.source,
        request.top_k,
        request.retrieval_top_k,
    )

    return QueryResponse(**result)