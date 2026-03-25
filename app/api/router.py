from fastapi import APIRouter
from app.api.endpoints import ingestion, rag

api_router = APIRouter()

api_router.include_router(ingestion.router, prefix="/ingestion", tags=["ingestion"])
api_router.include_router(rag.router, prefix="/rag", tags=["rag"])
