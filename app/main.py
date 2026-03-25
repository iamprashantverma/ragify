from fastapi import FastAPI
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from app.haystack.pipelines.indexing_pipeline import get_indexing_pipeline
from app.haystack.pipelines.hybrid_retrieval import get_hybrid_pipeline
from contextlib import asynccontextmanager
load_dotenv()

from app.api.router import api_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    get_indexing_pipeline()
    get_hybrid_pipeline()
    yield

app = FastAPI(title="RAG-Application", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)

@app.get("/")
async def root():
    return {"message": "RAG App is running"}