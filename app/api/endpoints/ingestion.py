from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from typing import List, Optional
from PyPDF2 import PdfReader

from app.schemas.ingestion import IngestResponse
from app.services.ingestion_service import ingest_data

router = APIRouter()


@router.post("/ingest", response_model=IngestResponse)
async def ingest_documents(
    files: List[UploadFile] = File(...), 
    source: Optional[str] = Form(None)
):
    final_source = source if source else "public"
    all_texts = []

    try:
        for file in files:
            if not file.filename.endswith(".pdf"):
                raise HTTPException(
                    status_code=400,
                    detail=f"{file.filename} is not a PDF"
                )

            pdf = PdfReader(file.file)

            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    all_texts.append(text)

        if not all_texts:
            raise HTTPException(
                status_code=400,
                detail="No text found in uploaded PDFs"
            )
        count = ingest_data(all_texts, final_source)

        return IngestResponse(
            message="PDFs ingested successfully",
            documents_count=count
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))