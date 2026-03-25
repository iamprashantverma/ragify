from haystack import Document
from typing import List
from app.haystack.pipelines.indexing_pipeline import indexing_pipeline

def ingest_data(texts: List[str], source: str) -> int:
    docs = [
        Document(content=text, meta={"source": source})
        for text in texts
    ]

    result = indexing_pipeline.run({
        "document_splitter": {"documents": docs}
    })
    
    return result["document_writer"]["documents_written"]