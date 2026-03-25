from typing import Dict, Optional
from app.haystack.pipelines.hybrid_retrieval import get_hybrid_pipeline


def retrieve_and_generate_hybrid(query: str, source: Optional[str] = None, top_k: int = 10, retrieval_top_k: int = 30) -> Dict:
    hybrid_pipeline = get_hybrid_pipeline(top_k=top_k, retrieval_top_k=retrieval_top_k)

    filters = None
    if source and source.strip():
        filters = {
            "field": "meta.source",
            "operator": "==",
            "value": source
        }

    run_data = {
        "text_embedder": {"text": query},
        "bm25_retriever": {"query": query},
        "ranker":         {"query": query},
    }

    if filters:
        run_data["embedding_retriever"] = {"filters": filters}
        run_data["bm25_retriever"]["filters"] = filters

    result = hybrid_pipeline.run(data=run_data, include_outputs_from=["ranker"])

    documents = result.get("ranker", {}).get("documents", [])

    return {
        "query": query,
        "retrieved_documents": [
            {
                "content":  doc.content,
                "score":    getattr(doc, "score", None),
                "metadata": getattr(doc, "meta", {}),
            }
            for doc in documents
        ],
    }