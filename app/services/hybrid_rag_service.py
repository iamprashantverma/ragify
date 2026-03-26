from typing import Dict, Optional
from app.haystack.pipelines.hybrid_retrieval import get_hybrid_pipeline
from app.services.chat_history_service import add_to_history, get_history, format_history_for_context


def retrieve_and_generate_hybrid(
    query: str,
    user_id: str, 
    source: Optional[str] = None, 
    top_k: int = 10, 
    retrieval_top_k: int = 30
) -> Dict:
    hybrid_pipeline = get_hybrid_pipeline(top_k=top_k, retrieval_top_k=retrieval_top_k)

    # Get chat history context
    history_context = format_history_for_context(user_id, limit=3)
    
    # Enhance query with history context if available
    enhanced_query = query
    if history_context:
        enhanced_query = f"{history_context}\nCurrent query: {query}"

    filters = None
    if source and source.strip():
        filters = {
            "field": "meta.source",
            "operator": "==",
            "value": source
        }

    run_data = {
        "text_embedder": {"text": enhanced_query},
        "bm25_retriever": {"query": enhanced_query},
        "ranker":         {"query": query},  # Use original query for ranking
    }

    if filters:
        run_data["embedding_retriever"] = {"filters": filters}
        run_data["bm25_retriever"]["filters"] = filters

    result = hybrid_pipeline.run(data=run_data, include_outputs_from=["ranker"])

    documents = result.get("ranker", {}).get("documents", [])
    
    # Format response summary for history
    response_summary = f"Found {len(documents)} relevant documents"
    if documents:
        response_summary += f". Top result: {documents[0].content[:100]}..."
    
    # Add to chat history
    add_to_history(user_id, query, response_summary)
    
    # Get updated history
    chat_history = get_history(user_id, limit=5)

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
        "chat_history": chat_history
    }