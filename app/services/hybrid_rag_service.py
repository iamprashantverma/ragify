from typing import Dict, Optional, List
from haystack.dataclasses import ChatRole
from app.haystack.pipelines.chat_retrieval import get_chat_retrieval_pipeline
from app.services.chat_history_service import get_chat_store


def retrieve_and_generate_hybrid(query: str,user_id: str, source: Optional[str] = None, top_k: int = 3, retrieval_top_k: int = 30) -> Dict:
    pipeline = get_chat_retrieval_pipeline(top_k=top_k, retrieval_top_k=retrieval_top_k)

    chat_history_id = f"user_{user_id}_session"

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
        "ranker": {"query": query},
        "prompt_builder": {"query": query},
        "message_retriever": {"chat_history_id": chat_history_id},
        "message_writer": {"chat_history_id": chat_history_id},
    }

    if filters:
        run_data["embedding_retriever"] = {"filters": filters}
        run_data["bm25_retriever"]["filters"] = filters

    result = pipeline.run(
        data=run_data,
        include_outputs_from=["ranker", "llm", "message_retriever"]
    )

    documents = result.get("ranker", {}).get("documents", [])
    replies = result.get("llm", {}).get("replies", [])
    answer = replies[0].text if replies else "No answer generated"


    history_messages = result.get("message_retriever", {}).get("messages", [])
    print(f"DEBUG: Retrieved {len(history_messages)} messages from history for {chat_history_id}")


    store = get_chat_store()
    all_messages = store.retrieve_messages(chat_history_id)
    print(f"DEBUG: Store has {len(all_messages)} total messages for {chat_history_id}")


    user_msgs = [m for m in history_messages if m.role == ChatRole.USER]
    asst_msgs = [m for m in history_messages if m.role == ChatRole.ASSISTANT]

    chat_history = [
        {"query": u.text, "response": a.text}
        for u, a in zip(user_msgs, asst_msgs)
    ]

    return {
        "query": query,
        "answer": answer,
        "retrieved_documents": [
            {
                "content": doc.content,
                "score": getattr(doc, "score", None),
                "metadata": getattr(doc, "meta", {}),
            }
            for doc in documents
        ],
        "chat_history": chat_history[-5:] 
    }