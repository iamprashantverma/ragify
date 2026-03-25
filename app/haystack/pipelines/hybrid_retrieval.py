import threading
from haystack import Pipeline
from haystack.components.joiners import DocumentJoiner
from app.haystack.retrievers.dense import get_dense_retriever
from app.haystack.retrievers.bm25 import get_bm25_retriever
from app.haystack.rankers.cross_encoder import get_ranker
from app.haystack.processors.embedding import get_text_embedder

_lock = threading.Lock()

def create_hybrid_pipeline(top_k: int = 10, retrieval_top_k: int = 30):
    hybrid_retrieval = Pipeline()
    hybrid_retrieval.add_component("text_embedder", get_text_embedder())
    hybrid_retrieval.add_component("embedding_retriever", get_dense_retriever(top_k=retrieval_top_k))
    hybrid_retrieval.add_component("bm25_retriever", get_bm25_retriever(top_k=retrieval_top_k))
    hybrid_retrieval.add_component("document_joiner", DocumentJoiner())
    hybrid_retrieval.add_component("ranker", get_ranker(top_k=top_k))
    hybrid_retrieval.connect("text_embedder.embedding", "embedding_retriever.query_embedding")
    hybrid_retrieval.connect("embedding_retriever", "document_joiner")
    hybrid_retrieval.connect("bm25_retriever", "document_joiner")
    hybrid_retrieval.connect("document_joiner", "ranker.documents")
    return hybrid_retrieval

_pipeline = None

def get_hybrid_pipeline(top_k: int = 10, retrieval_top_k: int = 30):
    global _pipeline
    if _pipeline is None:
        with _lock:
            if _pipeline is None:
                _pipeline = create_hybrid_pipeline(top_k=top_k, retrieval_top_k=retrieval_top_k)
                _pipeline.warm_up()
    return _pipeline