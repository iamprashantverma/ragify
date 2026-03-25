from haystack_integrations.components.retrievers.elasticsearch import ElasticsearchBM25Retriever
from app.haystack.document_store.elastic import document_store


def get_bm25_retriever(top_k: int = 10):
    return ElasticsearchBM25Retriever(document_store=document_store, top_k=top_k)
