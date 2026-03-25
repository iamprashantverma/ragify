from haystack_integrations.components.retrievers.elasticsearch import ElasticsearchEmbeddingRetriever
from app.haystack.document_store.elastic import document_store


def get_dense_retriever(top_k: int = 5):
    return ElasticsearchEmbeddingRetriever(document_store=document_store, top_k=top_k)
