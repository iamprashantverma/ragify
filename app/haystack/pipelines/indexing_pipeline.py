import threading
from haystack import Pipeline
from haystack.components.writers import DocumentWriter
from haystack.document_stores.types import DuplicatePolicy 
from app.haystack.processors.splitter import get_document_splitter
from app.haystack.processors.embedding import get_document_embedder
from app.haystack.document_store.elastic import document_store

_lock = threading.Lock()

def create_indexing_pipeline():
    indexing_pipeline = Pipeline()
    indexing_pipeline.add_component("document_splitter", get_document_splitter())
    indexing_pipeline.add_component("document_embedder", get_document_embedder())
    indexing_pipeline.add_component("document_writer", DocumentWriter(document_store=document_store, policy=DuplicatePolicy.OVERWRITE))
    indexing_pipeline.connect("document_splitter", "document_embedder")
    indexing_pipeline.connect("document_embedder", "document_writer")
    return indexing_pipeline

indexing_pipeline = None

def get_indexing_pipeline():
    global indexing_pipeline
    if indexing_pipeline is None:
        with _lock:
            if indexing_pipeline is None:
                indexing_pipeline = create_indexing_pipeline()
                indexing_pipeline.warm_up()
    return indexing_pipeline