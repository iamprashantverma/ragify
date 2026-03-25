from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder

def get_document_embedder():
    return SentenceTransformersDocumentEmbedder(
        model="sentence-transformers/all-MiniLM-L6-v2",
        progress_bar=True
    )

def get_text_embedder():
    return SentenceTransformersTextEmbedder(
        model="sentence-transformers/all-MiniLM-L6-v2",
        progress_bar=False
    )