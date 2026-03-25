from haystack.components.rankers import SentenceTransformersSimilarityRanker


def get_ranker(top_k: int = 5, model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
    return SentenceTransformersSimilarityRanker(model=model, top_k=top_k)
