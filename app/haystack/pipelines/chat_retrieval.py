from haystack import Pipeline
from haystack.components.builders import ChatPromptBuilder
from haystack.components.converters import OutputAdapter
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.joiners import DocumentJoiner, BranchJoiner
from haystack.dataclasses import ChatMessage
from haystack.utils import Secret
from haystack_experimental.components.retrievers.chat_message_retriever import ChatMessageRetriever
from haystack_experimental.components.writers.chat_message_writer import ChatMessageWriter

from app.haystack.retrievers.dense import get_dense_retriever
from app.haystack.retrievers.bm25 import get_bm25_retriever
from app.haystack.rankers.cross_encoder import get_ranker
from app.haystack.processors.embedding import get_text_embedder
from app.services.chat_history_service import get_chat_store


def create_chat_retrieval_pipeline(top_k: int = 3, retrieval_top_k: int = 30):
    pipeline = Pipeline()

    message_store = get_chat_store()

    # Document retrieval components
    pipeline.add_component("text_embedder", get_text_embedder())
    pipeline.add_component("embedding_retriever", get_dense_retriever(top_k=retrieval_top_k))
    pipeline.add_component("bm25_retriever", get_bm25_retriever(top_k=retrieval_top_k))
    pipeline.add_component("document_joiner", DocumentJoiner())
    pipeline.add_component("ranker", get_ranker(top_k=top_k))

    # Chat history components — use keyword args
    pipeline.add_component("message_retriever", ChatMessageRetriever(chat_message_store=message_store))
    pipeline.add_component("message_writer", ChatMessageWriter(chat_message_store=message_store))

    # Prompt builder
    pipeline.add_component(
        "prompt_builder",
        ChatPromptBuilder(
            template=[
                ChatMessage.from_system(
                    "You are a helpful AI assistant. Answer questions based on the provided "
                    "context documents. If the context doesn't contain enough information, say so clearly."
                ),
                ChatMessage.from_user(
                    "Context:\n{% for doc in documents %}{{ doc.content }}\n\n{% endfor %}"
                    "\nQuestion: {{query}}"
                ),
            ],
            required_variables=["documents", "query"],
        ),
    )

    # LLM — receives history + current prompt merged
    pipeline.add_component(
        "llm",
        OpenAIChatGenerator(
            api_key=Secret.from_env_var("OPENAI_API_KEY"),
            model="gpt-4o",
            generation_kwargs={"temperature": 0.0, "max_tokens": 600},
        ),
    )

    # Merge history messages + current prompt into one list for LLM
    pipeline.add_component(
        "message_joiner",
        OutputAdapter(
            template="{{ history + prompt }}",   
            output_type=list[ChatMessage],
            unsafe=True,
        ),
    )

    # Store both the prompt + LLM reply into history
    pipeline.add_component(
        "history_joiner",
        OutputAdapter(
            template="{{ prompt + replies }}",
            output_type=list[ChatMessage],
            unsafe=True,
        ),
    )

    # --- Retrieval connections ---
    pipeline.connect("text_embedder.embedding", "embedding_retriever.query_embedding")
    pipeline.connect("embedding_retriever", "document_joiner")
    pipeline.connect("bm25_retriever", "document_joiner")
    pipeline.connect("document_joiner", "ranker.documents")

    # --- Prompt building ---
    pipeline.connect("ranker.documents", "prompt_builder.documents")

    # --- Merge history + prompt → LLM ---
    pipeline.connect("message_retriever.messages", "message_joiner.history")
    pipeline.connect("prompt_builder.prompt", "message_joiner.prompt")
    pipeline.connect("message_joiner.output", "llm.messages")

    # --- Store conversation in history ---
    pipeline.connect("prompt_builder.prompt", "history_joiner.prompt")
    pipeline.connect("llm.replies", "history_joiner.replies")
    pipeline.connect("history_joiner.output", "message_writer.messages")

    return pipeline


_pipeline = None

def get_chat_retrieval_pipeline(top_k: int = 3, retrieval_top_k: int = 30):
    global _pipeline
    if _pipeline is None:
        _pipeline = create_chat_retrieval_pipeline(top_k=top_k, retrieval_top_k=retrieval_top_k)
        _pipeline.warm_up()
    return _pipeline