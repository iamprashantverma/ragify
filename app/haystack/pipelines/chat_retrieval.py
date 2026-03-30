from haystack import Pipeline
from haystack.components.builders import ChatPromptBuilder
from haystack.components.converters import OutputAdapter
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.joiners import DocumentJoiner
from haystack.dataclasses import ChatMessage
from haystack.utils import Secret

from app.haystack.retrievers.dense import get_dense_retriever
from app.haystack.retrievers.bm25 import get_bm25_retriever
from app.haystack.rankers.cross_encoder import get_ranker
from app.haystack.processors.embedding import get_text_embedder
from app.services.chat_history_service import get_message_store
from haystack_experimental.components.retrievers import ChatMessageRetriever
from haystack_experimental.components.writers import ChatMessageWriter


def create_chat_retrieval_pipeline(top_k: int = 3, retrieval_top_k: int = 30):

    message_store = get_message_store()

    pipeline = Pipeline()

    # RAG components 
    pipeline.add_component("text_embedder", get_text_embedder())
    pipeline.add_component("embedding_retriever", get_dense_retriever(top_k=retrieval_top_k))
    pipeline.add_component("bm25_retriever", get_bm25_retriever(top_k=retrieval_top_k))
    pipeline.add_component("document_joiner", DocumentJoiner())
    pipeline.add_component("ranker", get_ranker(top_k=top_k))

    pipeline.add_component(
    "prompt_builder",
    ChatPromptBuilder(
        template=[
            ChatMessage.from_system(
                "You are a knowledgeable assistant.\n\n"

                "You will receive:\n"
                "1. Prior conversation history — previous turns of this conversation.\n"
                "2. Context documents — retrieved knowledge relevant to the query.\n"
                "3. The current user query.\n\n"

                "## How to Use Each Source\n\n"

                "### Conversation History\n"
                "- Use history to understand references, pronouns, conversational intent, and user-provided facts.\n"
                "- If history contains a clear, relevant answer or explicitly stated facts → use them directly.\n"
                "- Treat history as TRUSTED but potentially incomplete or outdated.\n\n"
                "### Context Documents\n"
                "- Use documents as the PRIMARY factual source.\n"
                "- If documents provide a more complete or updated answer → prefer documents over history.\n"
                "- If documents and history AGREE → combine them for a richer answer.\n"
                "- If documents and history CONFLICT → always prefer documents, and note the conflict.\n\n"

                "### Combining Both\n"
                "- Use history and query to understand WHAT is being asked.\n"
                "- Use documents and history to determine WHAT the answer is.\n"
                "- When both support the answer, synthesize naturally — do not repeat the same point twice.\n\n"

                "## Strict Rules\n"
                "- Do NOT hallucinate or add anything not present in history or documents.\n"
                "- If answer is fully supported → provide complete answer.\n"
                "- If partially supported → state clearly what is available and what is missing.\n"
                "- If not supported by either → say it is out of scope.\n"
                "- For greetings or casual queries → respond naturally and briefly.\n\n"

                "## Classification\n"
                "- Answerable → Fully supported by documents and/or history\n"
                "- Needs More Information → Partially supported\n"
                "- Out of Scope → Not supported by either source\n\n"

                "## Priority\n"
                "- High → Legal, financial, critical decisions\n"
                "- Medium → Important informational queries\n"
                "- Low → Casual or general queries\n\n"

                "## Citations\n"
                "- Cite documents like: [Document 1], [Document 2]\n"
                "- Do NOT cite conversation history — it is context, not a source\n"
            ),

            ChatMessage.from_user(
                "{% if history %}"
                "### Conversation History:\n"
                "{% for message in history %}"
                "{{ message.role | capitalize }}: {{ message.content }}\n"
                "{% endfor %}"
                "\n"
                "{% endif %}"

                "### Context Documents:\n"
                "{% for doc in documents %}"
                "--- Document {{ loop.index }} ---\n{{ doc.content }}\n\n"
                "{% endfor %}"

                "### Current Query:\n{{ query }}\n\n"

                "### Answering Logic (follow in order):\n"
                "1. Check if history already answers the query → if yes, use it.\n"
                "2. Check if documents answer or add to it → if yes, prefer or combine.\n"
                "3. If both agree → synthesize into one answer.\n"
                "4. If they conflict → use documents, mention the conflict briefly.\n"
                "5. If neither answers → mark as Out of Scope.\n\n"

                "### Response Format:\n"
                "- Classification: <Answerable / Needs More Information / Out of Scope>\n"
                "- Priority: <High / Medium / Low>\n"
                "- Answer:\n"
                "  <Your answer here>\n"
                "  [If partial:]\n"
                "  - Available Information: <what was found>\n"
                "  - Missing Information: <what is missing>\n"
                "  [If conflict:]\n"
                "  - Note: History and documents disagree. Document answer used.\n"
            ),
        ],
        required_variables=["documents", "query"],
    ),
)

    # Memory components 
    pipeline.add_component("message_retriever", ChatMessageRetriever(message_store))
    pipeline.add_component("message_writer", ChatMessageWriter(message_store))

    # After LLM replies, join prompt + replies so the full turn is persisted.
    pipeline.add_component(
        "message_joiner",
        OutputAdapter(
            template="{{ prompt + replies }}",
            output_type=list[ChatMessage],
            unsafe=True,
        ),
    )

    # LLM 
    pipeline.add_component(
        "llm",
        OpenAIChatGenerator(
            api_key=Secret.from_env_var("OPENAI_API_KEY"),
            model="gpt-5.2",
            generation_kwargs={"temperature": 0.0, "max_completion_tokens": 900},
        )
    )

    # RAG connections
    pipeline.connect("text_embedder.embedding", "embedding_retriever.query_embedding")
    pipeline.connect("embedding_retriever", "document_joiner")
    pipeline.connect("bm25_retriever", "document_joiner")
    pipeline.connect("document_joiner", "ranker.documents")
    pipeline.connect("ranker.documents", "prompt_builder.documents")

    # Memory connections
    # prompt_builder produces the current turn's messages (system + user).
    # message_retriever prepends stored history -> full message list -> llm.
    pipeline.connect("prompt_builder.prompt", "message_retriever.current_messages")
    pipeline.connect("message_retriever.messages", "llm.messages")

    # Persist the full turn (prompt + reply) to the store.
    pipeline.connect("prompt_builder.prompt", "message_joiner.prompt")
    pipeline.connect("llm.replies", "message_joiner.replies")
    pipeline.connect("message_joiner.output", "message_writer.messages")

    return pipeline


_pipeline = None


def get_chat_retrieval_pipeline(top_k: int = 3, retrieval_top_k: int = 30):
    global _pipeline
    if _pipeline is None:
        _pipeline = create_chat_retrieval_pipeline(top_k=top_k, retrieval_top_k=retrieval_top_k)
        _pipeline.warm_up()
    return _pipeline

