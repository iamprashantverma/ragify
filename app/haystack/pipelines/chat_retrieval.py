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

    # Prompt builder — uses retrieved docs + current query only
    # History is injected by message_retriever via current_messages
    pipeline.add_component(
    "prompt_builder",
    ChatPromptBuilder(
        template=[
            ChatMessage.from_system(
                "You are a triage assistant. Your job is to analyze the user query using the provided context and conversation history to decide the appropriate outcome.\n"
                "Instructions:\n"
                "1. Carefully analyze the query, the provided context documents, AND the conversation history together.\n"
                "2. Before classifying as 'Needs More Information', check if the answer can be inferred by combining:\n"
                "   - Context documents\n"
                "   - Conversation history (e.g., resolve pronouns like 'their', 'he', 'this case' from prior turns)\n"
                "3. Determine whether:\n"
                "   - The query can be fully answered\n"
                "   - Partially answered\n"
                "   - Not answered at all\n"
                "4. If FULLY answerable:\n"
                "   - Provide a clear and concise answer based on context and/or conversation history.\n"
                "5. If PARTIALLY answerable:\n"
                "   - Provide the available answer from the context.\n"
                "   - Clearly mention which parts of the query are NOT present in the context or history.\n"
                "   - Do NOT hallucinate or assume missing details.\n"
                "6. If NOT answerable:\n"
                "   - Respond with: 'I don't have enough information to answer this question.'\n"
                "   - Also briefly mention what information is missing.\n"
                "7. Classification:\n"
                "   - 'Answerable' → Fully supported by context and/or conversation history\n"
                "   - 'Needs More Information' → Partially or not supported even after checking history\n"
                "   - 'Out of Scope' → Query unrelated to context and history\n"
                "8. Priority:\n"
                "   - 'High' → Legal, financial, or critical decision-making queries\n"
                "   - 'Medium' → Informational but important\n"
                "   - 'Low' → General or simple queries\n"
                "Important Rules:\n"
                "- Use ONLY the provided context documents and conversation history.\n"
                "- Do NOT add external knowledge.\n"
                "- Always try to resolve follow-up references (e.g., 'their', 'he', 'this case', 'which witness') "
                "using the conversation history before deciding the query is unanswerable.\n"
                "- Be explicit about missing information when applicable.\n"
                "- Prioritize context documents over conversation history.\n"
                "- If the query is a greeting, reaction (e.g., 'nice', 'ok', 'thanks'), or non-question, "
                "classify it as 'Out of Scope' with Low priority."
            ),
            ChatMessage.from_user(
                "Context Documents:\n"
                "{% for doc in documents %}"
                "Document {{ loop.index }}:\n{{ doc.content }}\n\n"
                "{% endfor %}"
                "{% if history %}"
                "\nConversation History:\n"
                "{% for message in history %}"
                "{{ message.role | capitalize }}: {{ message.content }}\n"
                "{% endfor %}"
                "\n{% endif %}"
                "\nQuery: {{ query }}\n\n"
                "Output format:\n"
                "- Classification: <Answerable / Needs More Information / Out of Scope>\n"
                "- Priority: <High / Medium / Low>\n"
                "- Answer:\n"
                "  <Provide answer here. If partial, clearly include:\n"
                "   - Available Information:\n"
                "   - Missing Information:>"
            ),
        ],
        required_variables=["documents", "query"],
    ),
)

    # prompt -> message_retriever (current_messages) -> combines with stored history -> llm
    pipeline.add_component("message_retriever", ChatMessageRetriever(message_store))
    pipeline.add_component("message_writer", ChatMessageWriter(message_store))

    # After LLM replies, join prompt + replies and write to store
    pipeline.add_component(
        "message_joiner",
        OutputAdapter(
            template="{{ prompt + replies }}",
            output_type=list[ChatMessage],
            unsafe=True,
        ),
    )

    pipeline.add_component(
        "llm",
        OpenAIChatGenerator(
            api_key=Secret.from_env_var("OPENAI_API_KEY"),
            model="gpt-4o",
            generation_kwargs={"temperature": 0.0, "max_tokens": 600},
        ),
    )

    # RAG connections
    pipeline.connect("text_embedder.embedding", "embedding_retriever.query_embedding")
    pipeline.connect("embedding_retriever", "document_joiner")
    pipeline.connect("bm25_retriever", "document_joiner")
    pipeline.connect("document_joiner", "ranker.documents")
    pipeline.connect("ranker.documents", "prompt_builder.documents")


    # message_retriever.messages -> llm (full history + current message)
    pipeline.connect("prompt_builder.prompt", "message_retriever.current_messages")
    pipeline.connect("message_retriever.messages", "llm.messages")

    #join prompt + llm reply
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