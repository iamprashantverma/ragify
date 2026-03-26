from haystack import Pipeline
from haystack.components.builders import PromptBuilder
from haystack.components.generators import OpenAIGenerator
from haystack.components.joiners import DocumentJoiner
from haystack.utils import Secret

from app.haystack.retrievers.dense import get_dense_retriever
from app.haystack.retrievers.bm25 import get_bm25_retriever
from app.haystack.rankers.cross_encoder import get_ranker
from app.haystack.processors.embedding import get_text_embedder

prompt_template = """
You are a triage assistant. Your job is to analyze the user query using the provided context and decide the appropriate outcome.

Instructions:
1. Carefully analyze the query and the provided context.
2. Determine whether:
   - The query can be fully answered
   - Partially answered
   - Not answered at all

3. If FULLY answerable:
   - Provide a clear and concise answer based only on the context.

4. If PARTIALLY answerable:
   - Provide the available answer from the context.
   - Clearly mention which parts of the query are NOT present in the context.
   - Do NOT hallucinate or assume missing details.

5. If NOT answerable:
   - Respond with: "I don't have enough information to answer this question."
   - Also briefly mention what information is missing.

6. Classification:
   - "Answerable" → Fully supported by context
   - "Needs More Information" → Partially or not supported
   - "Out of Scope" → Query unrelated to context

7. Priority:
   - "High" → Legal, financial, or critical decision-making queries
   - "Medium" → Informational but important
   - "Low" → General or simple queries

Important Rules:
- Use ONLY the provided context.
- Do NOT add external knowledge.
- Be explicit about missing information when applicable.

Context:
{% for doc in documents %}
Document {{ loop.index }}:
{{ doc.content }}

{% endfor %}

Query: {{ query }}

Output format:
- Classification: <Answerable / Needs More Information / Out of Scope>
- Priority: <High / Medium / Low>
- Answer:
  <Provide answer here. If partial, clearly include:
   - Available Information:
   - Missing Information:>
"""

def create_hybrid_pipeline(top_k: int = 3, retrieval_top_k: int = 30):
   hybrid_retrieval = Pipeline()

   hybrid_retrieval.add_component("text_embedder", get_text_embedder())
   hybrid_retrieval.add_component("embedding_retriever", get_dense_retriever(top_k=retrieval_top_k))
   hybrid_retrieval.add_component("bm25_retriever", get_bm25_retriever(top_k=retrieval_top_k))
   hybrid_retrieval.add_component("document_joiner", DocumentJoiner())
   hybrid_retrieval.add_component("ranker", get_ranker(top_k=top_k))
   hybrid_retrieval.add_component("prompt_builder", PromptBuilder(template=prompt_template, required_variables=["documents", "query"]))
   hybrid_retrieval.add_component("llm", OpenAIGenerator(
      api_key=Secret.from_env_var("OPENAI_API_KEY"),
      model="gpt-5.4",
      generation_kwargs={"temperature": 0.0, "max_completion_tokens": 1000}
    ))
   
   hybrid_retrieval.connect("text_embedder.embedding", "embedding_retriever.query_embedding")
   hybrid_retrieval.connect("embedding_retriever", "document_joiner")
   hybrid_retrieval.connect("bm25_retriever", "document_joiner")
   hybrid_retrieval.connect("document_joiner", "ranker.documents")
   hybrid_retrieval.connect("ranker.documents", "prompt_builder.documents")
   hybrid_retrieval.connect("prompt_builder", "llm")
   
   return hybrid_retrieval


_pipeline = None

def get_hybrid_pipeline(top_k: int = 3, retrieval_top_k: int = 30):
   global _pipeline
   if _pipeline is None:
       _pipeline = create_hybrid_pipeline(top_k=top_k, retrieval_top_k=retrieval_top_k)
       _pipeline.show()
       _pipeline.warm_up()
   return _pipeline






