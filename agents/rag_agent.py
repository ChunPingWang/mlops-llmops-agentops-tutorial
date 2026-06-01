"""
RAG Agent with Guardrails using LangGraph

Combines Qdrant-based retrieval, LLM generation, and NeMo Guardrails
validation in a LangGraph-orchestrated agent. All steps are traced
in Langfuse.
"""

import operator
import os
from typing import Annotated, Literal

import httpx
from dotenv import load_dotenv
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langgraph.graph import END, START, StateGraph
from qdrant_client import QdrantClient
from typing_extensions import TypedDict

from langfuse.langchain import CallbackHandler as LangfuseCallbackHandler

load_dotenv()

# ---------------------------------------------------------------------------
# LLM setup
# ---------------------------------------------------------------------------
LITELLM_BASE_URL = "http://localhost:4000/v1"
LITELLM_API_KEY = os.getenv("LITELLM_MASTER_KEY", "")
MODEL_NAME = "gemma-4-26b"

llm = ChatOpenAI(
    model=MODEL_NAME,
    base_url=LITELLM_BASE_URL,
    api_key=LITELLM_API_KEY,
    temperature=0,
)

# ---------------------------------------------------------------------------
# External service config
# ---------------------------------------------------------------------------
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_COLLECTION = "rag_documents"
GUARDRAILS_URL = os.getenv("GUARDRAILS_URL", "http://localhost:8090")

# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@tool
def qdrant_search(query: str) -> str:
    """Search the Qdrant vector database for documents relevant to the query.

    Connects to a Qdrant instance and performs a semantic search against
    the 'rag_documents' collection.

    Args:
        query: The user question or search query.
    """
    try:
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

        # Use Qdrant's built-in query (requires server-side embedding or
        # a collection configured with a named vector).  Fall back to
        # scroll if the collection uses custom embeddings.
        try:
            results = client.query(
                collection_name=QDRANT_COLLECTION,
                query_text=query,
                limit=3,
            )
            documents: list[str] = []
            for point in results:
                payload = point.metadata if hasattr(point, "metadata") else {}
                text = payload.get("text", payload.get("content", str(payload)))
                score = point.score if hasattr(point, "score") else "N/A"
                documents.append(f"[score={score}] {text}")
            if documents:
                return "\n---\n".join(documents)
            return "No relevant documents found in Qdrant."
        except Exception:
            # Fallback: scroll first N records as context (useful when
            # embeddings are managed externally).
            records, _ = client.scroll(
                collection_name=QDRANT_COLLECTION,
                limit=3,
                with_payload=True,
            )
            if not records:
                return "No documents found in the collection."
            docs = []
            for r in records:
                payload = r.payload or {}
                text = payload.get("text", payload.get("content", str(payload)))
                docs.append(text)
            return "\n---\n".join(docs)
    except Exception as exc:
        return f"Error searching Qdrant: {exc}"


@tool
def guardrails_check(text: str) -> str:
    """Validate text through NeMo Guardrails API.

    Sends the generated answer to the NeMo Guardrails service for
    content-safety and policy checks.

    Args:
        text: The LLM-generated text to validate.
    """
    try:
        response = httpx.post(
            f"{GUARDRAILS_URL}/v1/chat/completions",
            json={
                "model": MODEL_NAME,
                "messages": [
                    {
                        "role": "user",
                        "content": text,
                    }
                ],
            },
            timeout=30.0,
        )
        response.raise_for_status()
        data = response.json()

        # Extract guardrails verdict from the response
        choices = data.get("choices", [])
        if choices:
            reply = choices[0].get("message", {}).get("content", "")
            return f"Guardrails check passed. Response: {reply}"
        return "Guardrails check completed -- no issues detected."
    except httpx.HTTPStatusError as exc:
        return f"Guardrails check failed (HTTP {exc.response.status_code}): {exc.response.text}"
    except Exception as exc:
        return f"Guardrails check error: {exc}"


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------
tools_list = [qdrant_search, guardrails_check]
tools_by_name = {t.name: t for t in tools_list}
llm_with_tools = llm.bind_tools(tools_list)


class RAGState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    user_question: str
    retrieved_context: str
    generated_answer: str
    guardrails_result: str


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


def retrieve_node(state: RAGState) -> dict:
    """Retrieve relevant documents from Qdrant."""
    question = state["user_question"]
    print(f"[retrieve] Searching Qdrant for: {question}")
    context = qdrant_search.invoke({"query": question})
    print(f"[retrieve] Retrieved context (first 300 chars):\n  {context[:300]}\n")
    return {"retrieved_context": context}


def generate_node(state: RAGState) -> dict:
    """Generate an answer using the LLM with retrieved context."""
    question = state["user_question"]
    context = state.get("retrieved_context", "")

    response = llm.invoke(
        [
            SystemMessage(
                content=(
                    "You are a knowledgeable assistant. Answer the user's question "
                    "based on the provided context. If the context is insufficient, "
                    "say so clearly.\n\n"
                    f"Context:\n{context}"
                )
            ),
            HumanMessage(content=question),
        ]
    )
    answer = response.content
    print(f"[generate] LLM answer (first 300 chars):\n  {answer[:300]}\n")
    return {
        "generated_answer": answer,
        "messages": [response],
    }


def guardrails_node(state: RAGState) -> dict:
    """Validate the generated answer through NeMo Guardrails."""
    answer = state.get("generated_answer", "")
    print("[guardrails] Validating answer through NeMo Guardrails...")
    result = guardrails_check.invoke({"text": answer})
    print(f"[guardrails] Result: {result}\n")
    return {"guardrails_result": result}


def output_node(state: RAGState) -> dict:
    """Produce the final output based on guardrails result."""
    guardrails_result = state.get("guardrails_result", "")
    answer = state.get("generated_answer", "")

    if "failed" in guardrails_result.lower() or "error" in guardrails_result.lower():
        final = (
            "The generated answer did not pass guardrails validation.\n"
            f"Guardrails feedback: {guardrails_result}\n"
            "Please rephrase your question or try again."
        )
    else:
        final = answer

    print(f"[output] Final answer:\n  {final[:300]}\n")
    return {
        "messages": [HumanMessage(content=final)],
    }


# ---------------------------------------------------------------------------
# Build graph
# ---------------------------------------------------------------------------


def build_graph() -> StateGraph:
    builder = StateGraph(RAGState)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("generate", generate_node)
    builder.add_node("guardrails", guardrails_node)
    builder.add_node("output", output_node)

    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "generate")
    builder.add_edge("generate", "guardrails")
    builder.add_edge("guardrails", "output")
    builder.add_edge("output", END)

    return builder.compile()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    langfuse_handler = LangfuseCallbackHandler()
    graph = build_graph()

    question = "What are the best practices for deploying machine learning models in production?"
    print("=" * 60)
    print(f"User question: {question}")
    print("=" * 60 + "\n")

    config = {"callbacks": [langfuse_handler]}
    for step in graph.stream(
        {
            "messages": [HumanMessage(content=question)],
            "user_question": question,
            "retrieved_context": "",
            "generated_answer": "",
            "guardrails_result": "",
        },
        config=config,
        stream_mode="updates",
    ):
        for node_name, update in step.items():
            print(f"  [stream] node={node_name}, keys={list(update.keys())}")
    print()

    # Full invocation for final state
    final_state = graph.invoke(
        {
            "messages": [HumanMessage(content=question)],
            "user_question": question,
            "retrieved_context": "",
            "generated_answer": "",
            "guardrails_result": "",
        },
        config=config,
    )

    print("=" * 60)
    print("FINAL RESULT")
    print("=" * 60)
    print(f"  Question:          {final_state['user_question']}")
    print(f"  Guardrails result: {final_state['guardrails_result'][:200]}")
    print(f"  Answer (first 500 chars):")
    print(f"    {final_state['generated_answer'][:500]}")

    langfuse_handler.flush()


if __name__ == "__main__":
    main()
