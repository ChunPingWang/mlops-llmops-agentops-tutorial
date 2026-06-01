"""
Ragas evaluation script for the RAG pipeline.

Evaluates retrieval-augmented generation quality using the Ragas framework with
a small synthetic test dataset. Metrics computed:
  - faithfulness
  - answer_relevancy
  - context_precision
  - context_recall

The LLM backend is the LiteLLM proxy at localhost:4000 (OpenAI-compatible).
Results are printed as a pandas DataFrame and optionally logged to Langfuse.

Environment variables (loaded from .env):
  - LITELLM_MASTER_KEY: API key for the LiteLLM proxy.
  - LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_HOST: optional Langfuse.
"""

import os

from dotenv import load_dotenv

load_dotenv()

import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


# ---------------------------------------------------------------------------
# Test dataset
# ---------------------------------------------------------------------------

EVAL_DATA: dict[str, list] = {
    "question": [
        "What is retrieval-augmented generation?",
        "How does a vector database store documents?",
        "What are the benefits of using LLM observability tools?",
        "Explain the concept of embedding similarity search.",
        "What is prompt engineering?",
    ],
    "answer": [
        (
            "Retrieval-augmented generation (RAG) is a technique that combines "
            "a retrieval system with a generative language model. It first "
            "retrieves relevant documents from a knowledge base, then feeds "
            "them as context to the LLM to produce grounded answers."
        ),
        (
            "A vector database stores documents as high-dimensional embedding "
            "vectors. When a query arrives, it is also converted to a vector "
            "and the database performs approximate nearest-neighbor search to "
            "find the most similar documents."
        ),
        (
            "LLM observability tools help monitor model performance, track "
            "token usage and latency, detect hallucinations, and provide "
            "tracing for debugging complex chains and agent workflows."
        ),
        (
            "Embedding similarity search converts text into dense numerical "
            "vectors using an embedding model, then measures how close two "
            "vectors are—typically via cosine similarity—to find semantically "
            "similar content."
        ),
        (
            "Prompt engineering is the practice of designing and refining "
            "input prompts to guide a language model toward producing desired "
            "outputs, including techniques such as few-shot examples, chain-of-"
            "thought reasoning, and system instructions."
        ),
    ],
    "contexts": [
        [
            "RAG (Retrieval-Augmented Generation) enhances LLM responses by "
            "retrieving relevant documents from external knowledge bases before "
            "generating an answer. This grounds the model's output in factual data.",
            "The retrieval step typically uses a vector database to find the "
            "most relevant chunks of text for a given query.",
        ],
        [
            "Vector databases like Qdrant, Pinecone, and Weaviate store data "
            "as embedding vectors. Documents are first converted into vectors "
            "using an embedding model, then indexed for fast similarity search.",
            "Approximate nearest-neighbor (ANN) algorithms allow vector "
            "databases to scale to millions of vectors while maintaining low "
            "query latency.",
        ],
        [
            "Observability platforms such as Langfuse and LangSmith provide "
            "tracing, logging, and analytics for LLM applications. They help "
            "developers monitor token costs, latency, and output quality.",
            "By tracing each step of a chain or agent, developers can quickly "
            "identify bottlenecks and debug failures in production systems.",
        ],
        [
            "Embedding models map text to dense vectors in a high-dimensional "
            "space. Cosine similarity between vectors measures semantic "
            "closeness, enabling efficient similarity search over large corpora.",
            "Modern embedding models like text-embedding-ada-002 and sentence "
            "transformers produce 768- or 1536-dimensional vectors.",
        ],
        [
            "Prompt engineering involves crafting inputs that steer LLMs toward "
            "accurate, relevant, and well-structured responses. Techniques "
            "include zero-shot, few-shot, and chain-of-thought prompting.",
            "System prompts set the overall behavior of a model, while user "
            "prompts contain the specific question or task.",
        ],
    ],
    "ground_truth": [
        (
            "RAG combines information retrieval with text generation. A "
            "retriever fetches relevant documents and the LLM uses them as "
            "context to generate factually grounded answers."
        ),
        (
            "Vector databases convert documents into embedding vectors and "
            "use approximate nearest-neighbor search to quickly retrieve the "
            "most similar vectors for a given query."
        ),
        (
            "LLM observability tools provide tracing, cost monitoring, latency "
            "tracking, and quality analytics to help developers maintain and "
            "debug production LLM applications."
        ),
        (
            "Embedding similarity search uses dense vector representations of "
            "text and measures their cosine similarity to identify semantically "
            "related documents."
        ),
        (
            "Prompt engineering is the design of model inputs—using techniques "
            "like few-shot examples and chain-of-thought—to control LLM "
            "behavior and improve output quality."
        ),
    ],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_env(name: str, default: str | None = None) -> str:
    value = os.getenv(name, default)
    if value is None:
        raise EnvironmentError(f"Missing required environment variable: {name}")
    return value


def get_eval_llm() -> ChatOpenAI:
    """LLM used by Ragas for evaluation (via LiteLLM proxy)."""
    return ChatOpenAI(
        model="gemma-4-26b",
        temperature=0.0,
        openai_api_base="http://localhost:4000/v1",
        openai_api_key=_get_env("LITELLM_MASTER_KEY"),
    )


def get_eval_embeddings() -> OpenAIEmbeddings:
    """Embedding model used by Ragas (via local LLM endpoint)."""
    base_url = _get_env("LOCAL_LLM_BASE_URL")
    return OpenAIEmbeddings(
        model="text-embedding",
        openai_api_base=base_url,
        openai_api_key="no-key-needed",
    )


def log_results_to_langfuse(results_df: pd.DataFrame) -> None:
    """Log evaluation results to Langfuse as a score/trace if env vars are set."""
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    host = os.getenv("LANGFUSE_HOST")

    if not all([public_key, secret_key, host]):
        print("[info] Langfuse env vars not set — skipping result logging.")
        return

    try:
        from langfuse import Langfuse

        langfuse = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=host,
        )

        trace = langfuse.trace(name="ragas-evaluation")

        # Log each metric as a score on the trace
        for column in results_df.columns:
            if column in ("question", "answer", "contexts", "ground_truth"):
                continue
            mean_value = results_df[column].mean()
            trace.score(name=column, value=float(mean_value))
            print(f"  [langfuse] Logged score {column} = {mean_value:.4f}")

        langfuse.flush()
        print("[info] Evaluation results logged to Langfuse.")

    except Exception as exc:
        print(f"[warn] Failed to log to Langfuse: {exc}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Run Ragas evaluation on the test dataset and print results."""
    print("=" * 60)
    print("Ragas RAG Evaluation")
    print("=" * 60)

    # Build HuggingFace Dataset from test data
    dataset = Dataset.from_dict(EVAL_DATA)
    print(f"[info] Test dataset: {len(dataset)} sample(s)")

    # Prepare LLM and embeddings wrappers for Ragas
    eval_llm = LangchainLLMWrapper(get_eval_llm())
    eval_embeddings = LangchainEmbeddingsWrapper(get_eval_embeddings())

    # Run evaluation
    metrics = [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    ]

    print("[info] Running Ragas evaluation (this may take a few minutes)...")
    results = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=eval_llm,
        embeddings=eval_embeddings,
    )

    # Convert to DataFrame and display
    results_df = results.to_pandas()
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(results_df.to_string(index=False))

    # Print summary
    print("\n--- Metric Averages ---")
    metric_names = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    for metric in metric_names:
        if metric in results_df.columns:
            print(f"  {metric:25s}: {results_df[metric].mean():.4f}")

    # Optionally log to Langfuse
    log_results_to_langfuse(results_df)

    print("\nDone.")
    return results_df


if __name__ == "__main__":
    main()
