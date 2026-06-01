"""
Seed script for Qdrant vector database.

Creates the "rag_documents" collection in Qdrant, generates embeddings for a
set of demo documents via the local LLM embedding endpoint, upserts them, and
runs a test similarity query to verify everything works.

Environment variables (loaded from .env):
  - LOCAL_LLM_BASE_URL: Base URL for the embedding endpoint.
  - QDRANT_HOST / QDRANT_PORT: Qdrant connection details.
"""

import os
import uuid

from dotenv import load_dotenv

load_dotenv()

import httpx
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

COLLECTION_NAME = "rag_documents"
VECTOR_SIZE = 768  # Gemma embedding dimensionality
EMBEDDING_MODEL = "text-embedding"


def _get_env(name: str, default: str | None = None) -> str:
    value = os.getenv(name, default)
    if value is None:
        raise EnvironmentError(f"Missing required environment variable: {name}")
    return value


# ---------------------------------------------------------------------------
# Demo documents
# ---------------------------------------------------------------------------

DEMO_DOCUMENTS = [
    {
        "id": str(uuid.uuid5(uuid.NAMESPACE_DNS, "doc-1")),
        "text": (
            "Large Language Models (LLMs) are deep neural networks trained on "
            "massive text corpora. They can generate human-like text, answer "
            "questions, summarise documents, and write code. Popular examples "
            "include GPT-4, Claude, Gemma, and Llama."
        ),
        "metadata": {"source": "intro_llm.txt", "topic": "LLM basics"},
    },
    {
        "id": str(uuid.uuid5(uuid.NAMESPACE_DNS, "doc-2")),
        "text": (
            "Retrieval-Augmented Generation (RAG) combines a retrieval system "
            "with a generative model. By fetching relevant documents from a "
            "knowledge base before generation, RAG reduces hallucinations and "
            "keeps answers grounded in up-to-date facts."
        ),
        "metadata": {"source": "rag_overview.txt", "topic": "RAG"},
    },
    {
        "id": str(uuid.uuid5(uuid.NAMESPACE_DNS, "doc-3")),
        "text": (
            "Vector databases such as Qdrant, Pinecone, and Weaviate store data "
            "as high-dimensional embedding vectors. They support approximate "
            "nearest-neighbor (ANN) search, enabling sub-millisecond retrieval "
            "over millions of vectors."
        ),
        "metadata": {"source": "vector_db.txt", "topic": "vector databases"},
    },
    {
        "id": str(uuid.uuid5(uuid.NAMESPACE_DNS, "doc-4")),
        "text": (
            "LLMOps (Large Language Model Operations) encompasses the tools and "
            "practices for deploying, monitoring, and maintaining LLM-powered "
            "applications in production. Key components include prompt "
            "management, model gateways like LiteLLM, and observability "
            "platforms like Langfuse."
        ),
        "metadata": {"source": "llmops.txt", "topic": "LLMOps"},
    },
    {
        "id": str(uuid.uuid5(uuid.NAMESPACE_DNS, "doc-5")),
        "text": (
            "Embedding models convert text into dense numerical vectors that "
            "capture semantic meaning. These vectors can be compared using "
            "cosine similarity or dot product to find related passages. Common "
            "embedding dimensions are 384, 768, and 1536."
        ),
        "metadata": {"source": "embeddings.txt", "topic": "embeddings"},
    },
]


# ---------------------------------------------------------------------------
# Embedding helper
# ---------------------------------------------------------------------------

def get_embeddings(texts: list[str]) -> list[list[float]]:
    """Call the local LLM embedding endpoint to embed a list of texts."""
    base_url = _get_env("LOCAL_LLM_BASE_URL").rstrip("/")
    url = f"{base_url}/embeddings"

    payload = {
        "model": EMBEDDING_MODEL,
        "input": texts,
    }

    print(f"[info] Requesting embeddings from {url} for {len(texts)} text(s)...")
    response = httpx.post(url, json=payload, timeout=60.0)
    response.raise_for_status()

    data = response.json()
    embeddings = [item["embedding"] for item in data["data"]]

    # Validate dimension
    if embeddings and len(embeddings[0]) != VECTOR_SIZE:
        print(
            f"[warn] Expected vector size {VECTOR_SIZE}, got {len(embeddings[0])}. "
            f"Update VECTOR_SIZE if your model uses a different dimension."
        )

    return embeddings


# ---------------------------------------------------------------------------
# Qdrant operations
# ---------------------------------------------------------------------------

def create_collection(client: QdrantClient) -> None:
    """Create (or recreate) the Qdrant collection with cosine distance."""
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=VECTOR_SIZE,
            distance=Distance.COSINE,
        ),
    )
    print(f"[info] Collection '{COLLECTION_NAME}' created (vector_size={VECTOR_SIZE}, distance=cosine)")


def upsert_documents(client: QdrantClient, documents: list[dict], embeddings: list[list[float]]) -> None:
    """Upsert document embeddings into the Qdrant collection."""
    points = []
    for doc, embedding in zip(documents, embeddings):
        point = PointStruct(
            id=doc["id"],
            vector=embedding,
            payload={
                "text": doc["text"],
                **doc["metadata"],
            },
        )
        points.append(point)

    client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"[info] Upserted {len(points)} point(s) into '{COLLECTION_NAME}'")


def test_query(client: QdrantClient) -> None:
    """Run a test similarity search to verify retrieval works."""
    query_text = "How does RAG work?"
    print(f"\n[test] Query: \"{query_text}\"")

    query_embedding = get_embeddings([query_text])[0]

    results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_embedding,
        limit=3,
    )

    print(f"[test] Top {len(results)} results:")
    for i, hit in enumerate(results, 1):
        score = hit.score
        text_preview = hit.payload.get("text", "")[:100]
        source = hit.payload.get("source", "unknown")
        print(f"  {i}. [score={score:.4f}] ({source}) {text_preview}...")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Seed Qdrant with demo documents and verify retrieval."""
    print("=" * 60)
    print("Qdrant Seed Script")
    print("=" * 60)

    qdrant_host = _get_env("QDRANT_HOST", "localhost")
    qdrant_port = int(_get_env("QDRANT_PORT", "6333"))

    print(f"[info] Connecting to Qdrant at {qdrant_host}:{qdrant_port}")
    client = QdrantClient(host=qdrant_host, port=qdrant_port)

    # 1. Create collection
    create_collection(client)

    # 2. Generate embeddings for demo documents
    texts = [doc["text"] for doc in DEMO_DOCUMENTS]
    embeddings = get_embeddings(texts)
    print(f"[info] Generated {len(embeddings)} embedding(s), dimension={len(embeddings[0])}")

    # 3. Upsert into Qdrant
    upsert_documents(client, DEMO_DOCUMENTS, embeddings)

    # 4. Verify with a test query
    test_query(client)

    # 5. Show collection info
    info = client.get_collection(COLLECTION_NAME)
    print(f"\n[info] Collection '{COLLECTION_NAME}' status: {info.status}")
    print(f"[info] Points count: {info.points_count}")
    print(f"[info] Vectors count: {info.vectors_count}")

    print("\nDone.")


if __name__ == "__main__":
    main()
