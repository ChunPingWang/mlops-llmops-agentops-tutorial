"""
RAG Pipeline using LangChain, Qdrant, and LiteLLM.

This module implements an end-to-end Retrieval-Augmented Generation pipeline:
  1. Load documents from a local ./data/ directory (PDF and text files).
  2. Split documents into chunks with RecursiveCharacterTextSplitter.
  3. Embed chunks via an OpenAI-compatible embedding endpoint (served by LiteLLM).
  4. Store embeddings in Qdrant vector database.
  5. Answer queries through a retrieval chain with Langfuse tracing.

Environment variables (loaded from .env):
  - LOCAL_LLM_BASE_URL: Base URL for the local embedding endpoint.
  - QDRANT_HOST / QDRANT_PORT: Qdrant connection details.
  - LITELLM_MASTER_KEY: API key for the LiteLLM proxy.
  - LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_HOST: Langfuse tracing.
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.document_loaders import PyPDFLoader
# langchain >=0.3 split text splitters out to a standalone package
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_qdrant import QdrantVectorStore
# RetrievalQA was removed from langchain ≥0.4; rebuild with LCEL primitives.
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langfuse.langchain import CallbackHandler as LangfuseCallbackHandler


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------

def _get_env(name: str, default: str | None = None) -> str:
    """Return an environment variable or raise if missing and no default."""
    value = os.getenv(name, default)
    if value is None:
        raise EnvironmentError(f"Missing required environment variable: {name}")
    return value


def get_langfuse_handler() -> LangfuseCallbackHandler | None:
    """Create a Langfuse v3 callback handler if credentials are available.

    v3's CallbackHandler reads credentials from env vars automatically
    (LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_HOST); the v2-style
    constructor kwargs were removed.
    """
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    host = os.getenv("LANGFUSE_HOST")

    if not all([public_key, secret_key, host]):
        print("[warn] Langfuse env vars not fully set — tracing disabled.")
        return None

    return LangfuseCallbackHandler()


# ---------------------------------------------------------------------------
# Embeddings & LLM
# ---------------------------------------------------------------------------

def get_embeddings(
    model: str = "text-embedding",
    base_url: str | None = None,
) -> OpenAIEmbeddings:
    """Return an OpenAI-compatible embedding model pointing at the local LLM endpoint."""
    base_url = base_url or _get_env("LOCAL_LLM_BASE_URL")
    return OpenAIEmbeddings(
        model=model,
        base_url=base_url,
        api_key="no-key-needed",  # local endpoint; key not required
        check_embedding_ctx_length=False,
    )


def get_llm(
    model: str = "gemma-4-26b",
    temperature: float = 0.2,
) -> ChatOpenAI:
    """Return a ChatOpenAI instance routed through the LiteLLM proxy."""
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        base_url="http://localhost:4000/v1",
        api_key=_get_env("LITELLM_MASTER_KEY"),
    )


# ---------------------------------------------------------------------------
# Document ingestion
# ---------------------------------------------------------------------------

def load_documents(data_dir: str = "./data") -> list:
    """Load PDF and text files from *data_dir*."""
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_path.resolve()}")

    docs = []

    # Load text files
    txt_loader = DirectoryLoader(
        str(data_path),
        glob="**/*.txt",
        loader_cls=TextLoader,
        show_progress=True,
    )
    docs.extend(txt_loader.load())

    # Load PDF files
    pdf_loader = DirectoryLoader(
        str(data_path),
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True,
    )
    docs.extend(pdf_loader.load())

    print(f"[info] Loaded {len(docs)} document(s) from {data_path.resolve()}")
    return docs


def split_documents(docs: list, chunk_size: int = 1000, chunk_overlap: int = 200) -> list:
    """Split documents into chunks using RecursiveCharacterTextSplitter."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(docs)
    print(f"[info] Split into {len(chunks)} chunk(s)")
    return chunks


# ---------------------------------------------------------------------------
# Vector store
# ---------------------------------------------------------------------------

def build_vectorstore(
    chunks: list,
    collection_name: str = "rag_documents",
    embeddings: OpenAIEmbeddings | None = None,
) -> QdrantVectorStore:
    """Embed *chunks* and upsert them into a Qdrant collection."""
    embeddings = embeddings or get_embeddings()

    qdrant_host = _get_env("QDRANT_HOST", "localhost")
    qdrant_port = int(_get_env("QDRANT_PORT", "6333"))
    url = f"http://{qdrant_host}:{qdrant_port}"

    vectorstore = QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        url=url,
        collection_name=collection_name,
        force_recreate=True,
    )
    print(
        f"[info] Upserted {len(chunks)} chunk(s) into Qdrant "
        f"collection '{collection_name}' at {url}"
    )
    return vectorstore


def get_vectorstore(
    collection_name: str = "rag_documents",
    embeddings: OpenAIEmbeddings | None = None,
) -> QdrantVectorStore:
    """Connect to an existing Qdrant collection (no ingestion)."""
    from qdrant_client import QdrantClient

    embeddings = embeddings or get_embeddings()

    qdrant_host = _get_env("QDRANT_HOST", "localhost")
    qdrant_port = int(_get_env("QDRANT_PORT", "6333"))

    client = QdrantClient(host=qdrant_host, port=qdrant_port)

    return QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
    )


# ---------------------------------------------------------------------------
# Retrieval chain
# ---------------------------------------------------------------------------

RAG_PROMPT_TEMPLATE = """\
Use the following context to answer the question. If you cannot find the
answer in the context, say "I don't have enough information to answer that."

Context:
{context}

Question: {question}

Answer:"""


def build_retrieval_chain(
    vectorstore: QdrantVectorStore,
    llm: ChatOpenAI | None = None,
    search_k: int = 4,
):
    """Build a retrieval chain using LCEL (replaces deprecated RetrievalQA).

    Returns an object exposing .invoke({"query": ...}) that produces a dict
    with `result` and `source_documents`, matching the old RetrievalQA API
    enough for the demo below.
    """
    llm = llm or get_llm()

    prompt = PromptTemplate(
        template=RAG_PROMPT_TEMPLATE,
        input_variables=["context", "question"],
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": search_k})

    def _format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    # LCEL pipeline: retrieve → format → prompt → llm → parse string
    answer_chain = (
        {
            "context": (lambda x: x["query"]) | retriever | _format_docs,
            "question": (lambda x: x["query"]),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    class _ChainWrapper:
        """Thin wrapper to keep the .invoke({"query": ...}) → {"result", "source_documents"} shape."""

        def invoke(self, payload, config=None):
            question = payload["query"]
            docs = retriever.invoke(question, config=config)
            answer = answer_chain.invoke({"query": question}, config=config)
            return {"result": answer, "source_documents": docs, "query": question}

    return _ChainWrapper()


def ask(chain, question: str, langfuse_handler=None) -> dict:
    """Run a question through the retrieval chain with optional Langfuse tracing."""
    callbacks = [langfuse_handler] if langfuse_handler else []
    result = chain.invoke({"query": question}, config={"callbacks": callbacks})
    return result


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------

def main():
    """Demo: ingest documents from ./data, then ask a sample question."""
    print("=" * 60)
    print("RAG Pipeline Demo")
    print("=" * 60)

    # 1. Ingest documents
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    try:
        docs = load_documents(data_dir)
    except FileNotFoundError:
        print(f"[error] Create a '{data_dir}' directory with .txt or .pdf files first.")
        sys.exit(1)

    if not docs:
        print("[error] No documents found in data directory.")
        sys.exit(1)

    chunks = split_documents(docs)

    # 2. Build vector store
    embeddings = get_embeddings()
    vectorstore = build_vectorstore(chunks, embeddings=embeddings)

    # 3. Build retrieval chain
    chain = build_retrieval_chain(vectorstore)

    # 4. Setup Langfuse tracing
    langfuse_handler = get_langfuse_handler()

    # 5. Ask a sample question
    sample_question = "What are the main topics covered in the documents?"
    print(f"\n[query] {sample_question}")
    result = ask(chain, sample_question, langfuse_handler=langfuse_handler)

    print(f"\n[answer] {result['result']}")
    print(f"\n[sources] Retrieved {len(result.get('source_documents', []))} source chunk(s)")
    for i, doc in enumerate(result.get("source_documents", []), 1):
        source = doc.metadata.get("source", "unknown")
        print(f"  {i}. {source}  (first 120 chars: {doc.page_content[:120]}...)")

    # Flush Langfuse events
    if langfuse_handler:
        # v3 LangchainCallbackHandler has no .flush(); use the global client.
        from langfuse import get_client
        get_client().flush()
        print("\n[info] Langfuse traces flushed.")

    print("\nDone.")


if __name__ == "__main__":
    main()
