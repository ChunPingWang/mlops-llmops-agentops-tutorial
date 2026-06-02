"""
Mem0 Memory Management Agent (V-19)

Demonstrates cross-conversation agent memory using Mem0 in self-hosted mode
with Qdrant as the vector store and an OpenAI-compatible LLM via LiteLLM.
All LLM calls are traced through Langfuse.
"""

import os
import json
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Langfuse v3 tracing setup
# ---------------------------------------------------------------------------
# v3 replaced trace()/span()/score() with start_as_current_span/observation
# context managers. The Langfuse() constructor still reads credentials from
# LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_HOST env vars.
from langfuse import Langfuse

langfuse = Langfuse()

# ---------------------------------------------------------------------------
# Mem0 configuration (self-hosted backends)
# ---------------------------------------------------------------------------
from mem0 import Memory

config = {
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "host": os.getenv("QDRANT_HOST", "localhost"),
            "port": int(os.getenv("QDRANT_PORT", 6333)),
            "collection_name": "mem0_memories",
        },
    },
    "llm": {
        "provider": "openai",
        "config": {
            "model": "gemma-4-26b",
            "api_key": os.getenv("LITELLM_MASTER_KEY"),
            "openai_base_url": "http://localhost:4000/v1",
        },
    },
    "embedder": {
        "provider": "openai",
        # Embedding model is a separate identifier — use the embedding model
        # exposed by LiteLLM/OMLX, not the chat model.
        "config": {
            "model": "text-embedding",
            "api_key": os.getenv("LITELLM_MASTER_KEY"),
            "openai_base_url": "http://localhost:4000/v1",
        },
    },
}


def pretty_print(label: str, data) -> None:
    """Print a labelled result in a readable format."""
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")
    if isinstance(data, (dict, list)):
        print(json.dumps(data, indent=2, default=str))
    else:
        print(data)
    print()


def main() -> None:
    # In v3, the parent trace is created implicitly by the outer
    # start_as_current_span() — all nested spans attach to it.
    with langfuse.start_as_current_span(
        name="mem0-memory-demo",
        metadata={"version": "V-19"},
    ) as root_span:
        root_span.update_trace(tags=["mem0", "memory", "demo"])

        # --------------------------------------------------------------
        # Initialise Mem0 Memory client
        # --------------------------------------------------------------
        with langfuse.start_as_current_span(name="mem0-init") as span_init:
            print("Initialising Mem0 Memory client ...")
            try:
                m = Memory.from_config(config)
                span_init.update(output={"status": "ok"})
                print("Mem0 client ready.")
            except Exception as exc:
                span_init.update(output={"status": "error", "error": str(exc)})
                print(f"Failed to initialise Mem0: {exc}")
                raise

        user_id = "alice"

        # --------------------------------------------------------------
        # 1. Conversation 1 -- add memories
        # --------------------------------------------------------------
        conv1_message = (
            "My name is Alice and I work at TechCorp as a senior engineer"
        )
        pretty_print("Conversation 1 -- User says", conv1_message)

        with langfuse.start_as_current_span(
            name="mem0-add", input={"message": conv1_message}
        ) as span_add:
            try:
                add_result = m.add(
                    messages=[{"role": "user", "content": conv1_message}],
                    user_id=user_id,
                )
                span_add.update(output=add_result)
                pretty_print("mem0.add() result", add_result)
            except Exception as exc:
                span_add.update(output={"error": str(exc)})
                print(f"Error adding memory: {exc}")
                raise

        # --------------------------------------------------------------
        # 2. Conversation 2 -- search memories (cross-conversation recall)
        # --------------------------------------------------------------
        conv2_query = "What's my name and where do I work?"
        pretty_print("Conversation 2 -- User asks", conv2_query)

        with langfuse.start_as_current_span(
            name="mem0-search", input={"query": conv2_query}
        ) as span_search:
            try:
                search_results = m.search(
                    query=conv2_query,
                    user_id=user_id,
                )
                span_search.update(output=search_results)
                pretty_print("mem0.search() result", search_results)

                if search_results:
                    print("The agent can recall from Conversation 1:")
                    for entry in (
                        search_results
                        if isinstance(search_results, list)
                        else search_results.get("results", [])
                    ):
                        memory_text = (
                            entry.get("memory", entry.get("text", ""))
                            if isinstance(entry, dict)
                            else str(entry)
                        )
                        print(f"  - {memory_text}")
                else:
                    print("No memories found for the query.")
            except Exception as exc:
                span_search.update(output={"error": str(exc)})
                print(f"Error searching memory: {exc}")
                raise

        # --------------------------------------------------------------
        # 3. get_all -- list every memory for the user
        # --------------------------------------------------------------
        with langfuse.start_as_current_span(name="mem0-get-all") as span_getall:
            try:
                all_memories = m.get_all(user_id=user_id)
                span_getall.update(output=all_memories)
                pretty_print("mem0.get_all() -- all memories for alice", all_memories)
            except Exception as exc:
                span_getall.update(output={"error": str(exc)})
                print(f"Error listing memories: {exc}")
                raise

        # --------------------------------------------------------------
        # 4. delete -- remove a specific memory
        # --------------------------------------------------------------
        memories_list = (
            all_memories
            if isinstance(all_memories, list)
            else all_memories.get("results", [])
        )

        if memories_list:
            target_id = memories_list[0].get("id", memories_list[0].get("memory_id"))
            pretty_print("Deleting memory with id", target_id)

            with langfuse.start_as_current_span(
                name="mem0-delete", input={"memory_id": target_id}
            ) as span_delete:
                try:
                    delete_result = m.delete(memory_id=target_id)
                    span_delete.update(output={"deleted": target_id})
                    pretty_print("mem0.delete() result", delete_result)
                except Exception as exc:
                    span_delete.update(output={"error": str(exc)})
                    print(f"Error deleting memory: {exc}")
                    raise

            remaining = m.get_all(user_id=user_id)
            pretty_print("Remaining memories after deletion", remaining)
        else:
            print("No memories to delete.")

    # Outer with-block exited; flush before process exit.
    langfuse.flush()
    print("Done. Langfuse events flushed.")


if __name__ == "__main__":
    main()
