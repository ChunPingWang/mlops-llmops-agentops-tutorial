"""
Human-in-the-Loop Agent using LangGraph (V-17)

Demonstrates a graph with research -> human_review (interrupt) -> execute states,
using MemorySaver for checkpointing and Langfuse for tracing.
"""

import operator
import os
from typing import Annotated, Literal

from dotenv import load_dotenv
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from typing_extensions import TypedDict

from langfuse import get_client as _langfuse_client
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
# State
# ---------------------------------------------------------------------------


class HITLState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    task: str
    action_plan: str
    human_approved: bool
    execution_result: str


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


def research_node(state: HITLState) -> dict:
    """Call the LLM to propose an action plan for the given task."""
    task = state["task"]
    response = llm.invoke(
        [
            SystemMessage(
                content=(
                    "You are a research assistant. Given the following task, "
                    "propose a concise action plan with numbered steps. "
                    "Do NOT execute the plan -- only propose it."
                )
            ),
            HumanMessage(content=task),
        ]
    )
    plan = response.content
    print(f"[research] Proposed action plan:\n{plan}\n")
    return {
        "action_plan": plan,
        "messages": [response],
    }


def human_review_node(state: HITLState) -> dict:
    """Pause execution and wait for human approval via interrupt()."""
    plan = state["action_plan"]

    # This call pauses the graph and surfaces the payload to the caller.
    decision = interrupt(
        {
            "action_plan": plan,
            "prompt": "Do you approve this action plan? Respond with 'approve' or 'reject'.",
        }
    )

    approved = isinstance(decision, str) and decision.strip().lower() == "approve"
    status = "APPROVED" if approved else "REJECTED"
    print(f"[human_review] Human decision: {status}")
    return {"human_approved": approved}


def execute_node(state: HITLState) -> dict:
    """Execute the approved plan, or report rejection."""
    if not state.get("human_approved", False):
        result = "Execution skipped -- plan was rejected by human reviewer."
        print(f"[execute] {result}")
        return {"execution_result": result}

    plan = state["action_plan"]
    response = llm.invoke(
        [
            SystemMessage(
                content=(
                    "You are an execution agent. The following action plan has "
                    "been approved by a human reviewer. Summarize the actions "
                    "you would take to carry it out, as if you were executing them."
                )
            ),
            HumanMessage(content=plan),
        ]
    )
    result = response.content
    print(f"[execute] Execution result:\n{result}\n")
    return {
        "execution_result": result,
        "messages": [response],
    }


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


def route_after_review(state: HITLState) -> Literal["execute"]:
    """Always proceed to execute (the execute node handles rejection logic)."""
    return "execute"


# ---------------------------------------------------------------------------
# Build graph
# ---------------------------------------------------------------------------


def build_graph() -> StateGraph:
    checkpointer = MemorySaver()

    builder = StateGraph(HITLState)
    builder.add_node("research", research_node)
    builder.add_node("human_review", human_review_node)
    builder.add_node("execute", execute_node)

    builder.add_edge(START, "research")
    builder.add_edge("research", "human_review")
    builder.add_conditional_edges("human_review", route_after_review, ["execute"])
    builder.add_edge("execute", END)

    return builder.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    langfuse_handler = LangfuseCallbackHandler()
    graph = build_graph()

    task = "Research the top 3 Python web frameworks and recommend one for a new REST API project."
    thread_config = {
        "configurable": {"thread_id": "hitl-demo-1"},
        "callbacks": [langfuse_handler],
    }

    print("=" * 60)
    print("STEP 1: Start the agent -- it will pause at human_review")
    print("=" * 60)

    # First invocation: runs research -> pauses at human_review (interrupt)
    for step in graph.stream(
        {
            "messages": [HumanMessage(content=task)],
            "task": task,
            "action_plan": "",
            "human_approved": False,
            "execution_result": "",
        },
        config=thread_config,
        stream_mode="updates",
    ):
        # LangGraph 0.4+ can yield tuples (namespace, payload) when nested
        # subgraphs or interrupts are involved, not just plain dicts.
        items = step.items() if isinstance(step, dict) else [(("__interrupt__",), step)]
        for node_name, update in items:
            keys = list(update.keys()) if isinstance(update, dict) else type(update).__name__
            print(f"  [stream] node={node_name}, keys={keys}")
    print()

    # Inspect the paused state
    current_state = graph.get_state(thread_config)
    print(f"Current node(s) to run next: {current_state.next}")
    if current_state.tasks:
        for t in current_state.tasks:
            if hasattr(t, "interrupts") and t.interrupts:
                for intr in t.interrupts:
                    print(f"Interrupt payload: {intr.value}")
    print()

    print("=" * 60)
    print("STEP 2: Simulate human approval and resume execution")
    print("=" * 60)

    # Resume with human approval
    for step in graph.stream(
        Command(resume="approve"),
        config=thread_config,
        stream_mode="updates",
    ):
        # LangGraph 0.4+ can yield tuples (namespace, payload) when nested
        # subgraphs or interrupts are involved, not just plain dicts.
        items = step.items() if isinstance(step, dict) else [(("__interrupt__",), step)]
        for node_name, update in items:
            keys = list(update.keys()) if isinstance(update, dict) else type(update).__name__
            print(f"  [stream] node={node_name}, keys={keys}")
    print()

    # Final state
    final_state = graph.get_state(thread_config)
    print("=" * 60)
    print("FINAL STATE")
    print("=" * 60)
    print(f"  Task:            {final_state.values.get('task', '')}")
    print(f"  Human approved:  {final_state.values.get('human_approved', False)}")
    print(f"  Execution result (first 200 chars):")
    print(f"    {final_state.values.get('execution_result', '')[:200]}")

    _langfuse_client().flush()


if __name__ == "__main__":
    main()
