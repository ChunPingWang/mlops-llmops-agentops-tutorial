"""
ReAct Agent using LangGraph (V-15, V-16)

A ReAct agent with calculator, web search, and time tools,
orchestrated via LangGraph with Langfuse tracing.
"""

import operator
import os
import math
import re
from datetime import datetime
from typing import Annotated, Literal

from dotenv import load_dotenv
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langgraph.graph import END, START, StateGraph
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
# Tools
# ---------------------------------------------------------------------------
SAFE_MATH_NAMES = {
    "abs": abs, "round": round, "min": min, "max": max,
    "pow": pow, "sum": sum, "int": int, "float": float,
    "sqrt": math.sqrt, "log": math.log, "log10": math.log10,
    "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "pi": math.pi, "e": math.e,
}


@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression safely.

    Args:
        expression: A math expression string, e.g. '2 + 3 * 4' or 'sqrt(144)'.
    """
    # Basic safety: reject anything that looks like attribute access or imports
    if re.search(r"__|import|exec|eval|open|os\.|sys\.", expression):
        return "Error: expression contains disallowed constructs."
    try:
        result = eval(expression, {"__builtins__": {}}, SAFE_MATH_NAMES)  # noqa: S307
        return str(result)
    except Exception as exc:
        return f"Error evaluating expression: {exc}"


@tool
def search_web(query: str) -> str:
    """Simulate a web search and return mock results (for demo purposes).

    Args:
        query: The search query string.
    """
    mock_results: dict[str, str] = {
        "current year": "The current year is 2026.",
        "square root": "The square root of 144 is 12.",
        "python": "Python is a popular programming language.",
    }
    query_lower = query.lower()
    for key, value in mock_results.items():
        if key in query_lower:
            return value
    return f"Mock search results for '{query}': No specific results found. Try a different query."


@tool
def get_current_time() -> str:
    """Return the current date and time as an ISO-formatted string."""
    return datetime.now().isoformat()


# ---------------------------------------------------------------------------
# Graph state & nodes
# ---------------------------------------------------------------------------
tools = [calculator, search_web, get_current_time]
tools_by_name: dict[str, tool] = {t.name: t for t in tools}
llm_with_tools = llm.bind_tools(tools)


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]


def agent_node(state: AgentState) -> dict:
    """Call the LLM with tool bindings."""
    system = SystemMessage(
        content=(
            "You are a helpful ReAct agent. Think step by step. "
            "Use the available tools to answer the user's question."
        )
    )
    response = llm_with_tools.invoke([system] + state["messages"])
    return {"messages": [response]}


def tool_node(state: AgentState) -> dict:
    """Execute any tool calls made by the LLM."""
    last_message = state["messages"][-1]
    results: list[ToolMessage] = []
    for tc in last_message.tool_calls:
        tool_fn = tools_by_name[tc["name"]]
        observation = tool_fn.invoke(tc["args"])
        results.append(ToolMessage(content=str(observation), tool_call_id=tc["id"]))
    return {"messages": results}


def should_continue(state: AgentState) -> Literal["tool_node", "__end__"]:
    """Route to tool_node if there are tool calls, otherwise end."""
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tool_node"
    return END


# ---------------------------------------------------------------------------
# Build graph
# ---------------------------------------------------------------------------
def build_graph() -> StateGraph:
    builder = StateGraph(AgentState)
    builder.add_node("agent", agent_node)
    builder.add_node("tool_node", tool_node)

    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", should_continue, ["tool_node", END])
    builder.add_edge("tool_node", "agent")

    return builder.compile()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    langfuse_handler = LangfuseCallbackHandler()
    graph = build_graph()

    question = "What is the square root of 144 plus the current year?"
    print(f"Question: {question}\n")

    config = {"callbacks": [langfuse_handler]}
    for step in graph.stream(
        {"messages": [HumanMessage(content=question)]},
        config=config,
        stream_mode="updates",
    ):
        for node_name, update in step.items():
            print(f"--- {node_name} ---")
            for msg in update.get("messages", []):
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        print(f"  Tool call: {tc['name']}({tc['args']})")
                elif hasattr(msg, "content"):
                    content = msg.content
                    if content:
                        print(f"  {content}")
        print()

    # Print final answer
    final_state = graph.invoke(
        {"messages": [HumanMessage(content=question)]},
        config=config,
    )
    final_answer = final_state["messages"][-1].content
    print(f"Final Answer: {final_answer}")

    # v3 LangchainCallbackHandler has no .flush(); use the global client.
    _langfuse_client().flush()


if __name__ == "__main__":
    main()
