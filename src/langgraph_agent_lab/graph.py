"""Graph construction."""

from __future__ import annotations

from typing import Any

from .state import AgentState


def build_graph(checkpointer: Any | None = None) -> Any:
    """Build and compile the LangGraph workflow."""

    from langgraph.graph import END, START, StateGraph

    from .nodes import (
        answer_node,
        approval_node,
        ask_clarification_node,
        classify_node,
        dead_letter_node,
        evaluate_node,
        finalize_node,
        intake_node,
        retry_or_fallback_node,
        risky_action_node,
        tool_node,
    )
    from .routing import (
        route_after_approval,
        route_after_classify,
        route_after_evaluate,
        route_after_retry,
    )

    graph = StateGraph(AgentState)
    graph.add_node("intake", intake_node)
    graph.add_node("classify", classify_node)
    graph.add_node("tool", tool_node)
    graph.add_node("evaluate", evaluate_node)
    graph.add_node("answer", answer_node)
    graph.add_node("clarify", ask_clarification_node)
    graph.add_node("risky_action", risky_action_node)
    graph.add_node("approval", approval_node)
    graph.add_node("retry", retry_or_fallback_node)
    graph.add_node("dead_letter", dead_letter_node)
    graph.add_node("finalize", finalize_node)

    graph.add_edge(START, "intake")
    graph.add_edge("intake", "classify")
    graph.add_conditional_edges(
        "classify",
        route_after_classify,
        {
            "answer": "answer",
            "tool": "tool",
            "clarify": "clarify",
            "risky_action": "risky_action",
            "retry": "retry",
            "dead_letter": "dead_letter",
        },
    )

    graph.add_edge("tool", "evaluate")
    graph.add_conditional_edges(
        "evaluate",
        route_after_evaluate,
        {"answer": "answer", "retry": "retry", "dead_letter": "dead_letter"},
    )
    graph.add_conditional_edges(
        "retry",
        route_after_retry,
        {"tool": "tool", "dead_letter": "dead_letter"},
    )

    graph.add_edge("risky_action", "approval")
    graph.add_conditional_edges(
        "approval",
        route_after_approval,
        {"tool": "tool", "clarify": "clarify"},
    )

    graph.add_edge("answer", "finalize")
    graph.add_edge("clarify", "finalize")
    graph.add_edge("dead_letter", "finalize")
    graph.add_edge("finalize", END)

    return graph.compile(checkpointer=checkpointer)
