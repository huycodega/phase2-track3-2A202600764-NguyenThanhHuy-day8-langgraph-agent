"""Routing functions for conditional edges."""

from __future__ import annotations

from typing import Any

from .state import AgentState, Route


def route_after_classify(state: AgentState) -> str:
    """Map classified route to the next graph node."""

    route = state.get("route", "")
    if route == Route.ERROR.value and _attempt_exhausted(state):
        return "dead_letter"
    return {
        Route.SIMPLE.value: "answer",
        Route.TOOL.value: "tool",
        Route.MISSING_INFO.value: "clarify",
        Route.RISKY.value: "risky_action",
        Route.ERROR.value: "retry",
    }.get(route, "answer")


def route_after_evaluate(state: AgentState) -> str:
    """Route successful tool results to answer and failed results to retry/dead-letter."""

    status = _evaluation_status(state.get("evaluation_result"))
    if status == "needs_retry":
        return "dead_letter" if _attempt_exhausted(state) else "retry"
    return "answer"


def route_after_retry(state: AgentState) -> str:
    """Bound the retry loop by ``attempt < max_attempts``."""

    return "dead_letter" if _attempt_exhausted(state) else "tool"


def route_after_approval(state: AgentState) -> str:
    """Route based on human approval decision."""

    approval = state.get("approval") or {}
    if bool(approval.get("approved")) or approval.get("status") == "approved":
        return "tool"
    return "clarify"


def _attempt_exhausted(state: AgentState) -> bool:
    return int(state.get("attempt", 0)) >= int(state.get("max_attempts", 3))


def _evaluation_status(evaluation_result: Any) -> str:
    if isinstance(evaluation_result, str):
        return evaluation_result
    if isinstance(evaluation_result, dict):
        if "status" in evaluation_result:
            return str(evaluation_result["status"])
        return "success" if evaluation_result.get("is_sufficient") else "needs_retry"
    return "success"
