"""Node functions for the LangGraph workflow.

Each function receives ``AgentState`` and returns a partial state update dict.
Nodes never mutate the input state in-place.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Literal

from pydantic import BaseModel, Field

from .llm import DeterministicSupportLLM, get_llm
from .retrieval import get_knowledge_base
from .state import AgentState, Route, make_event


class ClassificationDecision(BaseModel):
    route: Literal["simple", "tool", "missing_info", "risky", "error"]
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str
    needs_approval: bool = False


class EvaluationDecision(BaseModel):
    is_sufficient: bool
    score: float = Field(ge=0.0, le=1.0)
    reason: str


class VaguenessDecision(BaseModel):
    is_missing_info: bool
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str


CLASSIFICATION_PROMPT = """You are routing a support-ticket agent.

Choose exactly one route:
- risky: side-effecting or destructive actions such as refunds, account deletion,
  cancellation, sending emails, changing billing, or irreversible operations.
- tool: information lookup/search/status requests that need a support-system tool.
- missing_info: vague or underspecified user requests where safe action is impossible.
- error: explicit system failures such as timeout, crash, service unavailable, or
  unrecoverable errors.
- simple: concrete general support questions answerable without tools or side effects.

Priority order if multiple apply: risky > tool > missing_info > error > simple.
Use missing_info when the user asks to fix/help/handle something but does not specify
the affected product, account, order, error symptom, or desired outcome.
Return a structured object with route, confidence, rationale, and needs_approval.

SUPPORT_TICKET:
{query}
END_TICKET
"""


VAGUENESS_REVIEW_PROMPT = """You are reviewing whether a support ticket is too vague.

Return is_missing_info=true if the ticket lacks enough concrete details to act safely,
especially when it asks to fix/help/handle "it", "this", or an unspecified issue.
Return false only when the ticket names a concrete topic, product, account/order,
error symptom, or desired support outcome.

SUPPORT_TICKET:
{query}
END_TICKET
"""


EVALUATION_PROMPT = """You are judging whether the latest tool result is sufficient to answer.

Mark is_sufficient=false if the tool result contains an error, timeout, unavailable
service, missing data, or a failed status. Otherwise mark it true.

LATEST_TOOL_RESULT:
{latest_tool_result}
END_TOOL_RESULT
"""


ANSWER_PROMPT = """You are a support-ticket agent writing the final user-facing response.

Rules:
- Ground the answer only in the support ticket, retrieved knowledge base, tool
  context, proposed action, and approval context below.
- Prefer concrete facts (numbers, durations, names) found in the knowledge base and
  quote them exactly. If the knowledge base answers the ticket, lead with that fact.
- Do not invent order status, policy details, or action completion that is not present in context.
- For risky actions, explain the approval status and avoid claiming a real side
  effect was performed.
- For missing information, ask for the minimum extra details needed.
- Answer in the same language as the support ticket. Keep the response concise and actionable.

ROUTE: {route}

SUPPORT_TICKET:
{query}
END_TICKET

KNOWLEDGE_BASE:
{retrieved_context}
END_KNOWLEDGE_BASE

TOOL_CONTEXT:
{tool_context}
END_TOOL_CONTEXT

PROPOSED_ACTION:
{proposed_action}
END_PROPOSED_ACTION

APPROVAL_CONTEXT:
{approval_context}
END_APPROVAL_CONTEXT
"""


def intake_node(state: AgentState) -> dict[str, Any]:
    """Normalize raw query."""

    query = state.get("query", "").strip()
    return {
        "query": query,
        "messages": [f"intake:{query[:40]}"],
        "events": [make_event("intake", "completed", "query normalized")],
    }


def classify_node(state: AgentState) -> dict[str, Any]:
    """Classify the query into a route using LLM structured output."""

    query = state.get("query", "").strip()
    prompt = CLASSIFICATION_PROMPT.format(query=query)
    try:
        llm = get_llm(temperature=0.0)
        decision = llm.with_structured_output(ClassificationDecision).invoke(prompt)
        classification = _model_dump(decision)
        source = "llm_structured_output"
    except Exception as exc:
        fallback = DeterministicSupportLLM().with_structured_output(ClassificationDecision)
        decision = fallback.invoke(prompt)
        classification = _model_dump(decision)
        source = "offline_fallback_after_llm_error"
        classification["fallback_error"] = str(exc)

    route = str(classification.get("route") or Route.MISSING_INFO.value)
    if route not in {item.value for item in Route if item not in {Route.DEAD_LETTER, Route.DONE}}:
        route = Route.MISSING_INFO.value
        classification["rationale"] = "Classifier returned an unknown route; defaulted safely."

    if route == Route.SIMPLE.value and _needs_missing_info_review(query, classification):
        review = _review_vagueness(query)
        classification["vagueness_review"] = review
        if review.get("is_missing_info") and float(review.get("confidence", 0.0)) >= 0.6:
            route = Route.MISSING_INFO.value
            classification["rationale"] = (
                f"{classification.get('rationale', '')} "
                f"Vagueness review: {review.get('reason', '')}"
            ).strip()

    needs_approval = bool(classification.get("needs_approval")) or route == Route.RISKY.value
    classification["needs_approval"] = needs_approval
    risk_level = "high" if route == Route.RISKY.value else "low"
    return {
        "route": route,
        "risk_level": risk_level,
        "classification": classification,
        "messages": [f"classify:{route}"],
        "events": [
            make_event(
                "classify",
                "completed",
                f"route={route}",
                confidence=classification.get("confidence"),
                source=source,
                rationale=classification.get("rationale"),
            )
        ],
    }


def tool_node(state: AgentState) -> dict[str, Any]:
    """Execute a mock support-system tool call."""

    route = state.get("route", Route.SIMPLE.value)
    should_retry = bool(state.get("should_retry")) or route == Route.ERROR.value
    prior_failure = any(
        not result.get("success", False) for result in state.get("tool_results", [])
    )
    should_fail = should_retry and not prior_failure

    if should_fail:
        result = {
            "tool_name": _tool_name_for_route(route),
            "status": "error",
            "content": "",
            "success": False,
            "error": "Transient support-system timeout; retry may recover.",
        }
        error_text = str(result["error"])
        tool_name = str(result["tool_name"])
        return {
            "tool_results": [result],
            "errors": [error_text],
            "messages": [f"tool:{result['tool_name']}:error"],
            "events": [make_event("tool", "failed", error_text, tool_name=tool_name)],
        }

    result = _successful_tool_result(state)
    content = str(result["content"])
    tool_name = str(result["tool_name"])
    return {
        "tool_results": [result],
        "messages": [f"tool:{tool_name}:success"],
        "events": [make_event("tool", "completed", content, tool_name=tool_name)],
    }


def evaluate_node(state: AgentState) -> dict[str, Any]:
    """Evaluate the latest tool result and decide whether retry is needed."""

    latest = _latest_tool_result(state)
    latest_json = json.dumps(latest, ensure_ascii=False, sort_keys=True)
    prompt = EVALUATION_PROMPT.format(latest_tool_result=latest_json)
    try:
        judge = get_llm(temperature=0.0).with_structured_output(EvaluationDecision)
        decision = judge.invoke(prompt)
        evaluation = _model_dump(decision)
        source = "llm_as_judge"
    except Exception as exc:
        decision = (
            DeterministicSupportLLM()
            .with_structured_output(EvaluationDecision)
            .invoke(prompt)
        )
        evaluation = _model_dump(decision)
        evaluation["fallback_error"] = str(exc)
        source = "heuristic_fallback"

    evaluation = _guard_evaluation_with_tool_schema(evaluation, latest)
    status = "success" if evaluation.get("is_sufficient") else "needs_retry"
    evaluation["status"] = status
    return {
        "evaluation_result": evaluation,
        "messages": [f"evaluate:{status}"],
        "events": [
            make_event(
                "evaluate",
                "completed",
                status,
                score=evaluation.get("score"),
                reason=evaluation.get("reason"),
                source=source,
            )
        ],
    }


def answer_node(state: AgentState) -> dict[str, Any]:
    """Generate a grounded final response using an LLM and knowledge-base retrieval."""

    query = state.get("query", "")
    retrieved_docs, top1_doc_id = _retrieve_context(query)
    retrieved_context = _format_retrieved_context(retrieved_docs)

    prompt = ANSWER_PROMPT.format(
        route=state.get("route", Route.SIMPLE.value),
        query=query,
        retrieved_context=retrieved_context,
        tool_context=json.dumps(state.get("tool_results", []), ensure_ascii=False),
        proposed_action=state.get("proposed_action") or "",
        approval_context=json.dumps(state.get("approval") or {}, ensure_ascii=False),
    )
    try:
        response = get_llm(temperature=0.2).invoke(prompt)
        answer = _message_content(response)
        source = "llm_grounded_generation"
    except Exception as exc:
        response = DeterministicSupportLLM().invoke(prompt)
        answer = _message_content(response)
        source = "offline_fallback_after_llm_error"
        answer = f"{answer}\n\nNote: offline fallback used because LLM generation failed: {exc}"

    answer = answer.strip() or _fallback_final_answer(state)
    return {
        "final_answer": answer,
        "retrieved_docs": retrieved_docs,
        "top1_doc_id": top1_doc_id,
        "messages": ["answer:final"],
        "events": [
            make_event(
                "answer",
                "completed",
                "final answer generated",
                source=source,
                top1_doc_id=top1_doc_id,
            )
        ],
    }


def ask_clarification_node(state: AgentState) -> dict[str, Any]:
    """Ask for missing information instead of hallucinating."""

    question = (
        "Could you share the affected product or account/order ID, what went wrong, "
        "and what outcome you want?"
    )
    approval = state.get("approval")
    if approval and not bool(approval.get("approved")):
        question = (
            "I cannot proceed with the requested action without approval. "
            "Would you like to provide an alternative, lower-risk request?"
        )
    return {
        "pending_question": question,
        "final_answer": question,
        "messages": ["clarify:pending_question"],
        "events": [make_event("clarify", "completed", "clarification requested")],
    }


def risky_action_node(state: AgentState) -> dict[str, Any]:
    """Prepare a risky action for human approval without executing it."""

    query = state.get("query", "").strip()
    proposed_action = (
        "Requires human approval before any side effect. Proposed support action: "
        f"{query or 'review and perform the requested account change'}."
    )
    return {
        "proposed_action": proposed_action,
        "messages": ["risky_action:approval_required"],
        "events": [
            make_event(
                "risky_action",
                "approval_required",
                "risky action prepared but not executed",
                proposed_action=proposed_action,
            )
        ],
    }


def approval_node(state: AgentState) -> dict[str, Any]:
    """Human-in-the-loop approval step with optional LangGraph interrupt."""

    if os.getenv("LANGGRAPH_INTERRUPT", "").lower() in {"1", "true", "yes"}:
        try:
            from langgraph.types import interrupt

            raw_decision = interrupt(
                {
                    "scenario_id": state.get("scenario_id"),
                    "query": state.get("query"),
                    "proposed_action": state.get("proposed_action"),
                    "instruction": "Approve or reject this risky support action.",
                }
            )
            approval = _coerce_approval(raw_decision)
            source = "langgraph_interrupt"
        except Exception as exc:
            approval = _mock_approval(comment=f"Interrupt unavailable, used mock approval: {exc}")
            source = "mock_after_interrupt_error"
    else:
        approval = _mock_approval()
        source = "mock"

    status = "approved" if approval["approved"] else "rejected"
    approval["status"] = status
    return {
        "approval": approval,
        "messages": [f"approval:{status}"],
        "events": [
            make_event(
                "approval",
                status,
                f"human approval {status}",
                reviewer=approval.get("reviewer"),
                source=source,
            )
        ],
    }


def retry_or_fallback_node(state: AgentState) -> dict[str, Any]:
    """Increment the bounded retry counter and record the failure."""

    current_attempt = int(state.get("attempt", 0))
    max_attempts = int(state.get("max_attempts", 3))
    next_attempt = current_attempt + 1
    message = (
        f"Retry attempt {next_attempt}/{max_attempts} scheduled "
        f"for route {state.get('route')}"
    )
    if next_attempt >= max_attempts:
        message = f"Retry budget exhausted at attempt {next_attempt}/{max_attempts}"
    return {
        "attempt": next_attempt,
        "errors": [message],
        "messages": [f"retry:{next_attempt}/{max_attempts}"],
        "events": [
            make_event(
                "retry",
                "completed",
                message,
                attempt=next_attempt,
                max_attempts=max_attempts,
            )
        ],
    }


def dead_letter_node(state: AgentState) -> dict[str, Any]:
    """Handle failures after the retry budget is exhausted."""

    answer = (
        "I could not complete this support request after the configured retry policy. "
        "It has been moved to the dead-letter queue for manual investigation."
    )
    return {
        "dead_letter": True,
        "final_answer": answer,
        "errors": ["Dead-letter fallback reached after retry exhaustion."],
        "messages": ["dead_letter:manual_review"],
        "events": [make_event("dead_letter", "completed", "manual investigation required")],
    }


def finalize_node(state: AgentState) -> dict[str, Any]:
    """Emit a final audit event before END."""

    final_answer = (
        state.get("final_answer")
        or state.get("pending_question")
        or _fallback_final_answer(state)
    )
    return {
        "final_answer": final_answer,
        "metrics": {
            "route": state.get("route"),
            "attempt": state.get("attempt", 0),
            "dead_letter": bool(state.get("dead_letter")),
            "approval_observed": state.get("approval") is not None,
        },
        "events": [
            make_event(
                "finalize",
                "completed",
                "workflow finished",
                route=state.get("route"),
                attempt=state.get("attempt", 0),
                has_final_answer=bool(final_answer),
            )
        ],
    }


def _model_dump(value: Any) -> dict[str, Any]:
    if isinstance(value, BaseModel):
        return value.model_dump()
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "model_dump"):
        return dict(value.model_dump())
    raise TypeError(f"Expected structured LLM output, got {type(value).__name__}")


def _retrieve_context(query: str, k: int = 4) -> tuple[list[dict[str, Any]], str | None]:
    """Retrieve top-k knowledge base docs for grounding; degrade gracefully."""
    if not query.strip():
        return [], None
    try:
        results = get_knowledge_base().search(query, k=k)
    except Exception:
        return [], None
    docs = [doc.to_dict() for doc in results]
    top1_doc_id = docs[0]["doc_id"] if docs else None
    return docs, top1_doc_id


def _format_retrieved_context(retrieved_docs: list[dict[str, Any]]) -> str:
    if not retrieved_docs:
        return "(no knowledge base match)"
    return "\n".join(
        f"[{doc.get('doc_id')}] {doc.get('title')}: {doc.get('text')}"
        for doc in retrieved_docs
    )


def _message_content(response: Any) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, list):
        return " ".join(str(part) for part in content)
    return str(content)


def _tool_name_for_route(route: str) -> str:
    if route == Route.TOOL.value:
        return "order_status_lookup"
    if route == Route.RISKY.value:
        return "approval_gate"
    return "support_diagnostics"


def _latest_tool_result(state: AgentState) -> dict[str, Any]:
    results = state.get("tool_results", [])
    if not results:
        return {
            "tool_name": "none",
            "status": "missing",
            "content": "",
            "success": False,
            "error": "No tool result is available.",
        }
    return results[-1]


def _needs_missing_info_review(query: str, classification: dict[str, Any]) -> bool:
    words = re.findall(r"[A-Za-z0-9]+", query)
    confidence = float(classification.get("confidence", 1.0) or 0.0)
    return confidence < 0.9 or len(words) <= 6


def _review_vagueness(query: str) -> dict[str, Any]:
    prompt = VAGUENESS_REVIEW_PROMPT.format(query=query)
    try:
        review = get_llm(temperature=0.0).with_structured_output(VaguenessDecision).invoke(prompt)
        return _model_dump(review)
    except Exception as exc:
        fallback_is_vague = len(re.findall(r"[A-Za-z0-9]+", query)) <= 4
        return {
            "is_missing_info": fallback_is_vague,
            "confidence": 0.7 if fallback_is_vague else 0.3,
            "reason": f"Fallback vagueness review after LLM error: {exc}",
        }


def _guard_evaluation_with_tool_schema(
    evaluation: dict[str, Any],
    latest: dict[str, Any],
) -> dict[str, Any]:
    guarded = dict(evaluation)
    status = str(latest.get("status", "")).lower()
    error = latest.get("error")
    if latest.get("success") is False or bool(error) or status in {"error", "failed", "missing"}:
        guarded["is_sufficient"] = False
        guarded["score"] = min(float(guarded.get("score", 0.0)), 0.3)
        guarded["reason"] = f"{guarded.get('reason', '')} Tool schema indicates failure.".strip()
    elif latest.get("success") is True:
        guarded["is_sufficient"] = True
        guarded["score"] = max(float(guarded.get("score", 0.0)), 0.85)
        guarded["reason"] = f"{guarded.get('reason', '')} Tool schema indicates success.".strip()
    return guarded


def _successful_tool_result(state: AgentState) -> dict[str, Any]:
    route = state.get("route", Route.SIMPLE.value)
    if route == Route.TOOL.value:
        order_id = _extract_order_id(state.get("query", ""))
        return {
            "tool_name": "order_status_lookup",
            "status": "success",
            "content": (
                f"Order {order_id or 'requested'} is in processing status "
                "and ready for normal follow-up."
            ),
            "success": True,
            "error": None,
        }
    if route == Route.RISKY.value:
        approval = state.get("approval") or {}
        return {
            "tool_name": "approval_gate",
            "status": "approved" if approval.get("approved") else "blocked",
            "content": "Risky action has approval recorded; no external side effect was executed.",
            "success": bool(approval.get("approved")),
            "error": None if approval.get("approved") else "Approval missing or rejected.",
        }
    return {
        "tool_name": "support_diagnostics",
        "status": "success",
        "content": "Transient system issue recovered after retry.",
        "success": True,
        "error": None,
    }


def _extract_order_id(query: str) -> str | None:
    match = re.search(r"\border\s*#?\s*([A-Za-z0-9-]+)", query, flags=re.IGNORECASE)
    if match:
        return match.group(1)
    match = re.search(r"\b[A-Z0-9]{4,}\b", query, flags=re.IGNORECASE)
    return match.group(0) if match else None


def _coerce_approval(raw_decision: Any) -> dict[str, Any]:
    if isinstance(raw_decision, dict):
        approved = bool(raw_decision.get("approved", raw_decision.get("status") == "approved"))
        return {
            "approved": approved,
            "reviewer": str(raw_decision.get("reviewer", "human-reviewer")),
            "comment": str(raw_decision.get("comment", raw_decision.get("reason", ""))),
        }
    if isinstance(raw_decision, bool):
        return {
            "approved": raw_decision,
            "reviewer": "human-reviewer",
            "comment": "Boolean approval response from interrupt.",
        }
    return _mock_approval(comment=f"Unrecognized interrupt payload: {raw_decision!r}")


def _mock_approval(comment: str | None = None) -> dict[str, Any]:
    rejected = os.getenv("MOCK_APPROVAL", "approved").lower() in {
        "reject",
        "rejected",
        "false",
        "0",
    }
    approved = not rejected
    return {
        "approved": approved,
        "reviewer": "mock-reviewer",
        "comment": comment
        or (
            "Approved by deterministic lab reviewer."
            if approved
            else "Rejected by mock config."
        ),
    }


def _fallback_final_answer(state: AgentState) -> str:
    route = state.get("route", Route.SIMPLE.value)
    if route == Route.MISSING_INFO.value:
        return "I need a little more information before I can help safely."
    if route == Route.RISKY.value:
        return "This request requires human approval before any side-effecting action can proceed."
    if route == Route.ERROR.value:
        return "The request hit a system error and needs manual follow-up."
    return "I can help with that support request."
