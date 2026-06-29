"""State schema for the Day 08 LangGraph lab.

Students should extend the schema only when needed. Keep state lean and serializable.
"""

from __future__ import annotations

from enum import StrEnum
from operator import add
from typing import Annotated, Any, TypedDict

from pydantic import BaseModel, Field, field_validator


class Route(StrEnum):
    SIMPLE = "simple"
    TOOL = "tool"
    MISSING_INFO = "missing_info"
    RISKY = "risky"
    ERROR = "error"
    DEAD_LETTER = "dead_letter"
    DONE = "done"


class LabEvent(BaseModel):
    """Append-only audit event for grading and debugging."""

    node: str
    event_type: str
    message: str
    latency_ms: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class ApprovalDecision(BaseModel):
    approved: bool = False
    reviewer: str = "mock-reviewer"
    comment: str = ""


class ClassificationResult(BaseModel):
    route: Route
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str
    needs_approval: bool = False


class ToolResult(BaseModel):
    tool_name: str
    status: str
    content: str
    success: bool
    error: str | None = None


class EvaluationResult(BaseModel):
    is_sufficient: bool
    score: float = Field(ge=0.0, le=1.0)
    reason: str


class AgentState(TypedDict, total=False):
    """LangGraph state.

    Keep values JSON-serializable so memory/SQLite checkpoints can persist state.
    Append-only fields use reducers; scalar/dict fields are overwritten by each node.
    """

    thread_id: str
    scenario_id: str
    query: str
    route: str
    risk_level: str
    classification: dict[str, Any] | None
    attempt: int
    max_attempts: int
    should_retry: bool
    requires_approval: bool
    tags: list[str]
    final_answer: str | None
    evaluation_result: str | dict[str, Any] | None
    pending_question: str | None
    proposed_action: str | None
    approval: dict[str, Any] | None
    dead_letter: bool
    metrics: dict[str, Any]
    messages: Annotated[list[str | dict[str, Any]], add]
    tool_results: Annotated[list[dict[str, Any]], add]
    errors: Annotated[list[str], add]
    events: Annotated[list[dict[str, Any]], add]


class Scenario(BaseModel):
    id: str
    query: str
    expected_route: Route
    requires_approval: bool = False
    should_retry: bool = False
    max_attempts: int = 3
    tags: list[str] = Field(default_factory=list)

    @field_validator("query")
    @classmethod
    def query_must_not_be_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query must not be empty")
        return value


def initial_state(scenario: Scenario) -> AgentState:
    """Create a serializable initial state for one scenario."""
    return {
        "thread_id": f"thread-{scenario.id}",
        "scenario_id": scenario.id,
        "query": scenario.query,
        "route": "",
        "risk_level": "unknown",
        "classification": None,
        "attempt": 0,
        "max_attempts": scenario.max_attempts,
        "should_retry": scenario.should_retry,
        "requires_approval": scenario.requires_approval,
        "tags": list(scenario.tags),
        "final_answer": None,
        "evaluation_result": None,
        "pending_question": None,
        "proposed_action": None,
        "approval": None,
        "dead_letter": False,
        "metrics": {},
        "messages": [],
        "tool_results": [],
        "errors": [],
        "events": [],
    }


def make_event(node: str, event_type: str, message: str, **metadata: Any) -> dict[str, Any]:
    """Create a normalized event payload."""
    return LabEvent(
        node=node,
        event_type=event_type,
        message=message,
        metadata=metadata,
    ).model_dump()
