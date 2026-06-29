"""Report generation helper."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from .metrics import MetricsReport, ScenarioMetric


def render_report(metrics: MetricsReport) -> str:
    """Render a complete lab report from metrics data."""

    scenario_rows = "\n".join(_scenario_row(item) for item in metrics.scenario_metrics)
    failure_notes = _failure_analysis(metrics)
    return f"""# Day 08 Lab Report - LangGraph Agentic Orchestration

## 1. Team / Student

- Name: Luong Trung Duc
- Repo/commit: local workspace submission
- Date: {datetime.now(UTC).date().isoformat()}

## 2. Architecture Summary

The workflow is a production-style support-ticket graph. It normalizes input, uses an LLM
structured-output classifier, routes by state, executes mock support tools when needed,
uses an LLM-as-judge gate for tool quality, handles bounded retries, gates risky actions
through HITL approval, and sends every branch through `finalize -> END`.

```mermaid
flowchart TD
    START([START]) --> intake
    intake --> classify
    classify -->|simple| answer
    classify -->|tool| tool
    classify -->|missing_info| clarify
    classify -->|risky| risky_action
    classify -->|error| retry
    risky_action --> approval
    approval -->|approved| tool
    approval -->|rejected| clarify
    tool --> evaluate
    evaluate -->|sufficient| answer
    evaluate -->|needs retry| retry
    retry -->|attempt < max_attempts| tool
    retry -->|attempt >= max_attempts| dead_letter
    dead_letter --> finalize
    answer --> finalize
    clarify --> finalize
    finalize --> END([END])
```

## 3. State Schema

State is intentionally lean and checkpoint-safe: only strings, numbers, booleans, dicts,
and lists are stored. Append-only reducers are used for audit trails; scalar fields are
overwritten by the node that owns them.

| Field | Reducer | Purpose |
|---|---|---|
| `query`, `route`, `risk_level` | overwrite | Current ticket and route decision |
| `classification` | overwrite | Structured LLM classification with confidence/rationale |
| `tool_results` | append | JSON-serializable mock tool outputs |
| `evaluation_result` | overwrite | LLM-as-judge retry gate |
| `pending_question` | overwrite | Clarification request for missing info |
| `proposed_action`, `approval` | overwrite | HITL approval context for risky actions |
| `attempt`, `max_attempts` | overwrite | Bounded retry control |
| `final_answer` | overwrite | Final user-facing response |
| `messages`, `errors`, `events` | append | Traceability, audit, and metrics |

## 4. Routing Table

| Route | Trigger | Next node |
|---|---|---|
| `simple` | General support question | `answer` |
| `tool` | Lookup/status/search needed | `tool -> evaluate` |
| `missing_info` | Vague or underspecified request | `clarify` |
| `risky` | Refund/delete/send/cancel or side effect | `risky_action -> approval` |
| `error` | Timeout/crash/service failure | `retry` |
| retry exhausted | `attempt >= max_attempts` | `dead_letter` |

## 5. Metrics Summary

| Metric | Value |
|---|---:|
| Total scenarios | {metrics.total_scenarios} |
| Success rate | {metrics.success_rate:.1%} |
| Route accuracy | {metrics.route_accuracy:.1%} |
| Average nodes visited | {metrics.avg_nodes_visited:.2f} |
| Total retries | {metrics.total_retries} |
| HITL approvals observed | {metrics.hitl_triggered} |
| Total approval node visits | {metrics.total_interrupts} |
| Dead-letter count | {metrics.dead_letter_count} |
| Missing final answers | {metrics.invalid_final_answer_count} |

## 6. Scenario Results

| Scenario | Expected | Actual | Success | Retries | HITL | Dead letter | Errors |
|---|---|---|---:|---:|---:|---:|---|
{scenario_rows}

## 7. Failure Analysis

{failure_notes}

## 8. Persistence / Recovery Evidence

The graph is compiled with a caller-provided checkpointer. Local runs use `MemorySaver`
by default, and `build_checkpointer("sqlite", "outputs/checkpoints.sqlite")` enables
SQLite persistence via `SqliteSaver(conn=sqlite3.connect(...))` with WAL mode. Each
scenario starts with a unique `thread_id` such as `thread-S01_simple`, so state history
and crash recovery can be scoped per ticket.

## 9. Extension Work

- SQLite checkpoint support was implemented as a persistence extension.
- The report includes a Mermaid graph diagram for demo/readability.
- `approval_node` supports `LANGGRAPH_INTERRUPT=true` for real LangGraph HITL interrupts,
  while keeping mock approval as the CI-safe default.

## 10. Improvement Plan

With another day, I would add provider-level retries around LLM calls, richer tool
schemas for real CRM/order systems, human reviewer identity propagation, and replay
tests that assert SQLite state history across process restarts.
"""


def write_report(metrics: MetricsReport, output_path: str | Path) -> None:
    """Write the rendered report to a file."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_report(metrics), encoding="utf-8")


def _scenario_row(item: ScenarioMetric) -> str:
    errors = "; ".join(item.errors) if item.errors else "-"
    return (
        f"| {item.scenario_id} | {item.expected_route} | {item.actual_route or '-'} | "
        f"{'yes' if item.success else 'no'} | {item.retry_count} | "
        f"{item.interrupt_count} | {'yes' if item.dead_letter else 'no'} | {errors} |"
    )


def _failure_analysis(metrics: MetricsReport) -> str:
    notes: list[str] = [
        "Retry/tool failure: transient tool errors are represented as failed `tool_results`; "
        "`evaluate` marks them insufficient and routes through `retry` until the "
        "bounded budget is exhausted.",
        "Risky action without approval: side-effecting requests first create `proposed_action`; "
        "`approval` must record a decision before the graph proceeds.",
    ]
    if metrics.dead_letter_count:
        notes.append(
            f"Dead-letter fallback: {metrics.dead_letter_count} scenario(s) exhausted "
            "retries and were "
            "converted into manual-investigation final answers instead of looping indefinitely."
        )
    if metrics.failure_count:
        notes.append(
            f"Residual failures: {metrics.failure_count} scenario(s) did not meet "
            "route/output criteria; "
            "inspect per-scenario errors above."
        )
    else:
        notes.append("Residual failures: none observed in the scenario suite.")
    return "\n".join(f"{index}. {note}" for index, note in enumerate(notes, start=1))
