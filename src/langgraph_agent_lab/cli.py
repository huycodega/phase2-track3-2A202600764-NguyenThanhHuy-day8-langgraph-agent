"""CLI for the lab."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
import yaml

from .graph import build_graph
from .metrics import MetricsReport, metric_from_state, summarize_metrics, write_metrics
from .persistence import build_checkpointer
from .report import write_report
from .retrieval import get_knowledge_base
from .scenarios import load_scenarios
from .state import Route, Scenario, initial_state

app = typer.Typer(no_args_is_help=True)


@app.command("run-scenarios")
def run_scenarios(
    config: Annotated[Path, typer.Option("--config")],
    output: Annotated[Path, typer.Option("--output")],
) -> None:
    """Run all grading scenarios and write metrics JSON."""
    cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
    scenarios = load_scenarios(cfg["scenarios_path"])
    checkpointer = build_checkpointer(cfg.get("checkpointer", "memory"), cfg.get("database_url"))
    graph = build_graph(checkpointer=checkpointer)
    metrics = []
    for scenario in scenarios:
        state = initial_state(scenario)
        run_config = {"configurable": {"thread_id": state["thread_id"]}}
        final_state = graph.invoke(state, config=run_config)
        metrics.append(
            metric_from_state(
                final_state,
                scenario.expected_route.value,
                scenario.requires_approval,
            )
        )
    report = summarize_metrics(metrics)
    write_metrics(report, output)
    if cfg.get("report_path"):
        write_report(report, cfg["report_path"])
    typer.echo(f"Wrote metrics to {output}")


@app.command("grade-rag")
def grade_rag(
    questions: Annotated[
        Path, typer.Option("--questions")
    ] = Path("data/grading_questions.json"),
    output: Annotated[
        Path, typer.Option("--output")
    ] = Path("outputs/rag_grading.jsonl"),
    top_k: Annotated[int, typer.Option("--top-k")] = 4,
) -> None:
    """Grade RAG questions: retrieval top-1 + grounded answer keyword checks."""
    qs = json.loads(questions.read_text(encoding="utf-8"))
    kb = get_knowledge_base()
    graph = build_graph(checkpointer=build_checkpointer("memory"))

    output.parent.mkdir(parents=True, exist_ok=True)
    passed = 0
    records = []
    for q in qs:
        text = str(q["question"])
        retrieved = kb.search(text, k=top_k)
        top1_doc_id = retrieved[0].doc_id if retrieved else ""
        retrieved_blob = " ".join(doc.text for doc in retrieved).lower()

        scenario = Scenario(id=str(q["id"]), query=text, expected_route=Route.TOOL)
        state = initial_state(scenario)
        run_config = {"configurable": {"thread_id": state["thread_id"]}}
        final_state = graph.invoke(state, config=run_config)
        answer = str(final_state.get("final_answer") or "")

        haystack = f"{answer}\n{retrieved_blob}".lower()
        must_any = [m.lower() for m in q.get("must_contain_any", [])]
        forbidden = [m.lower() for m in q.get("must_not_contain", [])]
        contains_expected = any(m in haystack for m in must_any) if must_any else True
        hits_forbidden = any(m in haystack for m in forbidden) if forbidden else False
        want_top1 = str(q.get("expect_top1_doc_id") or "").strip()
        top1_ok = (top1_doc_id == want_top1) if want_top1 else True
        ok = contains_expected and not hits_forbidden and top1_ok
        passed += int(ok)

        records.append(
            {
                "id": q.get("id"),
                "question": text,
                "passed": ok,
                "top1_doc_id": top1_doc_id,
                "top1_doc_matches": top1_ok if want_top1 else None,
                "contains_expected": contains_expected,
                "hits_forbidden": hits_forbidden,
                "answer": answer,
                "grading_criteria": q.get("grading_criteria", []),
            }
        )

    with output.open("w", encoding="utf-8") as handle:
        for rec in records:
            handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
    rate = passed / len(qs) if qs else 0.0
    typer.echo(f"RAG grading: {passed}/{len(qs)} passed ({rate:.0%}). Wrote {output}")
    if passed != len(qs):
        raise typer.Exit(code=1)


@app.command("validate-metrics")
def validate_metrics(metrics: Annotated[Path, typer.Option("--metrics")]) -> None:
    """Validate metrics JSON schema for grading."""
    payload = json.loads(metrics.read_text(encoding="utf-8"))
    report = MetricsReport.model_validate(payload)
    if report.total_scenarios < 6:
        raise typer.BadParameter("Expected at least 6 scenarios")
    typer.echo(f"Metrics valid. success_rate={report.success_rate:.2%}")


if __name__ == "__main__":
    app()
