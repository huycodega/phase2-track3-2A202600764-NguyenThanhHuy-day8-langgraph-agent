# Metrics Specification

`outputs/metrics.json` must validate against `MetricsReport` in `src/langgraph_agent_lab/metrics.py`.

Required fields:

- `total_scenarios`: number of scenarios executed. Minimum 6.
- `success_rate`: fraction of scenarios that meet expected route and output requirements.
- `avg_nodes_visited`: average number of audit events/nodes visited per scenario.
- `total_retries`: count of retry node visits across scenarios.
- `total_interrupts`: count of approval/HITL events across scenarios.
- `resume_success`: true if you demonstrate crash-resume or state-history replay.
- `scenario_metrics`: one object per scenario.

Each scenario metric should include:

- `scenario_id`
- `success`
- `expected_route`
- `actual_route`
- `nodes_visited`
- `retry_count`
- `interrupt_count`
- `approval_required`
- `approval_observed`
- `latency_ms`
- `errors`

## Grading notes

Metrics are not just numbers. Your report must explain why the numbers look the way they do.
