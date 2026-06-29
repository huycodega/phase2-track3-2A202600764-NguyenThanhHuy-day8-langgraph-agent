# Grading Rubric

| Category | Points | Evidence |
|---|---:|---|
| Architecture and state schema | 15 | Typed state with correct reducers, student-added fields (evaluation_result, proposed_action, etc.), lean serializable state, clear node boundaries |
| Graph construction and wiring | 15 | All nodes registered, edges correctly wired, conditional edges use routing functions, graph compiles and runs |
| LLM integration | 15 | classify_node uses LLM with structured output, answer_node uses LLM for grounded response, proper prompt engineering |
| Graph behavior | 20 | Correct routes for all scenarios, bounded retry loop, HITL approval path, all routes terminate at finalize→END |
| Persistence and recovery | 10 | Checkpointer wired, thread_id per run, state history or crash-resume evidence |
| Metrics and tests | 15 | `metrics.json` valid, scenario coverage, tests pass, meaningful error/retry counts |
| Report and demo | 10 | Architecture explanation, metrics table, failure analysis, improvement plan |

## LLM integration detail

| Node | Requirement | What we check |
|---|---|---|
| `classify_node` | **MUST use LLM** | Structured output call visible in code. Not keyword-only matching |
| `answer_node` | **MUST use LLM** | LLM generates response grounded in tool_results/context. Not hardcoded strings |
| `evaluate_node` | **SHOULD use LLM** (bonus) | LLM-as-judge pattern. Heuristic check is acceptable for base score |

Students who use only keyword heuristics for classify_node or hardcoded strings for answer_node will lose up to 15 points.

## Suggested grade bands

- 90-100: Production-quality graph + LLM integration + metrics + report + at least one extension.
- 75-89: Core graph works with LLM, metrics valid, report explains trade-offs.
- 60-74: Graph mostly works but LLM integration, persistence, or report incomplete.
- <60: Does not run, hard-codes scenarios, or lacks LLM integration/metrics/report.
