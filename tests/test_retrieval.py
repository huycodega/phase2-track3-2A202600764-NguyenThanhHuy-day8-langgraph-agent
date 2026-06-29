"""Deterministic tests for the knowledge-base retriever and RAG grading set.

These assert the TF-IDF retriever returns the expected top-1 document and that
the expected answer keywords are present in the retrieved context. No LLM call is
required, so they run offline in CI.
"""

import json
from pathlib import Path

from langgraph_agent_lab.retrieval import get_knowledge_base, load_knowledge_base

ROOT = Path(__file__).resolve().parents[1]
QUESTIONS = json.loads((ROOT / "data" / "grading_questions.json").read_text(encoding="utf-8"))


def test_knowledge_base_loads():
    kb = load_knowledge_base()
    doc_ids = {doc["doc_id"] for doc in kb.documents}
    assert {
        "policy_refund_v4",
        "sla_p1_2026",
        "it_helpdesk_faq",
        "hr_leave_policy",
        "access_control_sop",
    } <= doc_ids


def test_grading_questions_top1_retrieval():
    kb = get_knowledge_base()
    for q in QUESTIONS:
        results = kb.search(q["question"], k=4)
        assert results, f"{q['id']}: no results"
        top1 = results[0].doc_id
        assert top1 == q["expect_top1_doc_id"], (
            f"{q['id']}: top1={top1} expected={q['expect_top1_doc_id']}"
        )


def test_grading_questions_contains_and_forbidden():
    kb = get_knowledge_base()
    for q in QUESTIONS:
        blob = " ".join(d.text for d in kb.search(q["question"], k=4)).lower()
        must_any = [m.lower() for m in q.get("must_contain_any", [])]
        if must_any:
            assert any(m in blob for m in must_any), f"{q['id']}: missing expected keyword"
        for forbidden in q.get("must_not_contain", []):
            assert forbidden.lower() not in blob, f"{q['id']}: forbidden '{forbidden}' present"
