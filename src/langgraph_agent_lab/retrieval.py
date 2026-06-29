"""Lightweight knowledge-base retrieval for grounded answers.

This module implements a dependency-free TF-IDF retriever over a small JSON
knowledge base so the support agent can ground its answers in policy documents
(refund policy, SLA, IT helpdesk, HR leave, access control) and so the RAG
grading questions can be evaluated deterministically (no vector DB required).

It is intentionally Unicode-aware so Vietnamese documents tokenize correctly.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@dataclass(frozen=True)
class RetrievedDoc:
    doc_id: str
    title: str
    text: str
    score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "score": round(self.score, 4),
            "text": self.text,
        }


@dataclass
class KnowledgeBase:
    """In-memory TF-IDF index over a list of documents."""

    documents: list[dict[str, str]]
    _idf: dict[str, float] = field(default_factory=dict, init=False)
    _doc_vectors: list[dict[str, float]] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self._build_index()

    def _build_index(self) -> None:
        n_docs = len(self.documents)
        doc_term_freqs: list[Counter[str]] = []
        df: Counter[str] = Counter()
        for doc in self.documents:
            tokens = _tokenize(f"{doc.get('title', '')} {doc.get('text', '')}")
            tf = Counter(tokens)
            doc_term_freqs.append(tf)
            df.update(tf.keys())

        self._idf = {
            term: math.log((n_docs + 1) / (doc_freq + 1)) + 1.0
            for term, doc_freq in df.items()
        }
        self._doc_vectors = [self._vectorize(tf) for tf in doc_term_freqs]

    def _vectorize(self, tf: Counter[str]) -> dict[str, float]:
        vec = {
            term: (1.0 + math.log(count)) * self._idf.get(term, 0.0)
            for term, count in tf.items()
        }
        norm = math.sqrt(sum(weight * weight for weight in vec.values())) or 1.0
        return {term: weight / norm for term, weight in vec.items()}

    def search(self, query: str, k: int = 4) -> list[RetrievedDoc]:
        query_tf = Counter(_tokenize(query))
        if not query_tf:
            return []
        query_vec = self._vectorize(query_tf)
        scored: list[RetrievedDoc] = []
        for doc, doc_vec in zip(self.documents, self._doc_vectors, strict=True):
            score = _cosine(query_vec, doc_vec)
            scored.append(
                RetrievedDoc(
                    doc_id=str(doc.get("doc_id", "")),
                    title=str(doc.get("title", "")),
                    text=str(doc.get("text", "")),
                    score=score,
                )
            )
        scored.sort(key=lambda d: d.score, reverse=True)
        return scored[:k]


def _cosine(vec_a: dict[str, float], vec_b: dict[str, float]) -> float:
    if len(vec_a) > len(vec_b):
        vec_a, vec_b = vec_b, vec_a
    return sum(weight * vec_b.get(term, 0.0) for term, weight in vec_a.items())


def default_kb_path() -> Path:
    """Resolve the bundled knowledge base regardless of current working dir."""
    project_root = Path(__file__).resolve().parents[2]
    return project_root / "data" / "kb" / "knowledge_base.json"


def load_knowledge_base(path: str | Path | None = None) -> KnowledgeBase:
    kb_path = Path(path) if path else default_kb_path()
    documents = json.loads(kb_path.read_text(encoding="utf-8"))
    if not isinstance(documents, list) or not documents:
        raise ValueError(f"Knowledge base at {kb_path} is empty or malformed")
    return KnowledgeBase(documents=documents)


@lru_cache(maxsize=1)
def get_knowledge_base() -> KnowledgeBase:
    """Return a process-wide cached knowledge base."""
    return load_knowledge_base()
