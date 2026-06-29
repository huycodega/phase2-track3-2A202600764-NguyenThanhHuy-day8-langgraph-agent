"""Checkpointer adapter."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any


def build_checkpointer(kind: str = "memory", database_url: str | None = None) -> Any | None:
    """Return a LangGraph checkpointer."""

    normalized = kind.lower().strip()
    if normalized == "none":
        return None
    if normalized == "memory":
        from langgraph.checkpoint.memory import MemorySaver

        return MemorySaver()
    if normalized == "sqlite":
        return _build_sqlite_checkpointer(database_url)
    if normalized == "postgres":
        raise NotImplementedError(
            "Postgres checkpointing is optional for this lab; use CHECKPOINTER=sqlite locally."
        )
    raise ValueError(f"Unknown checkpointer kind: {kind}")


def _build_sqlite_checkpointer(database_url: str | None) -> Any:
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
    except ImportError as exc:
        raise RuntimeError(
            "Install SQLite checkpoint support: pip install langgraph-checkpoint-sqlite"
        ) from exc

    db_path = _sqlite_path(database_url)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    saver = SqliteSaver(conn=conn)
    setup = getattr(saver, "setup", None)
    if callable(setup):
        setup()
    return saver


def _sqlite_path(database_url: str | None) -> Path:
    if not database_url:
        return Path("outputs") / "checkpoints.sqlite"
    if database_url.startswith("sqlite:///"):
        return Path(database_url.removeprefix("sqlite:///"))
    if database_url.startswith("sqlite://"):
        return Path(database_url.removeprefix("sqlite://"))
    return Path(database_url)
