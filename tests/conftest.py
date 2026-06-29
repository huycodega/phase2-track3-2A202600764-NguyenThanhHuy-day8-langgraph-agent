"""Shared pytest configuration.

Loads the project ``.env`` before test collection so that LLM-gated smoke tests
can detect a configured provider key (GROQ/GEMINI/OPENAI/ANTHROPIC) via
``os.getenv`` at collection time.
"""

from __future__ import annotations

try:
    from dotenv import load_dotenv

    load_dotenv(encoding="utf-8-sig")
except ImportError:  # pragma: no cover - dotenv is optional
    pass
