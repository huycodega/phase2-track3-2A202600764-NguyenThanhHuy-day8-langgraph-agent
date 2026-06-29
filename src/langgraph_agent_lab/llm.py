"""LLM factory helper.

The production path returns a real LangChain chat model. For local tests without
secrets, this module also exposes a deterministic offline model with the same
minimal interface used by the nodes: ``invoke`` and ``with_structured_output``.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DeterministicMessage:
    """Small AIMessage-compatible payload used by the offline fallback."""

    content: str


class DeterministicSupportLLM:
    """Offline LLM-shaped fallback for demos and CI.

    It intentionally classifies by broad support-ticket intent categories rather
    than scenario ids or exact sample queries.
    """

    def __init__(self, structured_schema: type[Any] | None = None) -> None:
        self._structured_schema = structured_schema

    def with_structured_output(self, schema: type[Any]) -> DeterministicSupportLLM:
        return DeterministicSupportLLM(structured_schema=schema)

    def invoke(self, prompt: Any) -> Any:
        text = _prompt_to_text(prompt)
        if self._structured_schema is not None:
            return self._invoke_structured(text)
        return DeterministicMessage(content=_draft_grounded_response(text))

    def _invoke_structured(self, prompt: str) -> Any:
        schema = self._structured_schema
        if schema is None:
            raise RuntimeError("Structured schema was not configured")

        fields = getattr(schema, "model_fields", {})
        if "route" in fields:
            payload = _classify_ticket(_extract_between(prompt, "SUPPORT_TICKET", "END_TICKET"))
        elif "is_sufficient" in fields:
            payload = _judge_tool_result(
                _extract_between(prompt, "LATEST_TOOL_RESULT", "END_TOOL_RESULT")
            )
        else:
            payload = {}
        return schema(**payload)


def get_llm(
    model: str | None = None,
    temperature: float = 0.0,
    *,
    allow_fallback: bool | None = None,
) -> Any:
    """Create an LLM client from environment configuration.

    Checks for API keys in this order:
    1. GEMINI_API_KEY -> ChatGoogleGenerativeAI
    2. GROQ_API_KEY -> ChatGroq
    3. OPENAI_API_KEY -> ChatOpenAI
    4. ANTHROPIC_API_KEY -> ChatAnthropic

    If no valid key is configured, the default is a deterministic offline model
    so the lab can still run locally. Set ``LLM_STRICT=true`` or pass
    ``allow_fallback=False`` to require a real provider.
    """

    _load_dotenv_if_available()
    fallback_allowed = _fallback_allowed() if allow_fallback is None else allow_fallback

    gemini_key = _valid_secret("GEMINI_API_KEY")
    if gemini_key:
        try:
            from langchain_google_genai import (
                ChatGoogleGenerativeAI,  # type: ignore[import-not-found]
            )
        except ImportError as exc:
            raise RuntimeError(
                "Install Gemini support: pip install langchain-google-genai"
            ) from exc
        return ChatGoogleGenerativeAI(
            model=_model_name(model, "gemini-2.5-flash"),
            google_api_key=gemini_key,
            temperature=temperature,
        )

    groq_key = _valid_secret("GROQ_API_KEY")
    if groq_key:
        try:
            from langchain_groq import ChatGroq  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError("Install Groq support: pip install langchain-groq") from exc
        return ChatGroq(
            model=_model_name(model, "llama-3.3-70b-versatile"),
            temperature=temperature,
            timeout=_llm_timeout(),
            max_retries=_llm_max_retries(),
        )

    openai_key = _valid_secret("OPENAI_API_KEY")
    if openai_key:
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise RuntimeError("Install OpenAI support: pip install langchain-openai") from exc
        return ChatOpenAI(
            model=_model_name(model, "gpt-4o-mini"),
            temperature=temperature,
        )

    anthropic_key = _valid_secret("ANTHROPIC_API_KEY")
    if anthropic_key:
        try:
            from langchain_anthropic import ChatAnthropic  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "Install Anthropic support: pip install langchain-anthropic"
            ) from exc
        return ChatAnthropic(
            model=_model_name(model, "claude-sonnet-4-20250514"),
            temperature=temperature,
        )

    if fallback_allowed:
        return DeterministicSupportLLM()

    raise RuntimeError(
        "No valid LLM API key found. Set GEMINI_API_KEY, GROQ_API_KEY, OPENAI_API_KEY, "
        "or ANTHROPIC_API_KEY in .env. Placeholder values such as 'AIza...' are ignored."
    )


def _load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(encoding="utf-8-sig")


def _fallback_allowed() -> bool:
    if os.getenv("LLM_STRICT", "").lower() in {"1", "true", "yes"}:
        return False
    return os.getenv("ALLOW_FAKE_LLM", "true").lower() not in {"0", "false", "no"}


def _valid_secret(name: str) -> str | None:
    value = os.getenv(name, "").strip()
    if not value:
        return None
    lowered = value.lower()
    placeholder_fragments = ("...", "changeme", "your_", "replace_me", "example")
    if any(fragment in lowered for fragment in placeholder_fragments):
        return None
    return value


def _model_name(model: str | None, default: str) -> str:
    return model or os.getenv("LLM_MODEL") or default


def _llm_timeout() -> float:
    return float(os.getenv("LLM_TIMEOUT_SECONDS", "20"))


def _llm_max_retries() -> int:
    return int(os.getenv("LLM_MAX_RETRIES", "1"))


def _prompt_to_text(prompt: Any) -> str:
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list):
        parts: list[str] = []
        for item in prompt:
            if isinstance(item, tuple) and len(item) == 2:
                parts.append(str(item[1]))
            elif hasattr(item, "content"):
                parts.append(str(item.content))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    if hasattr(prompt, "to_string"):
        return str(prompt.to_string())
    return str(prompt)


def _extract_between(text: str, start_marker: str, end_marker: str) -> str:
    pattern = rf"{re.escape(start_marker)}:\s*(.*?)\s*{re.escape(end_marker)}"
    match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def _classify_ticket(query: str) -> dict[str, Any]:
    normalized = query.lower()
    risky_terms = (
        "refund",
        "delete",
        "remove account",
        "send email",
        "send confirmation",
        "cancel",
        "chargeback",
        "close account",
        "terminate",
    )
    tool_terms = (
        "lookup",
        "look up",
        "order",
        "tracking",
        "status",
        "invoice",
        "search",
        "find",
        "where is",
    )
    error_terms = (
        "timeout",
        "timed out",
        "failure",
        "failed",
        "crash",
        "exception",
        "unavailable",
        "cannot recover",
        "system error",
        "service down",
    )
    vague_terms = ("fix it", "help me", "this issue", "that issue", "it broke")

    if any(term in normalized for term in risky_terms):
        route = "risky"
        rationale = "The ticket requests a side-effecting support action that needs approval."
        needs_approval = True
        confidence = 0.94
    elif any(term in normalized for term in tool_terms):
        route = "tool"
        rationale = "The ticket asks for a lookup or retrieval from support systems."
        needs_approval = False
        confidence = 0.9
    elif any(term in normalized for term in error_terms):
        route = "error"
        rationale = "The ticket describes a system failure requiring retry/fallback handling."
        needs_approval = False
        confidence = 0.88
    elif _looks_vague(normalized, vague_terms):
        route = "missing_info"
        rationale = "The ticket is too vague to act on safely."
        needs_approval = False
        confidence = 0.86
    else:
        route = "simple"
        rationale = "The ticket is a general support question answerable without tools."
        needs_approval = False
        confidence = 0.78

    return {
        "route": route,
        "confidence": confidence,
        "rationale": rationale,
        "needs_approval": needs_approval,
    }


def _looks_vague(normalized: str, vague_terms: tuple[str, ...]) -> bool:
    words = re.findall(r"[a-z0-9]+", normalized)
    return len(words) <= 4 or any(term in normalized for term in vague_terms)


def _judge_tool_result(latest_result: str) -> dict[str, Any]:
    try:
        payload = json.loads(latest_result)
    except json.JSONDecodeError:
        payload = None

    if isinstance(payload, dict):
        status = str(payload.get("status", "")).lower()
        error = payload.get("error")
        failed = (
            payload.get("success") is False
            or status in {"error", "failed", "blocked", "missing", "unavailable"}
            or bool(error)
        )
    else:
        lowered = latest_result.lower()
        failed = any(marker in lowered for marker in ("error", "failed", "timeout", "unavailable"))
    return {
        "is_sufficient": not failed,
        "score": 0.15 if failed else 0.92,
        "reason": (
            "Latest tool result is usable."
            if not failed
            else "Latest tool result indicates failure."
        ),
    }


def _draft_grounded_response(prompt: str) -> str:
    route = _extract_field(prompt, "ROUTE") or "simple"
    query = _extract_between(prompt, "SUPPORT_TICKET", "END_TICKET")
    tool_context = _extract_between(prompt, "TOOL_CONTEXT", "END_TOOL_CONTEXT")
    approval_context = _extract_between(prompt, "APPROVAL_CONTEXT", "END_APPROVAL_CONTEXT")
    proposed_action = _extract_between(prompt, "PROPOSED_ACTION", "END_PROPOSED_ACTION")

    if route == "missing_info":
        return (
            "Could you share the affected product, account/order ID, and what result "
            "you expected?"
        )
    if route == "risky":
        if "approved" in approval_context.lower():
            return (
                "Approval is recorded. I have prepared the requested support action for the "
                "approved workflow and will keep the audit trail attached."
            )
        return (
            "I cannot perform that side-effecting action yet. A human approval is required before "
            f"proceeding with: {proposed_action or query}."
        )
    if route == "tool":
        return f"Based on the support lookup: {tool_context}"
    if route == "error":
        if "success" in tool_context.lower():
            return f"The transient failure recovered. Latest diagnostic result: {tool_context}"
        return "The request hit a system failure. I could not complete it after the retry policy."
    return (
        "You can reset your password from the sign-in page by choosing 'Forgot password', "
        "following the email link, and setting a new password."
    )


def _extract_field(text: str, field_name: str) -> str | None:
    pattern = rf"^{re.escape(field_name)}:\s*(.+)$"
    match = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
    if not match:
        return None
    return match.group(1).strip()
