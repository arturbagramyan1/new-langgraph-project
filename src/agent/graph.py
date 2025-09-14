"""Define the LangGraph agent graph.

This graph uses a standard messages-based state. It optionally calls an LLM
(GPT-5) when an ``OPENAI_API_KEY`` is configured; otherwise, it echoes input.
"""

from __future__ import annotations

import os
from typing import Annotated, TypedDict, Any, Iterable

from langchain_core.messages import HumanMessage, BaseMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.pregel import Pregel

try:
    # OpenAI >= 1.x provides an async client
    from openai import AsyncOpenAI  # type: ignore
except Exception:  # pragma: no cover - dependency might not be present in some envs
    AsyncOpenAI = None  # type: ignore


class AgentState(TypedDict):
    """Graph state with messages history."""

    messages: HumanMessage | list[HumanMessage]


def _as_openai_message(msg: Any) -> dict:
    """Convert a message-like object to OpenAI chat format dict."""

    role_map = {
        "human": "user",
        "ai": "assistant",
        "system": "system",
        "developer": "developer",
    }

    if isinstance(msg, BaseMessage):
        raw_type = getattr(msg, "type", None)
        role_key = str(raw_type) if raw_type is not None else "user"
        role = role_map.get(role_key, role_key)
        content = getattr(msg, "content", "")
        return {"role": role, "content": str(content)}
    if isinstance(msg, dict):
        raw_role = msg.get("role") or msg.get("type")
        role_key = str(raw_role) if raw_role is not None else "user"
        role = role_map.get(role_key, role_key)
        content = msg.get("content", "")
        return {"role": role, "content": str(content)}
    if isinstance(msg, (str, bytes)):
        return {
            "role": "user",
            "content": msg.decode() if isinstance(msg, bytes) else msg,
        }
    # Fallback: best-effort extraction
    role = getattr(msg, "role", None) or getattr(msg, "type", None) or "user"
    role = role_map.get(str(role), str(role))
    content = getattr(msg, "content", "")
    return {"role": str(role), "content": str(content)}


def _to_openai_messages(messages_in: Any) -> list[dict]:
    """Ensure we have a list of OpenAI-style messages."""

    if messages_in is None:
        return []
    if isinstance(messages_in, (BaseMessage, dict)) or isinstance(
        messages_in, (str, bytes)
    ):
        return [_as_openai_message(messages_in)]
    if isinstance(messages_in, Iterable):
        return [_as_openai_message(m) for m in messages_in]
    return [_as_openai_message(messages_in)]


async def model_or_echo(state: AgentState) -> AgentState:
    """Call GPT-5 if available; otherwise, echo user content.

    When ``messages`` is missing, derive a single-turn user message from
    ``prompt`` or ``changeme`` for convenience.
    """

    messages = state.get("messages")
    if not messages:
        # Derive a single user message from other common fields
        derived = str(
            (state.get("prompt") if isinstance(state, dict) else None)
            or (state.get("changeme") if isinstance(state, dict) else None)
            or ""
        )
        messages = HumanMessage(content=derived)

    api_key_present = bool(os.getenv("OPENAI_API_KEY"))
    model = os.getenv("OPENAI_MODEL", "gpt-5")

    if api_key_present and AsyncOpenAI is not None:
        try:
            client = AsyncOpenAI()
            completion = await client.chat.completions.create(
                model=model,
                messages=_to_openai_messages(messages),
            )
            content = completion.choices[0].message.content or ""
            return {"messages": [HumanMessage(content=content)]}
        except Exception as exc:  # fall back to echo with error note
            return {"messages": [HumanMessage(content=f"[error calling model: {exc}]")]}

    # Fallback echo: mirror the last user content or blank
    normalized = _to_openai_messages(messages)
    last_user = next((m for m in reversed(normalized) if m.get("role") == "user"), None)
    echoed = (last_user or {}).get("content", "")
    return {"messages": [HumanMessage(content=str(echoed))]}


_builder = StateGraph(AgentState)
_builder.add_node("model_or_echo", model_or_echo)
_builder.add_edge(START, "model_or_echo")
_builder.add_edge("model_or_echo", END)

# Compiled graph exported as `graph`
graph: Pregel = _builder.compile()
