"""In-memory per-session conversation state for the /chat endpoints.

Sessions are held in a plain process-local dict - state is lost on restart and
is not shared across worker processes. That's acceptable for now; swap this
module for a persistent/shared store if the app moves beyond a single process.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ChatSession:
    history: list[dict[str, str]] = field(default_factory=list)
    variables: dict[str, Any] = field(default_factory=dict)


_SESSIONS: dict[str, ChatSession] = {}


def get_session(session_id: str) -> ChatSession:
    return _SESSIONS.setdefault(session_id, ChatSession())


def reset_sessions() -> None:
    """Test helper: clear all in-memory session state."""
    _SESSIONS.clear()
