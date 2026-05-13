"""Persistent cross-session state for the knowledge pipeline."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from ..constants import _ADK_DIR

logger = logging.getLogger(__name__)

_STATE_PATH = _ADK_DIR / ".kg_state.json"


def _load() -> dict:
    if _STATE_PATH.exists():
        try:
            return json.loads(_STATE_PATH.read_text())
        except Exception:
            pass
    return {}


def _save(state: dict) -> None:
    _STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _STATE_PATH.write_text(json.dumps(state, indent=2))


def increment_exec_count() -> int:
    """Increment and persist the global execution-completion counter. Returns new value."""
    state = _load()
    count = state.get("exec_completion_count", 0) + 1
    state["exec_completion_count"] = count
    _save(state)
    return count


def get_exec_count() -> int:
    return _load().get("exec_completion_count", 0)


def record_synthesizer_run() -> None:
    state = _load()
    state["last_synthesizer_run"] = datetime.now(timezone.utc).isoformat()
    _save(state)
