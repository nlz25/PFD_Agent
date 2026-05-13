"""Memory utilities for the thinking agent.

Exposes knowledge-graph-based tools (preferred) and legacy MEMORY.md helpers
(kept for backward compatibility and manual use).
"""

from __future__ import annotations

import os
from ...workspace import WORKSPACE_ROOT

_MEMORY_PATH = WORKSPACE_ROOT / "MEMORY.md"


# ---------------------------------------------------------------------------
# Knowledge graph tools (preferred)
# ---------------------------------------------------------------------------

from ...knowledge.query import query_knowledge_graph, save_to_knowledge_graph  # noqa: F401


# ---------------------------------------------------------------------------
# Legacy MEMORY.md helpers (backward-compatible)
# ---------------------------------------------------------------------------

def load_memory() -> str:
    """Return the full contents of MEMORY.md, or an empty string if missing."""
    try:
        with open(_MEMORY_PATH, "r") as f:
            return f.read()
    except FileNotFoundError:
        return ""


def write_memory(content: str) -> str:
    """Append *content* to MEMORY.md and return a confirmation message."""
    os.makedirs(os.path.dirname(_MEMORY_PATH), exist_ok=True)
    with open(_MEMORY_PATH, "a") as f:
        f.write(content)
    return f"Memory appended successfully at {_MEMORY_PATH}"


def update_memory(new_entries: str) -> str:
    """Append new_entries to MEMORY.md.

    Prefer save_to_knowledge_graph for new knowledge. This function is kept
    for manual/legacy use.
    """
    return write_memory("\n" + new_entries)


def read_memory() -> str:
    """Read the full contents of MEMORY.md.

    Prefer query_knowledge_graph for targeted retrieval. This function loads
    the entire file and should be used only when a broad context dump is needed.
    """
    content = load_memory()
    if not content.strip():
        return "Memory is empty. No past context available."
    return content
