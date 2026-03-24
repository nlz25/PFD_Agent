"""Shared memory utilities for reading and writing MEMORY.md.

Import this module from any sub-agent that needs to access or update
persistent session memory stored in skills/MEMORY.md.
"""

from __future__ import annotations

import os
from ..workspace import WORKSPACE_ROOT
_MEMORY_PATH = WORKSPACE_ROOT / "MEMORY.md"

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

    Args:
        new_entries: The new Markdown-formatted entries to append.
            Include a blank line before each new entry. 
            Example: "- Always check available datasets and evaluate existing pre-trained models before launching full MLFF workflows."

    Returns:
        Confirmation message.
    """
    return write_memory("\n" + new_entries)