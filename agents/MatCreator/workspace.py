"""Workspace management for MatClaw — project-local overlay of skills/guides/memory.

The workspace root is resolved in this order:
1. ``MATCLAW_WORKSPACE`` environment variable (absolute or relative to CWD)
2. ``~/.workspace/`` in the user's home directory

On first use, call :func:`init_workspace` to create the directory tree and
copy the built-in default knowledge into it so the user has a starting point.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from .constants import _KNOWLEDGE_PATH

# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def get_workspace_root() -> Path:
    """Return the resolved workspace root path (not guaranteed to exist)."""
    env_val = os.environ.get("MATCLAW_WORKSPACE", "")
    if env_val:
        return Path(env_val).expanduser().resolve()
    return (Path(__file__).parent / ".workspace").resolve()


def workspace_skills_dir() -> Path:
    return get_workspace_root() / "skills"


def workspace_guides_dir() -> Path:
    return get_workspace_root() / "guides"


def workspace_memory_path() -> Path:
    return get_workspace_root() / "MEMORY.md"


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

def init_workspace(force: bool = False) -> str:
    """Create the workspace directory tree and seed it with built-in defaults.

    If the workspace already exists this is a no-op unless *force* is True,
    in which case built-in files that are missing from the workspace are
    re-copied (existing workspace files are never overwritten).

    Returns a human-readable status message.
    """
    root = get_workspace_root()
    skills_dir = workspace_skills_dir()
    guides_dir = workspace_guides_dir()

    skills_dir.mkdir(parents=True, exist_ok=True)
    guides_dir.mkdir(parents=True, exist_ok=True)

    copied: list[str] = []

    # Copy built-in flat skills → workspace flat skills (do not overwrite)
    builtin_skills = _KNOWLEDGE_PATH / "skills"
    for src in sorted(builtin_skills.glob("*.md")):
        dst = skills_dir / src.name
        if not dst.exists() or force:
            shutil.copy2(src, dst)
            copied.append(f"skills/{src.name}")

    # Copy built-in guides → workspace guides (do not overwrite)
    builtin_guides = _KNOWLEDGE_PATH / "guides"
    for src in sorted(builtin_guides.glob("*.md")):
        dst = guides_dir / src.name
        if not dst.exists() or force:
            shutil.copy2(src, dst)
            copied.append(f"guides/{src.name}")

    # Seed MEMORY.md
    mem = workspace_memory_path()
    if not mem.exists():
        mem.write_text("# Workspace Memory\n\n")
        copied.append("MEMORY.md")

    if copied:
        return f"Workspace initialised at {root}\nSeeded: {', '.join(copied)}"
    return f"Workspace already exists at {root} (nothing overwritten)."
