"""MatCreator agent configuration helpers."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

_script_dir = Path(__file__).parent.resolve()
load_dotenv(_script_dir / ".env", override=True)

LLM_MODEL: str = os.environ.get("LLM_MODEL", "")
LLM_API_KEY: str = os.environ.get("LLM_API_KEY", "")
LLM_BASE_URL: str = os.environ.get("LLM_BASE_URL", "")
BOHRIUM_USERNAME: str = os.environ.get("BOHRIUM_USERNAME", "")
BOHRIUM_PASSWORD: str = os.environ.get("BOHRIUM_PASSWORD", "")
BOHRIUM_PROJECT_ID: int | str = os.environ.get("BOHRIUM_PROJECT_ID", 00000)
EXECUTION_ENABLE_WITHIN_INVOCATION_COMPACTION: str|int = os.environ.get("EXECUTION_ENABLE_WITHIN_INVOCATION_COMPACTION", 1)
EXECUTION_COMPACT_KEEP_TAIL: int = int(os.environ.get("EXECUTION_COMPACT_KEEP_TAIL", "10"))
EXECUTION_COMPACT_EVERY_EVENTS: int = int(os.environ.get("EXECUTION_COMPACT_EVERY_EVENTS", "5"))


_AGENT_PATH = _script_dir
_ADK_DIR = _script_dir / ".adk"          # ADK internal storage (session.db, etc.)
_KNOWLEDGE_PATH= _script_dir / "knowledge"
_SKILLS_DIR = _script_dir / "skills"
_GUIDES_DIR = _script_dir/ "guides"
_MEMORY_PATH = _KNOWLEDGE_PATH /"MEMORY.md"

# Knowledge graph databases (split by lifecycle and ownership)
SKILL_GRAPH_DB  = _ADK_DIR / "skill_graph.db"   # dev-maintained, immutable skill/guide nodes
MEMORY_GRAPH_DB = _ADK_DIR / "memory_graph.db"  # user-generated memory nodes from sessions
# Legacy single-DB path (knowledge_graph.db) is no longer used.

# Workspace paths — resolved lazily at runtime via workspace.get_workspace_root()
# These are re-exported here for convenience so other modules only need one import.
def _workspace_root() -> "Path":
    from .workspace import get_workspace_root
    return get_workspace_root()

def _workspace_skills_dir() -> "Path":
    return _workspace_root() / "skills"

def _workspace_guides_dir() -> "Path":
    return _workspace_root() / "guides"

def _workspace_memory_path() -> "Path":
    return _workspace_root() / "MEMORY.md"