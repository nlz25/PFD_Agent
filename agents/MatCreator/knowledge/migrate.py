"""One-time migration: reads MEMORY.md and populates the knowledge graph."""

from __future__ import annotations

import logging
import re

from .graph_store import KnowledgeGraph  # noqa: F401 (kept for type hints)
from .query import _get_memory_kg

logger = logging.getLogger(__name__)


def migrate_memory_md(memory_path: str | None = None) -> dict:
    """Migrate entries from MEMORY.md into the knowledge graph.

    Each bullet line in MEMORY.md is treated as an Insight node. The function
    is idempotent — re-running it skips near-duplicate entries via fuzzy matching.

    Args:
        memory_path: Absolute path to MEMORY.md. Defaults to workspace MEMORY.md.

    Returns:
        Dict with keys: status, nodes_created, skipped, message.
    """
    from ..workspace import WORKSPACE_ROOT

    if memory_path is None:
        path = WORKSPACE_ROOT / "MEMORY.md"
    else:
        from pathlib import Path
        path = Path(memory_path)

    if not path.exists():
        return {
            "status": "skipped",
            "message": f"MEMORY.md not found at {path}",
            "nodes_created": 0,
            "skipped": 0,
        }

    text = path.read_text(encoding="utf-8")
    # Extract bullet list items (- text or * text) and standalone non-empty lines
    entries: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        # Strip leading bullet markers
        cleaned = re.sub(r"^[-*•]\s+", "", stripped)
        if len(cleaned) >= 10:
            entries.append(cleaned)

    kg = _get_memory_kg()
    created = 0
    skipped = 0

    for entry in entries:
        name = entry[:80].rstrip(".")
        node = kg.upsert_node(
            category="memory",
            name=name,
            description=entry,
            content={"raw": entry, "source": "MEMORY.md"},
            source_session="migration",
            confidence=0.9,
        )
        # If the node was just created (source_session will be "migration")
        if node.source_session == "migration" and not node.reference_count:
            created += 1
        else:
            skipped += 1

    logger.info("Migration: %d created, %d skipped", created, skipped)
    return {
        "status": "ok",
        "nodes_created": created,
        "skipped": skipped,
        "message": (
            f"Migrated MEMORY.md: {created} new Insight nodes, "
            f"{skipped} skipped (near-duplicates)."
        ),
    }
