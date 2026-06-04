"""Knowledge extractor: processes a session trajectory JSONL and populates the graph."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from ..workspace import WORKSPACE_ROOT
from .graph_store import KnowledgeGraph  # noqa: F401 (kept for type hints)
from .query import _get_memory_kg

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LLM-based extraction
# ---------------------------------------------------------------------------

_EXTRACTION_PROMPT = """You are a knowledge extraction assistant.

Given a session trajectory (per-step results) and an optional session-level summary
(global narrative, decisions, failures, lessons), extract knowledge graph entries.

Return a JSON array of objects. Each object must have:
  "name":        short canonical name (≤80 chars)
  "description": one sentence describing it
  "relations":   list of {{source_name, edge_type, target_name}} objects
                 edge_type must be one of: depends_on | belongs_to | relates_to

All extracted nodes are memory nodes (lessons, findings, heuristics).
Relations may reference skill node names (existing tool/guide names) as targets.

Rules:
- Extract 5-20 entries total — quality over quantity.
- Focus on lessons learned, heuristics, warnings, quantitative results.
- Prefer extracting from the session summary (lessons_learned, failed_attempts)
  and per-step results from the trajectory.
- Only emit relations where source appears in the entry list or is a known skill name.
- Output ONLY the JSON array, no prose.

Session summary (global narrative):
{session_summary}

Trajectory (per-step results):
{trajectory}
"""


def _call_llm(prompt: str) -> str:
    """Call the configured LLM. Falls back gracefully if unavailable."""
    try:
        from litellm import completion
        from ..constants import LLM_MODEL, LLM_API_KEY, LLM_BASE_URL

        model = os.environ.get("LLM_MODEL", LLM_MODEL)
        api_key = os.environ.get("LLM_API_KEY", LLM_API_KEY)
        base_url = os.environ.get("LLM_BASE_URL", LLM_BASE_URL)

        response = completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            api_key=api_key,
            base_url=base_url,
            temperature=0.0,
            max_tokens=2048,
        )
        return response.choices[0].message.content or ""
    except Exception as exc:
        logger.warning("LLM call failed in extractor: %s", exc)
        return ""


def _parse_extraction(raw: str) -> list[dict[str, Any]]:
    """Parse JSON array from LLM output, tolerating markdown fences."""
    import re
    raw = raw.strip()
    m = re.search(r"\[[\s\S]*\]", raw)
    if not m:
        return []
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return []


def _read_trajectory(session_id: str) -> str:
    """Return concatenated summaries from a session's JSONL trajectory."""
    traj_path = WORKSPACE_ROOT / "trajectories" / f"{session_id}.jsonl"
    if not traj_path.exists():
        return ""
    lines = []
    with traj_path.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                entry = json.loads(line)
                skill = entry.get("active_skill") or "unknown"
                summary = entry.get("concise_summary") or entry.get("key_results") or ""
                if summary:
                    lines.append(f"[Step {entry.get('step_index', '?')} | {skill}] {summary}")
            except json.JSONDecodeError:
                continue
    return "\n".join(lines)


def _read_session_summary(session_id: str) -> str:
    """Return a formatted session-level summary if one was written by the thinking agent."""
    summary_path = WORKSPACE_ROOT / "trajectories" / f"{session_id}_summary.json"
    if not summary_path.exists():
        return "(No session summary available)"
    try:
        data = json.loads(summary_path.read_text())
        parts = [f"Goal: {data.get('goal', '')}"]
        if data.get("approach"):
            parts.append(f"Approach: {data['approach']}")
        if data.get("outcome"):
            parts.append(f"Outcome: {data['outcome']}")
        if data.get("key_decisions"):
            parts.append("Key decisions:\n" + "\n".join(f"  - {d}" for d in data["key_decisions"]))
        if data.get("lessons_learned"):
            parts.append("Lessons learned:\n" + "\n".join(f"  - {l}" for l in data["lessons_learned"]))
        if data.get("failed_attempts"):
            parts.append("Failed attempts:\n" + "\n".join(f"  - {f}" for f in data["failed_attempts"]))
        return "\n".join(parts)
    except Exception as exc:
        logger.warning("Failed to read session summary for %s: %s", session_id, exc)
        return "(Session summary unreadable)"


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_knowledge_extractor(session_id: str) -> dict:
    """Extract knowledge from a completed session and upsert into the graph.

    Reads both the per-step trajectory JSONL and the session-level summary JSON
    (written by the thinking agent via write_session_summary). The session summary
    provides global narrative, key decisions, and failures that step-level entries miss.

    Args:
        session_id: The ADK session ID whose trajectory JSONL should be processed.

    Returns:
        Dict with keys: status, nodes_created, edges_created, message.
    """
    trajectory = _read_trajectory(session_id)
    if not trajectory.strip():
        return {
            "status": "skipped",
            "message": f"No trajectory found for session {session_id}",
            "nodes_created": 0,
            "edges_created": 0,
        }

    session_summary = _read_session_summary(session_id)
    prompt = _EXTRACTION_PROMPT.format(
        session_summary=session_summary,
        trajectory=trajectory,
    )
    raw = _call_llm(prompt)
    entries = _parse_extraction(raw)

    if not entries:
        logger.info("Extractor: no entries parsed for session %s", session_id)
        return {
            "status": "ok",
            "message": "LLM returned no extractable entries.",
            "nodes_created": 0,
            "edges_created": 0,
        }

    kg = _get_memory_kg()
    node_map: dict[str, str] = {}  # name → node id
    nodes_created = 0
    edges_created = 0
    new_nodes: list = []  # collect newly inserted nodes for batch embedding

    # First pass: upsert all nodes as memory category
    for entry in entries:
        nname = entry.get("name", "").strip()
        ndesc = entry.get("description", "")
        if not nname:
            continue
        node = kg.upsert_node(
            category="memory",
            name=nname,
            description=ndesc,
            source_session=session_id,
        )
        if node.source_session == session_id:
            nodes_created += 1
            if node.embedding is None:
                new_nodes.append(node)
        node_map[nname] = node.id

    # Batch-compute embeddings for newly inserted nodes
    if new_nodes:
        try:
            from .query import _embed_texts, _node_text
            texts = [_node_text(n.name, n.description or "") for n in new_nodes]
            vecs = _embed_texts(texts)
            if vecs:
                for node, vec in zip(new_nodes, vecs):
                    kg.set_embedding(node.id, vec)
        except Exception as emb_exc:
            logger.warning("Embedding computation failed in extractor: %s", emb_exc)

    # Second pass: upsert edges
    for entry in entries:
        for rel in entry.get("relations", []):
            src_name = rel.get("source_name", "")
            tgt_name = rel.get("target_name", "")
            etype    = rel.get("edge_type", "")
            if src_name in node_map and tgt_name in node_map:
                kg.upsert_edge(
                    source_id=node_map[src_name],
                    target_id=node_map[tgt_name],
                    edge_type=etype,
                )
                edges_created += 1

    logger.info(
        "Extractor session=%s: %d nodes, %d edges", session_id, nodes_created, edges_created
    )
    return {
        "status": "ok",
        "nodes_created": nodes_created,
        "edges_created": edges_created,
        "message": f"Extracted {nodes_created} nodes and {edges_created} edges from session {session_id}.",
    }
