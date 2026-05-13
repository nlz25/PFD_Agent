"""Semantic + BFS retrieval for the knowledge graph."""

from __future__ import annotations

import logging
import os
from collections import deque
from datetime import datetime, timezone
from typing import Optional

import numpy as np

from .graph_store import KnowledgeGraph

logger = logging.getLogger(__name__)

_kg: Optional[KnowledgeGraph] = None

# Embedding model — set EMBEDDING_MODEL env var to match your provider, e.g.:
#   OpenAI:     text-embedding-3-small
#   DashScope:  text-embedding-v3
#   Ollama:     ollama/nomic-embed-text
# Defaults to the chat LLM model as a best-effort fallback (works for some providers).
_DEFAULT_EMBEDDING_MODEL = None  # resolved at call time from LLM_MODEL


def _get_kg() -> KnowledgeGraph:
    global _kg
    if _kg is None:
        _kg = KnowledgeGraph()
    return _kg


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

def _embedding_model() -> str:
    explicit = os.environ.get("EMBEDDING_MODEL", "").strip()
    if explicit:
        return explicit
    # Fall back to the chat model — works for providers that expose both
    # chat and embedding under the same model name (e.g. some Ollama models).
    from ..constants import LLM_MODEL
    return os.environ.get("LLM_MODEL", LLM_MODEL)


def _embed_texts(texts: list[str]) -> list[list[float]] | None:
    """Call litellm embedding API for a batch of texts. Returns None on failure."""
    try:
        from litellm import embedding as litellm_embedding
        from ..constants import LLM_API_KEY, LLM_BASE_URL

        api_key  = os.environ.get("LLM_API_KEY", LLM_API_KEY)
        base_url = os.environ.get("LLM_BASE_URL", LLM_BASE_URL) or None

        response = litellm_embedding(
            model=_embedding_model(),
            input=texts,
            api_key=api_key,
            api_base=base_url,
            encoding_format="float",
        )
        return [item["embedding"] for item in response.data]
    except Exception as exc:
        logger.warning(
            "Embedding API unavailable (model=%s, error=%s): falling back to token search",
            _embedding_model(), exc,
        )
        return None


def _embed_one(text: str) -> list[float] | None:
    result = _embed_texts([text])
    return result[0] if result else None


def _node_text(name: str, description: str) -> str:
    """Text used for embedding — name carries more weight via repetition."""
    return f"{name}. {name}. {description}".strip()


# ---------------------------------------------------------------------------
# Cosine similarity
# ---------------------------------------------------------------------------

def _cosine_sim(a: list[float], b: list[float]) -> float:
    va, vb = np.array(a, dtype=np.float32), np.array(b, dtype=np.float32)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    return float(np.dot(va, vb) / denom) if denom > 0 else 0.0


def _top_k_semantic(
    query_vec: list[float],
    embeddings: list[tuple[str, list[float]]],
    top_k: int,
) -> list[str]:
    """Return node IDs sorted by cosine similarity to query_vec."""
    scored = [
        (node_id, _cosine_sim(query_vec, emb))
        for node_id, emb in embeddings
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [nid for nid, _ in scored[:top_k]]


# ---------------------------------------------------------------------------
# Lazy backfill
# ---------------------------------------------------------------------------

def _backfill_embeddings(kg: KnowledgeGraph) -> None:
    """Compute and store embeddings for any nodes that are missing one."""
    missing = kg.get_nodes_without_embeddings()
    if not missing:
        return

    logger.info("Backfilling embeddings for %d nodes…", len(missing))
    texts = [_node_text(n.name, n.description or "") for n in missing]
    vecs = _embed_texts(texts)
    if vecs is None:
        return

    for node, vec in zip(missing, vecs):
        kg.set_embedding(node.id, vec)
    logger.info("Backfill complete.")


# ---------------------------------------------------------------------------
# Seed finding — semantic primary, LIKE fallback
# ---------------------------------------------------------------------------

def _find_seeds(kg: KnowledgeGraph, query: str, top_k: int = 10) -> list[str]:
    """Return seed node IDs using semantic search with LIKE as fallback."""
    _backfill_embeddings(kg)
    query_vec = _embed_one(_node_text(query, ""))
    if query_vec is not None:
        all_embeddings = kg.get_all_embeddings()
        if all_embeddings:
            return _top_k_semantic(query_vec, all_embeddings, top_k)

    # Fallback: substring match on name
    logger.debug("Using LIKE fallback for query '%s'", query)
    nodes = kg.find_nodes_by_name(query)
    return [n.id for n in nodes[:top_k]]


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------

def _recency_decay(created_at_iso: str) -> float:
    try:
        dt = datetime.fromisoformat(created_at_iso)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        days = (datetime.now(timezone.utc) - dt).days
        return 1.0 / (1.0 + days)
    except Exception:
        return 0.5


def _rank_score(node_data: dict) -> float:
    rc    = node_data.get("reference_count", 0)
    conf  = node_data.get("confidence", 1.0)
    decay = _recency_decay(node_data.get("created_at", ""))
    return (1 + rc) * conf * decay


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def _format_nodes(kg: KnowledgeGraph, node_ids: list[str]) -> str:
    by_type: dict[str, list[str]] = {}
    for nid in node_ids:
        node = kg.get_node(nid)
        if not node:
            continue
        line = f"  - **{node.name}**"
        if node.description:
            line += f": {node.description}"
        by_type.setdefault(node.type, []).append(line)

    if not by_type:
        return "No relevant knowledge found."

    sections = []
    for t in ["Insight", "Workflow", "Skill", "Concept", "Material", "Result"]:
        if t in by_type:
            sections.append(f"### {t}s\n" + "\n".join(by_type[t]))
    for t, lines in by_type.items():
        if t not in {"Insight", "Workflow", "Skill", "Concept", "Material", "Result"}:
            sections.append(f"### {t}s\n" + "\n".join(lines))
    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def query_knowledge_graph(
    query: str,
    types: Optional[list] = None,
    depth: int = 2,
    top_k: int = 15,
) -> str:
    """Query the knowledge graph for concepts, skills, and insights relevant to *query*.

    Uses embedding-based semantic search to find seed nodes, then expands via BFS.
    Falls back to substring matching if the embedding API is unavailable.

    Call this at the start of planning to recall relevant past knowledge instead of
    loading the entire memory file.

    Args:
        query:  Free-text search string. Does not need to match node names exactly.
        types:  Optional list of node types to filter by
                (Concept, Skill, Material, Result, Insight, Workflow).
        depth:  BFS expansion depth from seed nodes (default 2).
        top_k:  Maximum number of nodes to return (default 15).

    Returns:
        Markdown-formatted knowledge context grouped by node type.
    """
    kg = _get_kg()
    try:
        import networkx as nx
        seed_ids = _find_seeds(kg, query, top_k=top_k)
        if not seed_ids:
            return f"No knowledge graph entries found for '{query}'."

        G = kg.load_networkx()

        # BFS (bidirectional) from all seeds
        visited: set[str] = set()
        queue: deque[tuple[str, int]] = deque()
        for sid in seed_ids:
            if sid in G:
                visited.add(sid)
                queue.append((sid, 0))

        while queue:
            nid, d = queue.popleft()
            if d >= depth:
                continue
            for neighbor in list(G.successors(nid)) + list(G.predecessors(nid)):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, d + 1))

        if types:
            visited = {nid for nid in visited if G.nodes[nid].get("type") in types}

        ranked = sorted(
            visited,
            key=lambda nid: _rank_score(G.nodes[nid]),
            reverse=True,
        )[:top_k]

        for nid in ranked:
            kg.increment_reference(nid)

        return _format_nodes(kg, ranked)

    except Exception as exc:
        logger.warning("query_knowledge_graph failed: %s", exc)
        return f"Knowledge graph query failed: {exc}"


def save_to_knowledge_graph(content: str, context: str = "") -> str:
    """Save a plain-text lesson or finding as an Insight node.

    Use this to persist an important finding during a session without waiting
    for the post-session extractor. An embedding is computed immediately so the
    node is searchable right away.

    Args:
        content: The lesson or fact to save (plain text, 1-3 sentences).
        context: Optional context hint (e.g. current skill name or goal).

    Returns:
        Confirmation message with the created node name.
    """
    kg = _get_kg()
    try:
        name = content[:80].strip().rstrip(".")
        node = kg.upsert_node(
            type="Insight",
            name=name,
            description=content,
            content={"raw": content, "context": context},
        )
        vec = _embed_one(_node_text(node.name, content))
        if vec:
            kg.set_embedding(node.id, vec)
        return f"Saved to knowledge graph as Insight: '{node.name}' (id={node.id})"
    except Exception as exc:
        logger.warning("save_to_knowledge_graph failed: %s", exc)
        return f"Failed to save to knowledge graph: {exc}"
