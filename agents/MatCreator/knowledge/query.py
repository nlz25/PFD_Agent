"""Semantic + BFS retrieval for the knowledge graph."""

from __future__ import annotations

import logging
import os
from collections import deque
from datetime import datetime, timezone
from typing import Optional

import numpy as np

from .graph_store import KnowledgeGraph
from .schema import KgNode

logger = logging.getLogger(__name__)

_skill_kg:  Optional[KnowledgeGraph] = None
_memory_kg: Optional[KnowledgeGraph] = None

# Embedding model — set EMBEDDING_MODEL env var to match your provider, e.g.:
#   OpenAI:     text-embedding-3-small
#   DashScope:  text-embedding-v3
#   Ollama:     ollama/nomic-embed-text
# Defaults to the chat LLM model as a best-effort fallback (works for some providers).
_DEFAULT_EMBEDDING_MODEL = None  # resolved at call time from LLM_MODEL


def _get_skill_kg() -> KnowledgeGraph:
    global _skill_kg
    if _skill_kg is None:
        from ..constants import SKILL_GRAPH_DB
        _skill_kg = KnowledgeGraph(db_path=SKILL_GRAPH_DB)
    return _skill_kg


def _get_memory_kg() -> KnowledgeGraph:
    global _memory_kg
    if _memory_kg is None:
        from ..constants import MEMORY_GRAPH_DB
        _memory_kg = KnowledgeGraph(db_path=MEMORY_GRAPH_DB)
    return _memory_kg


# Backward-compat alias used by skill.py and other internal callers
_get_kg = _get_skill_kg


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

def _find_seeds(kg: KnowledgeGraph, query: str, top_k: int = 10, category: Optional[str] = None) -> list[str]:
    """Return seed node IDs using semantic search with LIKE as fallback."""
    _backfill_embeddings(kg)
    query_vec = _embed_one(_node_text(query, ""))
    if query_vec is not None:
        all_embeddings = kg.get_all_embeddings()
        if all_embeddings:
            if category:
                # Filter embeddings to only the requested category
                with kg._Session() as sess:
                    from sqlalchemy import select as sa_select
                    cat_ids = {
                        row.id for row in sess.execute(
                            sa_select(KgNode.id).where(KgNode.category == category)
                        ).all()
                    }
                all_embeddings = [(nid, emb) for nid, emb in all_embeddings if nid in cat_ids]
            return _top_k_semantic(query_vec, all_embeddings, top_k)

    # Fallback: substring match on name
    logger.debug("Using LIKE fallback for query '%s'", query)
    nodes = kg.find_nodes_by_name(query, category=category)
    return [n.id for n in nodes[:top_k]]


def _find_node_id_by_name(kg: KnowledgeGraph, name: str, category: str = "skill") -> str | None:
    """Resolve a node name to its ID: exact → case-insensitive → substring match."""
    with kg._Session() as sess:
        from sqlalchemy import select as sa_select
        for clause in [
            KgNode.name == name,
            KgNode.name.ilike(name),
            KgNode.name.ilike(f"%{name}%"),
        ]:
            node = sess.execute(
                sa_select(KgNode).where(clause, KgNode.category == category)
            ).scalars().first()
            if node:
                return node.id
    return None


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

_CAT_HINT = {
    "skill": "*(use `load_skill` to load full instructions)*",
    "memory": "*(past findings/lessons — read directly, do NOT use `load_skill`)*",
}


def _format_nodes(kg: KnowledgeGraph, node_ids: list[str]) -> str:
    by_cat: dict[str, list[str]] = {}
    for nid in node_ids:
        node = kg.get_node(nid)
        if not node:
            continue
        cat = node.category or "memory"
        line = f"  - **{node.name}**"
        if node.description:
            line += f": {node.description}"
        by_cat.setdefault(cat, []).append(line)

    if not by_cat:
        return "No relevant knowledge found."

    sections = []
    for cat in ("skill", "memory"):
        if cat in by_cat:
            hint = _CAT_HINT.get(cat, "")
            sections.append(f"### {cat.capitalize()}\n{hint}\n" + "\n".join(by_cat[cat]))
    return "\n\n".join(sections)


def _format_nodes_multi(nodes: list[tuple[KnowledgeGraph, str]]) -> str:
    """Format nodes fetched from potentially different graph instances."""
    by_cat: dict[str, list[str]] = {}
    for kg, nid in nodes:
        node = kg.get_node(nid)
        if not node:
            continue
        cat = node.category or "memory"
        line = f"  - **{node.name}**"
        if node.description:
            line += f": {node.description}"
        by_cat.setdefault(cat, []).append(line)

    if not by_cat:
        return "No relevant knowledge found."

    sections = []
    for cat in ("skill", "memory"):
        if cat in by_cat:
            hint = _CAT_HINT.get(cat, "")
            sections.append(f"### {cat.capitalize()} nodes\n{hint}\n" + "\n".join(by_cat[cat]))
    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def query_knowledge_graph(
    query: str,
    depth: int = 2,
    top_k: int = 15,
) -> str:
    """Query the memory knowledge graph for lessons and past findings relevant to *query*.

    Uses embedding-based semantic search to find seed nodes, then expands via BFS.
    Falls back to substring matching if the embedding API is unavailable.

    Call this at the start of planning to recall relevant past knowledge.
    Memory nodes may reference skill names in their content to express skill associations.
    To discover skills, use `search_skills` instead.

    Args:
        query: Free-text search string.
        depth: BFS expansion depth from seed nodes (default 2).
        top_k: Maximum number of nodes to return (default 15).

    Returns:
        Markdown-formatted list of matching memory nodes.
    """
    kg = _get_memory_kg()
    try:
        seed_ids = _find_seeds(kg, query, top_k=top_k)
        if not seed_ids:
            return f"No knowledge graph entries found for '{query}'."

        G = kg.load_networkx()
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
    kg = _get_memory_kg()
    try:
        name = content[:80].strip().rstrip(".")
        node = kg.upsert_node(
            category="memory",
            name=name,
            description=content,
            content={"raw": content, "context": context},
        )
        vec = _embed_one(_node_text(node.name, content))
        if vec:
            kg.set_embedding(node.id, vec)
        return f"Saved to knowledge graph as memory node: '{node.name}' (id={node.id})"
    except Exception as exc:
        logger.warning("save_to_knowledge_graph failed: %s", exc)
        return f"Failed to save to knowledge graph: {exc}"


def search_skills(query: str, top_k: int = 5) -> str:
    """Search for skills semantically relevant to *query*.

    Uses embedding-based semantic search to rank skill nodes by relevance.
    Results are ordered by cosine similarity to the query — no recency bias.

    Skill names are NOT available in the agent context — this is the only way
    to discover them. Call this with a natural-language description of the task
    or goal, not a skill name.

    To explore the dependency graph from a skill you already know, use
    `get_related_skills` instead.

    Args:
        query: Free-text description of the task or goal.
        top_k: Number of results to return (default 5).

    Returns:
        Markdown list of matching skill names and descriptions, ranked by relevance.
    """
    kg = _get_skill_kg()
    try:
        node_ids = _find_seeds(kg, query, top_k=top_k, category="skill")
        if not node_ids:
            return f"No skills found for '{query}'."

        for nid in node_ids:
            kg.increment_reference(nid)

        lines = []
        for nid in node_ids:
            node = kg.get_node(nid)
            if not node:
                continue
            line = f"- **{node.name}**"
            if node.description:
                line += f": {node.description}"
            lines.append(line)

        return "\n".join(lines) if lines else f"No skills found for '{query}'."

    except Exception as exc:
        logger.warning("search_skills failed: %s", exc)
        return f"Skill search failed: {exc}"


def get_related_skills(start_node: str, top_k: int = 5, depth: int = 2) -> str:
    """Traverse the skill dependency graph from a known skill node.

    Starting from *start_node*, performs BFS over dependency edges to discover
    related skills. Nodes closer to the start rank higher; reference count
    breaks ties. No recency bias.

    Use this after `search_skills` when you already know a skill name and want
    to discover its dependencies or closely related workflows.

    Args:
        start_node: Name of a known skill node (exact, case-insensitive, or substring).
        top_k:      Maximum number of related skills to return (default 5).
        depth:      BFS depth limit (default 2).

    Returns:
        Markdown list of related skill names and descriptions, ranked by proximity.
    """
    kg = _get_skill_kg()
    try:
        start_id = _find_node_id_by_name(kg, start_node)
        if start_id is None:
            return f"No skill node found matching '{start_node}'."

        G = kg.load_networkx()
        if start_id not in G:
            return f"Skill '{start_node}' has no graph edges."

        # BFS tracking hop distance; exclude the start node itself
        visited: dict[str, int] = {}
        queue: deque[tuple[str, int]] = deque([(start_id, 0)])
        while queue:
            nid, d = queue.popleft()
            if nid in visited:
                continue
            visited[nid] = d
            if d >= depth:
                continue
            for neighbor in list(G.successors(nid)) + list(G.predecessors(nid)):
                if neighbor not in visited and G.nodes[neighbor].get("category") == "skill":
                    queue.append((neighbor, d + 1))

        neighbors = [(nid, hop) for nid, hop in visited.items() if nid != start_id]
        neighbors.sort(key=lambda x: (x[1], -G.nodes[x[0]].get("reference_count", 0)))
        top_ids = [nid for nid, _ in neighbors[:top_k]]

        if not top_ids:
            return f"No related skills found near '{start_node}'."

        for nid in top_ids:
            kg.increment_reference(nid)

        lines = []
        for nid in top_ids:
            node = kg.get_node(nid)
            if not node:
                continue
            line = f"- **{node.name}**"
            if node.description:
                line += f": {node.description}"
            lines.append(line)

        return "\n".join(lines) if lines else f"No related skills found near '{start_node}'."

    except Exception as exc:
        logger.warning("get_related_skills failed: %s", exc)
        return f"Related skills lookup failed: {exc}"
