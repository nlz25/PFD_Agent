"""Knowledge synthesizer: prune stale nodes, merge near-duplicates, and abstract patterns."""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from sqlalchemy import select

from .graph_store import KnowledgeGraph  # noqa: F401 (kept for type hints)
from .query import _get_memory_kg
from .schema import KgNode, KgEdge

logger = logging.getLogger(__name__)


def run_knowledge_synthesizer(
    stale_days: int = 30,
    stale_min_refs: int = 0,
    min_insights_for_workflow: int = 3,
) -> dict:
    """Prune, merge, and abstract the knowledge graph.

    Runs three passes:
    1. Prune: delete nodes older than *stale_days* with ≤ *stale_min_refs* references.
    2. Merge: collapse nodes linked by `similar_to` edges into the most-referenced one.
    3. Abstract: when ≥ *min_insights_for_workflow* Insight nodes share a `discovered_in`
       Skill/Workflow, synthesize a new Workflow abstraction node above them.

    Returns:
        Dict with keys: pruned, merged, abstracted, message.
    """
    kg = _get_memory_kg()
    stats = {"pruned": 0, "merged": 0, "abstracted": 0}

    # ------------------------------------------------------------------
    # Pass 1: Prune stale memory nodes (skill nodes are never pruned)
    # ------------------------------------------------------------------
    now = datetime.now(timezone.utc)
    with kg._Session() as sess:
        stale_candidates = sess.execute(
            select(KgNode).where(
                KgNode.reference_count <= stale_min_refs,
                KgNode.category == "memory",
            )
        ).scalars().all()

        to_delete: list[str] = []
        for node in stale_candidates:
            if node.created_at:
                age_days = (now - node.created_at.replace(tzinfo=timezone.utc)).days
                if age_days >= stale_days:
                    to_delete.append(node.id)

        for nid in to_delete:
            n = sess.get(KgNode, nid)
            if n:
                sess.delete(n)
        sess.commit()
        stats["pruned"] = len(to_delete)

    # ------------------------------------------------------------------
    # Pass 2: Merge relates_to clusters among memory nodes
    # ------------------------------------------------------------------
    with kg._Session() as sess:
        similar_edges = sess.execute(
            select(KgEdge).where(KgEdge.edge_type == "relates_to")
        ).scalars().all()

        # Build adjacency for union-find
        parent: dict[str, str] = {}

        def find(x: str) -> str:
            while parent.get(x, x) != x:
                parent[x] = parent.get(parent.get(x, x), x)
                x = parent[x]
            return x

        def union(a: str, b: str) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for edge in similar_edges:
            union(edge.source_id, edge.target_id)

        # Find clusters (groups with >1 member)
        clusters: dict[str, list[str]] = {}
        all_ids = set()
        for edge in similar_edges:
            all_ids.update([edge.source_id, edge.target_id])
        for nid in all_ids:
            root = find(nid)
            clusters.setdefault(root, []).append(nid)

        merged_count = 0
        for root, members in clusters.items():
            if len(members) <= 1:
                continue
            # Only merge if all members are memory nodes
            nodes = [sess.get(KgNode, m) for m in members if sess.get(KgNode, m)]
            nodes = [n for n in nodes if n and n.category == "memory"]
            if len(nodes) <= 1:
                continue
            canonical = max(nodes, key=lambda n: n.reference_count)
            for node in nodes:
                if node.id == canonical.id:
                    continue
                # Redirect all edges from/to this node to canonical
                for e in list(sess.execute(
                    select(KgEdge).where(KgEdge.source_id == node.id)
                ).scalars()):
                    e.source_id = canonical.id
                for e in list(sess.execute(
                    select(KgEdge).where(KgEdge.target_id == node.id)
                ).scalars()):
                    e.target_id = canonical.id
                canonical.reference_count += node.reference_count
                sess.delete(node)
                merged_count += 1
        sess.commit()
        stats["merged"] = merged_count

    # ------------------------------------------------------------------
    # Pass 3: Abstract memory clusters into skill concept nodes
    # When ≥ min_insights_for_workflow memory nodes all relate_to the same
    # skill node, synthesize a new "concept" skill node above them.
    # NOTE: With the split graph architecture (skill_graph.db / memory_graph.db),
    # cross-graph relates_to edges no longer exist, so this pass is currently a
    # no-op. TODO: replace with embedding-based cross-graph abstraction.
    # ------------------------------------------------------------------
    with kg._Session() as sess:
        relates_edges = sess.execute(
            select(KgEdge).where(KgEdge.edge_type == "relates_to")
        ).scalars().all()

        # Group memory nodes by the skill node they relate to
        skill_to_memories: dict[str, list[str]] = {}
        for edge in relates_edges:
            src = sess.get(KgNode, edge.source_id)
            tgt = sess.get(KgNode, edge.target_id)
            if src and tgt and src.category == "memory" and tgt.category == "skill":
                skill_to_memories.setdefault(edge.target_id, []).append(edge.source_id)

        abstracted_count = 0
        for skill_id, memory_ids in skill_to_memories.items():
            if len(memory_ids) < min_insights_for_workflow:
                continue
            skill_node = sess.get(KgNode, skill_id)
            if not skill_node:
                continue
            # Check if a synthesized concept node already exists for this skill
            concept_name = f"Concept: {skill_node.name}"
            existing = sess.execute(
                select(KgNode).where(
                    KgNode.category == "skill",
                    KgNode.name == concept_name,
                )
            ).scalars().first()
            if existing:
                continue

            concept = KgNode(
                id=str(uuid.uuid4()),
                category="skill",
                type="skill",
                name=concept_name,
                description=(
                    f"Synthesized concept from {len(memory_ids)} memory nodes "
                    f"accumulated around '{skill_node.name}'."
                ),
                source_session="synthesizer",
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
                reference_count=0,
                confidence=0.8,
            )
            sess.add(concept)
            sess.flush()

            # Link skill node → concept via belongs_to
            edge = KgEdge(
                id=str(uuid.uuid4()),
                source_id=skill_id,
                target_id=concept.id,
                edge_type="belongs_to",
                weight=1.0,
                created_at=datetime.now(timezone.utc),
            )
            sess.add(edge)
            abstracted_count += 1
        sess.commit()
        stats["abstracted"] = abstracted_count

    logger.info(
        "Synthesizer: pruned=%d merged=%d abstracted=%d",
        stats["pruned"], stats["merged"], stats["abstracted"],
    )
    return {
        **stats,
        "message": (
            f"Synthesizer complete: pruned {stats['pruned']}, "
            f"merged {stats['merged']}, abstracted {stats['abstracted']}."
        ),
    }
