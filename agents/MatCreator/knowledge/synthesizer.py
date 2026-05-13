"""Knowledge synthesizer: prune stale nodes, merge near-duplicates, and abstract patterns."""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from sqlalchemy import select

from .graph_store import KnowledgeGraph
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
    kg = KnowledgeGraph()
    stats = {"pruned": 0, "merged": 0, "abstracted": 0}

    # ------------------------------------------------------------------
    # Pass 1: Prune stale nodes
    # ------------------------------------------------------------------
    now = datetime.now(timezone.utc)
    with kg._Session() as sess:
        stale_candidates = sess.execute(
            select(KgNode).where(KgNode.reference_count <= stale_min_refs)
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
    # Pass 2: Merge similar_to clusters
    # ------------------------------------------------------------------
    with kg._Session() as sess:
        similar_edges = sess.execute(
            select(KgEdge).where(KgEdge.edge_type == "similar_to")
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
            # Keep the node with highest reference_count as canonical
            nodes = [sess.get(KgNode, m) for m in members if sess.get(KgNode, m)]
            if not nodes:
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
    # Pass 3: Abstract Insight clusters into Workflow nodes
    # ------------------------------------------------------------------
    with kg._Session() as sess:
        disc_edges = sess.execute(
            select(KgEdge).where(KgEdge.edge_type == "discovered_in")
        ).scalars().all()

        # Group Insight nodes by their target (Skill/Workflow they were discovered in)
        skill_to_insights: dict[str, list[str]] = {}
        for edge in disc_edges:
            src = sess.get(KgNode, edge.source_id)
            if src and src.type == "Insight":
                skill_to_insights.setdefault(edge.target_id, []).append(edge.source_id)

        abstracted_count = 0
        for skill_id, insight_ids in skill_to_insights.items():
            if len(insight_ids) < min_insights_for_workflow:
                continue
            skill_node = sess.get(KgNode, skill_id)
            if not skill_node:
                continue
            # Check if a Workflow abstraction already exists for this skill
            existing_name = f"Workflow: {skill_node.name}"
            existing = sess.execute(
                select(KgNode).where(
                    KgNode.type == "Workflow",
                    KgNode.name == existing_name,
                )
            ).scalars().first()
            if existing:
                continue

            wf = KgNode(
                id=str(uuid.uuid4()),
                type="Workflow",
                name=existing_name,
                description=(
                    f"Abstracted workflow pattern synthesized from "
                    f"{len(insight_ids)} insights about '{skill_node.name}'."
                ),
                source_session="synthesizer",
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
                reference_count=0,
                confidence=0.8,
            )
            sess.add(wf)
            sess.flush()

            # Link insights → new Workflow
            for iid in insight_ids:
                edge = KgEdge(
                    id=str(uuid.uuid4()),
                    source_id=iid,
                    target_id=wf.id,
                    edge_type="discovered_in",
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
