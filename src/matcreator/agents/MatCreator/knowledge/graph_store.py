"""CRUD operations and NetworkX graph loader for the knowledge graph."""

from __future__ import annotations

import uuid
import difflib
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import networkx as nx
from sqlalchemy import create_engine, select, text
from sqlalchemy.orm import sessionmaker

from .schema import Base, KgNode, KgEdge, CATEGORIES, EDGE_TYPES


def _db_url(db_path: Path) -> str:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{db_path}"


def _make_engine(db_path: Path):
    return create_engine(
        _db_url(db_path),
        connect_args={"check_same_thread": False},
    )


class KnowledgeGraph:
    """Persistent property graph backed by SQLite."""

    def __init__(self, db_path: Path) -> None:
        self._engine = _make_engine(db_path)
        Base.metadata.create_all(self._engine)
        self._Session = sessionmaker(bind=self._engine)
        self._migrate()

    def _migrate(self) -> None:
        """Add columns and migrate data introduced after initial schema creation."""
        _SKILL_TYPES = {"Concept", "Skill", "Workflow"}
        _EDGE_MAP = {
            "requires": "depends_on",
            "specializes": "belongs_to",
            "discovered_in": "relates_to",
            "similar_to": "relates_to",
            "produces": "relates_to",
            "tested_on": "relates_to",
            "supersedes": "relates_to",
        }
        with self._engine.connect() as conn:
            cols = [row[1] for row in conn.execute(text("PRAGMA table_info(kg_nodes)"))]
            if "embedding" not in cols:
                conn.execute(text("ALTER TABLE kg_nodes ADD COLUMN embedding TEXT"))
                conn.commit()
            if "immutable" not in cols:
                conn.execute(text("ALTER TABLE kg_nodes ADD COLUMN immutable INTEGER NOT NULL DEFAULT 0"))
                conn.commit()
            if "category" not in cols:
                conn.execute(text("ALTER TABLE kg_nodes ADD COLUMN category TEXT"))
                # Backfill: old node types → category
                conn.execute(text(
                    "UPDATE kg_nodes SET category = CASE "
                    "WHEN type IN ('Concept','Skill','Workflow') THEN 'skill' "
                    "ELSE 'memory' END"
                ))
                conn.commit()
            # Migrate legacy edge types to new 3-type set
            for old, new in _EDGE_MAP.items():
                conn.execute(
                    text("UPDATE kg_edges SET edge_type = :new WHERE edge_type = :old"),
                    {"new": new, "old": old},
                )
            conn.commit()

    # ------------------------------------------------------------------
    # Node operations
    # ------------------------------------------------------------------

    def upsert_node(
        self,
        category: str,
        name: str,
        description: str = "",
        content: Optional[dict] = None,
        source_session: Optional[str] = None,
        confidence: float = 1.0,
        similarity_threshold: float = 0.85,
        immutable: bool = False,
    ) -> KgNode:
        """Insert a new node or return an existing near-duplicate.

        Deduplication uses difflib on existing node names of the same category.
        If a match exceeds *similarity_threshold*, the existing node is returned
        (with description merged) instead of creating a duplicate.
        """
        if category not in CATEGORIES:
            raise ValueError(f"category must be one of {CATEGORIES}, got {category!r}")
        with self._Session() as sess:
            # Exact match first
            existing = sess.execute(
                select(KgNode).where(KgNode.category == category, KgNode.name == name)
            ).scalars().first()
            if existing:
                return existing

            # Fuzzy dedup among same-category nodes
            candidates = sess.execute(
                select(KgNode).where(KgNode.category == category)
            ).scalars().all()
            for c in candidates:
                ratio = difflib.SequenceMatcher(None, name.lower(), c.name.lower()).ratio()
                if ratio >= similarity_threshold:
                    # Immutable nodes are returned unchanged
                    if not c.immutable and description and (not c.description or len(description) > len(c.description)):
                        c.description = description
                        c.updated_at = datetime.now(timezone.utc)
                        sess.commit()
                    sess.expunge(c)
                    return c

            node = KgNode(
                id=str(uuid.uuid4()),
                category=category,
                type=category,  # kept for legacy compat
                name=name,
                description=description,
                content=content,
                source_session=source_session,
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
                reference_count=0,
                confidence=confidence,
                immutable=immutable,
            )
            sess.add(node)
            sess.commit()
            sess.refresh(node)
            sess.expunge(node)
            return node

    def get_node(self, node_id: str) -> Optional[KgNode]:
        with self._Session() as sess:
            n = sess.get(KgNode, node_id)
            if n:
                sess.expunge(n)
            return n

    def find_nodes_by_name(self, query: str, category: Optional[str] = None) -> list[KgNode]:
        """Token-based search across node name and description.

        Splits *query* into words and returns nodes where ANY token appears
        in either the name or the description (case-insensitive).
        """
        tokens = [t for t in query.split() if len(t) >= 2]
        if not tokens:
            return []
        with self._Session() as sess:
            from sqlalchemy import or_
            conditions = [
                or_(
                    KgNode.name.ilike(f"%{tok}%"),
                    KgNode.description.ilike(f"%{tok}%"),
                )
                for tok in tokens
            ]
            from sqlalchemy import or_ as sql_or
            stmt = select(KgNode).where(sql_or(*conditions))
            if category:
                stmt = stmt.where(KgNode.category == category)
            nodes = sess.execute(stmt).scalars().all()
            for n in nodes:
                sess.expunge(n)
            return list(nodes)

    def increment_reference(self, node_id: str) -> None:
        with self._Session() as sess:
            n = sess.get(KgNode, node_id)
            if n:
                n.reference_count += 1
                n.updated_at = datetime.now(timezone.utc)
                sess.commit()

    def delete_node(self, node_id: str) -> None:
        with self._Session() as sess:
            n = sess.get(KgNode, node_id)
            if n:
                sess.delete(n)
                sess.commit()

    def set_embedding(self, node_id: str, embedding: list[float]) -> None:
        """Persist the embedding vector for a node."""
        with self._Session() as sess:
            n = sess.get(KgNode, node_id)
            if n:
                n.embedding = embedding
                n.updated_at = datetime.now(timezone.utc)
                sess.commit()

    def get_all_embeddings(self) -> list[tuple[str, list[float]]]:
        """Return (node_id, embedding) for all nodes that have an embedding."""
        with self._Session() as sess:
            rows = sess.execute(
                select(KgNode.id, KgNode.embedding).where(KgNode.embedding.isnot(None))
            ).all()
            return [(row.id, row.embedding) for row in rows]

    def get_nodes_without_embeddings(self) -> list[KgNode]:
        """Return nodes that still need an embedding computed."""
        with self._Session() as sess:
            nodes = sess.execute(
                select(KgNode).where(KgNode.embedding.is_(None))
            ).scalars().all()
            for n in nodes:
                sess.expunge(n)
            return list(nodes)

    # ------------------------------------------------------------------
    # Edge operations
    # ------------------------------------------------------------------

    def upsert_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: str,
        weight: float = 1.0,
        properties: Optional[dict] = None,
    ) -> KgEdge:
        """Create an edge or increment weight if it already exists.

        Raises ValueError if the edge would go from a skill node to a memory node.
        """
        with self._Session() as sess:
            src = sess.get(KgNode, source_id)
            tgt = sess.get(KgNode, target_id)
            if src and tgt and src.category == "skill" and tgt.category == "memory":
                raise ValueError(
                    f"Forbidden edge: skill node '{src.name}' → memory node '{tgt.name}'. "
                    "Skill nodes must not depend on ephemeral memory."
                )
            existing = sess.execute(
                select(KgEdge).where(
                    KgEdge.source_id == source_id,
                    KgEdge.target_id == target_id,
                    KgEdge.edge_type == edge_type,
                )
            ).scalars().first()
            if existing:
                existing.weight += weight
                sess.commit()
                sess.expunge(existing)
                return existing

            edge = KgEdge(
                id=str(uuid.uuid4()),
                source_id=source_id,
                target_id=target_id,
                edge_type=edge_type,
                weight=weight,
                properties=properties,
                created_at=datetime.now(timezone.utc),
            )
            sess.add(edge)
            sess.commit()
            sess.refresh(edge)
            sess.expunge(edge)
            return edge

    def get_edges(
        self,
        source_id: Optional[str] = None,
        target_id: Optional[str] = None,
    ) -> list[KgEdge]:
        with self._Session() as sess:
            stmt = select(KgEdge)
            if source_id:
                stmt = stmt.where(KgEdge.source_id == source_id)
            if target_id:
                stmt = stmt.where(KgEdge.target_id == target_id)
            edges = sess.execute(stmt).scalars().all()
            for e in edges:
                sess.expunge(e)
            return list(edges)

    # ------------------------------------------------------------------
    # NetworkX loader
    # ------------------------------------------------------------------

    def load_networkx(self) -> nx.DiGraph:
        """Load the full graph into a NetworkX DiGraph for traversal."""
        G = nx.DiGraph()
        with self._Session() as sess:
            for n in sess.execute(select(KgNode)).scalars():
                G.add_node(
                    n.id,
                    category=n.category or "memory",
                    name=n.name,
                    description=n.description or "",
                    reference_count=n.reference_count,
                    confidence=n.confidence,
                    immutable=bool(n.immutable),
                    created_at=n.created_at.isoformat() if n.created_at else "",
                )
            for e in sess.execute(select(KgEdge)).scalars():
                G.add_edge(e.source_id, e.target_id,
                           edge_type=e.edge_type, weight=e.weight)
        return G

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        with self._Session() as sess:
            node_count = sess.execute(select(KgNode)).scalars().all()
            edge_count = sess.execute(select(KgEdge)).scalars().all()
            category_counts: dict[str, int] = {}
            for n in node_count:
                cat = n.category or "memory"
                category_counts[cat] = category_counts.get(cat, 0) + 1
            return {
                "nodes": len(node_count),
                "edges": len(edge_count),
                "by_category": category_counts,
            }
