"""SQLAlchemy ORM models for the knowledge graph."""

from __future__ import annotations

from datetime import datetime, timezone
from sqlalchemy import (
    Column, String, Integer, Float, DateTime, JSON, Boolean, ForeignKey,
    Index,
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()

CATEGORIES = {"skill", "memory"}
EDGE_TYPES = {"depends_on", "belongs_to", "relates_to"}


class KgNode(Base):
    __tablename__ = "kg_nodes"

    id              = Column(String, primary_key=True)
    category        = Column(String, nullable=False)   # "skill" | "memory"
    type            = Column(String, nullable=True)    # legacy, kept for migration compat
    name            = Column(String, nullable=False)
    description     = Column(String, nullable=True)
    content         = Column(JSON,   nullable=True)
    source_session  = Column(String, nullable=True)
    created_at      = Column(DateTime, nullable=False,
                             default=lambda: datetime.now(timezone.utc))
    updated_at      = Column(DateTime, nullable=False,
                             default=lambda: datetime.now(timezone.utc),
                             onupdate=lambda: datetime.now(timezone.utc))
    reference_count = Column(Integer, nullable=False, default=0)
    confidence      = Column(Float,   nullable=False, default=1.0)
    immutable       = Column(Boolean, nullable=False, default=False)
    embedding       = Column(JSON,    nullable=True)   # float list from embedding API

    out_edges = relationship("KgEdge", foreign_keys="KgEdge.source_id",
                             back_populates="source", cascade="all, delete-orphan")
    in_edges  = relationship("KgEdge", foreign_keys="KgEdge.target_id",
                             back_populates="target", cascade="all, delete-orphan")

    __table_args__ = (
        Index("idx_nodes_category", "category"),
        Index("idx_nodes_name", "name"),
    )


class KgEdge(Base):
    __tablename__ = "kg_edges"

    id         = Column(String, primary_key=True)
    source_id  = Column(String, ForeignKey("kg_nodes.id", ondelete="CASCADE"),
                        nullable=False)
    target_id  = Column(String, ForeignKey("kg_nodes.id", ondelete="CASCADE"),
                        nullable=False)
    edge_type  = Column(String, nullable=False)
    weight     = Column(Float,  nullable=False, default=1.0)
    properties = Column(JSON,   nullable=True)
    created_at = Column(DateTime, nullable=False,
                        default=lambda: datetime.now(timezone.utc))

    source = relationship("KgNode", foreign_keys=[source_id], back_populates="out_edges")
    target = relationship("KgNode", foreign_keys=[target_id], back_populates="in_edges")

    __table_args__ = (
        Index("idx_edges_source", "source_id"),
        Index("idx_edges_target", "target_id"),
    )
