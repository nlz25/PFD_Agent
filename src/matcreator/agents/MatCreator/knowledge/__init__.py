"""Hierarchical knowledge graph for MatCreator.

Stores concepts, skills, materials, results, insights, and workflows in
SQLite and exposes BFS-based retrieval for the thinking agent.
"""

from .graph_store import KnowledgeGraph
from .query import query_knowledge_graph, save_to_knowledge_graph, search_skills, get_related_skills
from .extractor import run_knowledge_extractor
from .synthesizer import run_knowledge_synthesizer
from .migrate import migrate_memory_md
from .kg_state import increment_exec_count, record_synthesizer_run, get_exec_count

__all__ = [
    "KnowledgeGraph",
    "query_knowledge_graph",
    "save_to_knowledge_graph",
    "search_skills",
    "get_related_skills",
    "run_knowledge_extractor",
    "run_knowledge_synthesizer",
    "migrate_memory_md",
    "increment_exec_count",
    "record_synthesizer_run",
    "get_exec_count",
]
