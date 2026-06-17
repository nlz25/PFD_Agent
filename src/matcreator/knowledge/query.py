"""Retrieval and write tools for unified Know-Do Graph storage."""

from __future__ import annotations

import logging
import os
from typing import Optional

from know_do_graph import (
    Entry,
    EntryType,
    KnowDoGraph,
)

from .kdg_memory import (
    add_memory,
    increment_usage,
)
from .review import (
    normalize_review_model,
    review_policy,
    review_threshold,
)

logger = logging.getLogger(__name__)
_graph: Optional[KnowDoGraph] = None
_migration_result: dict[str, int] | None = None


def _configure_auto_review(graph: KnowDoGraph) -> None:
    """Attach KDG's policy-controlled durable-node review scheduler."""
    enabled = os.environ.get("MATCREATOR_AUTO_REVIEW", "1").strip().lower()
    if enabled in {"0", "false", "no", "off"}:
        return

    from ..constants import GRAPH_AGENT_MODEL, LLM_API_KEY, LLM_BASE_URL

    threshold = review_threshold()
    model = normalize_review_model(
        os.environ.get("REVIEW_AGENT_MODEL")
        or os.environ.get("GRAPH_AGENT_MODEL", GRAPH_AGENT_MODEL)
    )
    api_key = (
        os.environ.get("LLM_API_KEY")
        or LLM_API_KEY
        or ""
    )
    base_url = (
        os.environ.get("LLM_BASE_URL")
        or LLM_BASE_URL
        or None
    )
    batch_size = int(os.environ.get("MATCREATOR_REVIEW_BATCH_SIZE", "5"))
    graph.auto_review(
        threshold=threshold,
        policy=review_policy(),
        strategy=os.environ.get("MATCREATOR_REVIEW_STRATEGY", "auto"),
        include_existing=True,
        model=model,
        api_key=api_key,
        base_url=base_url,
        batch_size=batch_size,
    )


def _get_kg() -> KnowDoGraph:
    global _graph, _migration_result
    if _graph is None:
        from ..constants import KNOW_DO_GRAPH_DB, KNOW_DO_MEMORY_DIR

        _graph = KnowDoGraph(path=KNOW_DO_GRAPH_DB, memory_dir=KNOW_DO_MEMORY_DIR)
        _migration_result = {"know_do_nodes": 0, "memory_entries": 0, "edges": 0}
        _configure_auto_review(_graph)
    return _graph

def get_migration_result() -> dict[str, int]:
    _get_kg()
    return dict(_migration_result or {})


# Compatibility aliases for callers from the previous split implementation.
_get_skill_kg = _get_kg
_get_memory_kg = _get_kg


def _format_durable(entries: list[Entry]) -> str:
    lines = []
    for entry in entries:
        level = entry.metadata.skill_level
        if level is None:
            level = {
                EntryType.capability: "L1",
                EntryType.workflow: "L1",
                EntryType.procedure: "L2",
                EntryType.heuristic: "L3",
                EntryType.constraint: "L4",
            }.get(entry.entry_type)
        elif hasattr(level, "value"):
            level = level.value
        label = f"{level} {entry.entry_type.value}" if level else entry.entry_type.value
        line = f"- **{entry.title}** [{label}]"
        if entry.content:
            line += f": {entry.content}"
        lines.append(line)
    return "\n".join(lines)


def _format_memory(entries: list[Entry]) -> str:
    lines = []
    for entry in entries:
        memory = entry.metadata.custom.get("memory", {})
        success = memory.get("success")
        status = "success" if success is True else "failed" if success is False else "unchecked"
        lines.append(f"- [{memory.get('session_id', 'default')}; {status}] {entry.content}")
    return "\n".join(lines)


def _matches_attachment_query(*, query: str, text_parts: list[str]) -> bool:
    if not query.strip():
        return True
    needle = query.casefold()
    return any(needle in part.casefold() for part in text_parts if part)


def _format_node_attachments(entry: Entry, *, query: str = "", top_k: int = 5) -> list[str]:
    sections: list[str] = []
    script_names = {script.filename for script in entry.scripts}

    matching_assets = [
        asset
        for asset in entry.assets
        if not (asset.kind == "script" and asset.filename in script_names)
        if _matches_attachment_query(
            query=query,
            text_parts=[
                asset.folder,
                asset.filename,
                asset.kind,
                asset.description,
                asset.content,
            ],
        )
    ][:top_k]
    if matching_assets:
        lines = []
        for asset in matching_assets:
            path = f"{asset.folder}/{asset.filename}" if asset.folder else asset.filename
            line = f"- **{path}** [{asset.kind}]"
            if asset.content:
                line += f": {asset.content}"
            lines.append(line)
        sections.append("### Attached Files\n" + "\n".join(lines))

    matching_scripts = [
        script
        for script in entry.scripts
        if _matches_attachment_query(
            query=query,
            text_parts=[script.filename, script.description, script.language, script.content],
        )
    ][:top_k]
    if matching_scripts:
        lines = []
        for script in matching_scripts:
            line = f"- **scripts/{script.filename}** [{script.language}]"
            if script.content:
                line += f": {script.content}"
            lines.append(line)
        sections.append("### Attached Scripts\n" + "\n".join(lines))

    matching_refs = [
        ref for ref in entry.internal_refs if _matches_attachment_query(query=query, text_parts=[ref])
    ][:top_k]
    if matching_refs:
        sections.append("### Internal References\n" + "\n".join(f"- `{ref}`" for ref in matching_refs))

    return sections


def query_knowledge_graph(query: str, depth: int = 2, top_k: int = 15) -> str:
    """Retrieve L1/L2 planning context and relevant working memory.

    ``depth`` is retained for compatibility with older callers. L3/L4 entries
    are intentionally not expanded here; use ``search_skill_context`` after
    selecting an L1/L2 node.
    """
    del depth
    graph = _get_kg()
    try:
        durable = [
            entry
            for entry in graph.plan(
                query,
                limit=max(top_k * 3, 20),
                mode="hybrid",
                include_procedures=True,
            )
            if entry.entry_type != EntryType.memory
        ][:top_k]
        for entry in durable:
            increment_usage(graph, entry)

        memory_entries = [
            entry
            for entry in graph.search(
                query,
                entry_type=EntryType.memory,
                limit=top_k,
                mode="hybrid",
            )
            if not entry.metadata.custom.get("memory", {}).get("promoted", False)
        ]
        for entry in memory_entries:
            increment_usage(graph, entry)

        sections = []
        if durable:
            sections.append("### L1/L2 Planning Knowledge\n" + _format_durable(durable))
        if memory_entries:
            sections.append("### Working Memory\n" + _format_memory(memory_entries))
        if durable:
            sections.append(
                "Select a planning node, then call `search_skill_context` to "
                "conditionally retrieve its attached L3/L4 knowledge."
            )
        return "\n\n".join(sections) or f"No knowledge graph entries found for '{query}'."
    except Exception as exc:
        logger.warning("query_knowledge_graph failed: %s", exc)
        return f"Knowledge graph query failed: {exc}"


def save_to_knowledge_graph(
    content: str,
    context: str = "",
    session_id: str = "default",
    success: bool | None = None,
) -> str:
    """Write an observation to MemGraph for later validation and distillation."""
    graph = _get_kg()
    tags = ["matcreator-memory"]
    if context:
        tags.append(f"context:{context[:80]}")
    try:
        memory = add_memory(
            graph,
            session_id,
            content,
            tags=tags,
            success=success,
        )
        return f"Saved working memory (id={memory.id}, session={session_id})."
    except Exception as exc:
        logger.warning("save_to_knowledge_graph failed: %s", exc)
        return f"Failed to save working memory: {exc}"


def search_skills(query: str, top_k: int = 5) -> str:
    """Search planner-level L1 capabilities/workflows and L2 procedures."""
    from ..config import get_disabled_skills

    graph = _get_kg()
    disabled = set(get_disabled_skills())
    try:
        results = [
            entry
            for entry in graph.plan(
                query,
                limit=max(top_k * 4, 20),
                mode="hybrid",
                include_procedures=True,
            )
            if "matcreator-skill" in entry.tags
            and entry.title not in disabled
        ][:top_k]
        for entry in results:
            increment_usage(graph, entry)
        return _format_durable(results) if results else f"No skills found for '{query}'."
    except Exception as exc:
        logger.warning("search_skills failed: %s", exc)
        return f"Skill search failed: {exc}"


def search_skill_context(
    skill: str,
    query: str = "",
    include_heuristics: bool = True,
    include_constraints: bool = True,
    top_k: int = 5,
) -> str:
    """Conditionally search L3/L4 nodes attached to a selected L1/L2 node.

    The candidate pool is scoped by ``heuristic_for``, ``constraint_on``, and
    ``warning_about`` edges before ranking, so unrelated L3/L4 nodes cannot leak
    into the result.
    """
    graph = _get_kg()
    try:
        start = graph.get(skill)
        if start is None:
            matches = graph.plan(
                skill,
                limit=1,
                mode="hybrid",
                include_procedures=True,
            )
            start = matches[0] if matches else None
        if start is None:
            return f"No L1/L2 node found matching '{skill}'."

        sections = [f"### Selected Node\n{_format_durable([start])}"]
        used_entries: list[Entry] = []
        attached = graph.count_attached(start.id)

        if include_heuristics and attached.get("heuristics", 0) > 0:
            heuristics, total = graph.search_attached(
                start.id,
                kind="heuristics",
                query=query,
                limit=top_k,
                mode="hybrid",
            )
            if heuristics:
                sections.append(
                    f"### Attached L3 Heuristics ({len(heuristics)}/{total})\n"
                    + _format_durable(heuristics)
                )
                used_entries.extend(heuristics)
        if include_constraints and attached.get("constraints", 0) > 0:
            constraints, total = graph.search_attached(
                start.id,
                kind="constraints",
                query=query,
                limit=top_k,
                mode="hybrid",
            )
            if constraints:
                sections.append(
                    f"### Attached L4 Constraints ({len(constraints)}/{total})\n"
                    + _format_durable(constraints)
                )
                used_entries.extend(constraints)

        for entry in used_entries:
            increment_usage(graph, entry)

        sections.extend(_format_node_attachments(start, query=query, top_k=top_k))

        if len(sections) == 1:
            sections.append("No attached L3/L4 context matched the requested scope.")
        return "\n\n".join(sections)
    except Exception as exc:
        logger.warning("search_skill_context failed: %s", exc)
        return f"Conditional skill context search failed: {exc}"


def get_related_skills(start_node: str, top_k: int = 5, depth: int = 2) -> str:
    """Traverse durable Know-Do relationships from a known skill."""
    graph = _get_kg()
    try:
        start = graph.get(start_node)
        if start is None:
            matches = graph.search(start_node, tags=["matcreator-skill"], limit=1)
            start = matches[0] if matches else None
        if start is None:
            return f"No skill node found matching '{start_node}'."

        related = [
            entry
            for entry in graph.related(start.id, depth=depth)
            if "matcreator-skill" in entry.tags
        ][:top_k]
        for entry in related:
            increment_usage(graph, entry)
        return _format_durable(related) if related else f"No related skills found near '{start_node}'."
    except Exception as exc:
        logger.warning("get_related_skills failed: %s", exc)
        return f"Related skills lookup failed: {exc}"
