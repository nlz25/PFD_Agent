"""Markdown-backed workflow skills and search utilities for MatCreator.

Skills are now loaded in standard ADK format via the top-level skill.py
(google.adk.skills.load_skill_from_dir).  This module bridges ADK Skill
objects to the interface expected by the planning/execution agents and keeps
the guide system unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from ..constants import _GUIDES_DIR, _workspace_guides_dir

# Re-export ADK Skill so callers can still do `from .skill import Skill`.
from google.adk.skills import Skill  # noqa: F401


@dataclass(frozen=True)
class Guide:
    """Higher-level guidance on how to organise and deploy skills for specific tasks."""
    name: str
    description: str
    body: str
    tags: List[str]
    skills: List[str]
    source_path: str


def _parse_list_value(raw_value: str) -> List[str]:
    value = (raw_value or "").strip()
    if not value:
        return []
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [item.strip().strip("\"'") for item in inner.split(",") if item.strip()]
    return [item.strip() for item in value.split(",") if item.strip()]


# ---------------------------------------------------------------------------
# Skill interface — bridged to ADK ALL_SKILLS
# ---------------------------------------------------------------------------

def _get_all_skills() -> list:
    """Return ALL_SKILLS from the top-level skill.py (lazy import)."""
    from ..skill import ALL_SKILLS  # noqa: PLC0415
    return ALL_SKILLS


def _load_skill_registry() -> Dict[str, Skill]:
    """Return a name → ADK Skill mapping.  Re-scans on every call."""
    return {s.name: s for s in _get_all_skills()}


def list_skill_name_descriptions() -> List[Dict[str, str]]:
    """Return planner-facing skill summaries (name, description)."""
    return [
        {"name": s.name, "description": s.description}
        for s in _get_all_skills()
    ]


def load_skill_content(skill_name: str) -> dict:
    """Fetch the full instruction body of a skill by name.

    Call this when a skill's description is not sufficient to inform planning
    and the full instruction text is needed.

    Args:
        skill_name: Exact skill name as listed by list_skill_name_descriptions.

    Returns:
        Dict with ``name``, ``description``, ``needed_tools``, ``dependent_skills``,
        and ``instruction`` (the full markdown body), or an ``error`` key if not found.
    """
    skills = _get_all_skills()
    skill = next((s for s in skills if s.name == skill_name), None)
    if skill is None:
        lowered = skill_name.lower()
        skill = next((s for s in skills if s.name.lower() == lowered), None)
    if skill is None:
        available = ", ".join(sorted(s.name for s in skills)) or "<none>"
        return {"error": f"Skill '{skill_name}' not found. Available skills: {available}"}
    metadata = skill.frontmatter.metadata if skill.frontmatter else {}
    return {
        "name": skill.name,
        "description": skill.description,
        "needed_tools": metadata.get("tools", []),
        "dependent_skills": metadata.get("dependent_skills", []),
        "instruction": skill.instructions,
    }


# Snapshot at import time for callers that cache it; live data is always
# available via _load_skill_registry() or _get_all_skills().
SKILL_LIBRARY: Dict[str, Skill] = _load_skill_registry()


# ---------------------------------------------------------------------------
# Guide system (unchanged)
# ---------------------------------------------------------------------------

def _parse_guide_markdown(path: Path) -> Guide:
    raw_text = path.read_text(encoding="utf-8")
    stripped = raw_text.strip()

    metadata: Dict[str, str] = {}
    body = raw_text

    if stripped.startswith("---"):
        first_sep = raw_text.find("---")
        second_sep = raw_text.find("\n---", first_sep + 3)
        if second_sep != -1:
            frontmatter = raw_text[first_sep + 3:second_sep].strip()
            body = raw_text[second_sep + 4:].strip()
            for line in frontmatter.splitlines():
                if ":" not in line:
                    continue
                key, value = line.split(":", 1)
                metadata[key.strip()] = value.strip()

    return Guide(
        name=metadata.get("name") or path.stem,
        description=metadata.get("description", ""),
        body=body.strip(),
        tags=_parse_list_value(metadata.get("tags", "")),
        skills=_parse_list_value(metadata.get("skills", "")),
        source_path=str(path),
    )


def _load_guide_registry() -> List[Guide]:
    """Build merged guide list (built-ins + workspace overlay, workspace wins on name clash)."""
    seen: Dict[str, Guide] = {}

    if _GUIDES_DIR.exists():
        for p in sorted(_GUIDES_DIR.glob("*.md")):
            g = _parse_guide_markdown(p)
            seen[g.name] = g

    try:
        ws_guides = _workspace_guides_dir()
        if ws_guides.exists():
            for p in sorted(ws_guides.glob("*.md")):
                g = _parse_guide_markdown(p)
                seen[g.name] = g
    except Exception:
        pass

    return list(seen.values())


def list_guide_metadata() -> List[Dict[str, str]]:
    """Return planner-facing guide summaries (name, description, tags) — no body."""
    return [
        {
            "name": g.name,
            "description": g.description,
            "tags": ", ".join(g.tags),
            "skills": ", ".join(g.skills),
        }
        for g in _load_guide_registry()
    ]


def load_guide_content(guide_name: str) -> dict:
    """Fetch the full body of a guide by name.
    Call this before building or updating an execution plan when a guide is
    relevant to the user's goal.
    Can be called multiple times before plan building as the agent discovers relevant guides.

    Args:
        guide_name: Exact guide name as listed by list_guide_metadata.

    Returns:
        Dict with ``name``, ``description``, ``tags``, ``skills``, and ``body``
        (the full markdown content), or an ``error`` key if not found.
    """
    registry = {g.name: g for g in _load_guide_registry()}
    guide = registry.get(guide_name)
    if guide is None:
        available = ", ".join(sorted(registry.keys())) or "<none>"
        return {"error": f"Guide '{guide_name}' not found. Available guides: {available}"}
    return {
        "name": guide.name,
        "description": guide.description,
        "tags": guide.tags,
        "skills": guide.skills,
        "body": guide.body,
    }


__all__ = [
    "Skill",
    "Guide",
    "list_skill_name_descriptions",
    "load_skill_content",
    "list_guide_metadata",
    "load_guide_content",
    "SKILL_LIBRARY",
    "_load_skill_registry",
]
