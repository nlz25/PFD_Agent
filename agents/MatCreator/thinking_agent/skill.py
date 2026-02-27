"""Markdown-backed workflow skills and search utilities for MatCreator."""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence
from ...constants import _SKILLS_DIR


@dataclass(frozen=True)
class Skill:
    """Container for agent skillsfrom markdown skills."""
    # deprecate workflow_type
    # workflow_type: str
    instruction: str
    tags: List[str]
    keywords: List[str]
    description: str
    allowed_agents: List[str]
    name: str
    source_path: str


@dataclass(frozen=True)
class SkillSearchResult:
    """Single ranked skill search hit."""

    name: str
    description: str
    tags: List[str]
    allowed_agents: List[str]
    source_path: str
    guidance: str
    score: float
    workflow_type: str = ""  # deprecated, kept for backward compatibility


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9_\-]+", (text or "").lower())


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


def _parse_skill_markdown(path: Path) -> Skill:
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

    name = metadata.get("name") or path.stem
    description = metadata.get("description", "")
    tags = _parse_list_value(metadata.get("tags", ""))
    keywords = _parse_list_value(metadata.get("triggers", ""))
    allowed_agents = _parse_list_value(metadata.get("allowed_agents", ""))

    return Skill(
        instruction=body.strip(),
        tags=tags,
        keywords=keywords,
        description=description,
        allowed_agents=allowed_agents,
        name=name,
        source_path=str(path),
    )


@lru_cache(maxsize=1)
def _load_skill_registry() -> Dict[str, Skill]:
    registry: Dict[str, Skill] = {}
    for md_path in sorted(_SKILLS_DIR.glob("*.md")):
        skill = _parse_skill_markdown(md_path)
        registry[skill.name] = skill
    return registry


def _skill_score(skill: Skill, query_tokens: Sequence[str]) -> float:
    if not query_tokens:
        return 0.0

    searchable = " ".join(
        [
            skill.name,
            skill.description,
            " ".join(skill.tags),
            " ".join(skill.keywords),
            skill.instruction,
        ]
    ).lower()

    score = 0.0
    for token in query_tokens:
        if token in searchable:
            score += 1.0
        if token in {keyword.lower() for keyword in skill.keywords}:
            score += 1.5
        if token in {tag.lower() for tag in skill.tags}:
            score += 1.0
    return score


def search_skills(
    query: str,
    top_k: int = 3,
) -> List[SkillSearchResult]:
    """Search markdown skills by semantic token overlap and optional workflow filter."""
    registry = _load_skill_registry()
    skills = list(registry.values())
    query_tokens = _tokenize(query)
    ranked = sorted(
        (
            SkillSearchResult(
                name=skill.name,
                description=skill.description,
                tags=skill.tags,
                allowed_agents=skill.allowed_agents,
                source_path=skill.source_path,
                guidance=skill.instruction,
                score=_skill_score(skill, query_tokens),
            )
            for skill in skills
        ),
        key=lambda item: item.score,
        reverse=True,
    )

    if not ranked:
        return []

    # Ensure at least one result even when all scores are zero.
    non_zero = [item for item in ranked if item.score > 0]
    effective = non_zero if non_zero else ranked
    return effective[: max(1, top_k)]


SKILL_LIBRARY = _load_skill_registry()


__all__ = [
    "Skill",
    "SkillSearchResult",
    "SKILL_LIBRARY",
    "search_skills"
]
