"""Markdown-backed workflow skills and search utilities for MatCreator.

Skills are now loaded in standard ADK format via the top-level skill.py
(google.adk.skills.load_skill_from_dir).  This module bridges ADK Skill
objects to the interface expected by the planning/execution agents and keeps
the guide system unchanged.
"""


from google.adk.skills import load_skill_from_dir
from google.adk.tools import skill_toolset
from .workspace import workspace_skills_dir
from pathlib import Path


def _discover_skill_dirs(skills_root: Path) -> list[Path]:
    """Return every directory under *skills_root* that contains ``SKILL.md``.

    This supports both the original one-level layout::

        skills/<name>/SKILL.md

    and nested skill bundles such as::

        skills/mattergen/mattergen_generation/SKILL.md
    """
    if not skills_root.exists():
        return []

    skill_dirs = {
        skill_md.parent
        for skill_md in skills_root.rglob("SKILL.md")
        if skill_md.is_file()
    }
    return sorted(skill_dirs, key=lambda path: path.relative_to(skills_root).as_posix())


def load_skills() -> list:
    """Load all workspace skills discovered by ``SKILL.md`` marker files."""
    skills_root = workspace_skills_dir()
    return [load_skill_from_dir(skill_dir) for skill_dir in _discover_skill_dirs(skills_root)]


ALL_SKILLS = load_skills()

ALL_SKILLS_TOOLSET = skill_toolset.SkillToolset(
    skills=ALL_SKILLS
    )


def refresh_skills() -> dict:
    """Reload all skills from the workspace and update the active toolset.

    Call this after creating or modifying a skill to make it available
    in the current session without restarting.
    """
    new_skills = load_skills()
    ALL_SKILLS.clear()
    ALL_SKILLS.extend(new_skills)
    ALL_SKILLS_TOOLSET._skills = {s.name: s for s in new_skills}
    return {
        "status": "ok",
        "skills": [s.name for s in new_skills],
        "count": len(new_skills),
        "message": f"Refreshed {len(new_skills)} skills.",
    }
