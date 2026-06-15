from __future__ import annotations

from pathlib import Path

from google.adk.skills import load_skill_from_dir
from know_do_graph import KnowDoGraph

from agents.MatCreator import guide, skill
from agents.MatCreator.knowledge import query


def _write_skill(skill_dir: Path) -> None:
    (skill_dir / "references").mkdir(parents=True)
    (skill_dir / "assets").mkdir(parents=True)
    (skill_dir / "scripts").mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        """---
name: demo-skill
description: Short summary only.
tools: []
dependent_skills: []
---
Use the detailed SKILL instructions here.

- Step 1: prepare inputs
- Step 2: validate outputs
""",
        encoding="utf-8",
    )
    (skill_dir / "references" / "tips.md").write_text(
        "Reference guidance for tricky edge cases.",
        encoding="utf-8",
    )
    (skill_dir / "assets" / "example.md").write_text(
        "Worked example content for the skill.",
        encoding="utf-8",
    )
    (skill_dir / "scripts" / "tool.py").write_text(
        "print('skill script')\n",
        encoding="utf-8",
    )
    (skill_dir / "README.md").write_text(
        "Extra README context for operators.",
        encoding="utf-8",
    )


def test_seed_skills_to_graph_stores_full_skill_body_and_native_attachments(
    tmp_path, monkeypatch
) -> None:
    skills_root = tmp_path / "skills"
    skill_dir = skills_root / "demo-skill"
    _write_skill(skill_dir)

    loaded_skill = load_skill_from_dir(skill_dir)
    graph = KnowDoGraph(tmp_path / "know-do.db")

    monkeypatch.setattr(skill, "ALL_SKILLS", [loaded_skill])
    monkeypatch.setattr(skill, "_MODULE_SKILLS_ROOT", tmp_path / "empty-defaults")
    monkeypatch.setattr(skill, "workspace_skills_dir", lambda: skills_root)
    monkeypatch.setattr(guide, "ALL_GUIDES", [])
    monkeypatch.setattr(query, "_get_kg", lambda: graph)

    result = skill.seed_skills_to_graph()

    assert result["seeded"] == 1
    assert result["attachments_seeded"] == 6

    matches = graph.search("demo-skill", tags=["matcreator-skill"], limit=5, mode="keyword")
    skill_entry = next(entry for entry in matches if entry.title == "demo-skill")
    assert "Use the detailed SKILL instructions here." in skill_entry.content
    assert "Short summary only." not in skill_entry.content

    assert skill_entry.internal_refs == ["references/tips.md", "README.md"]
    assert {(asset.folder, asset.filename) for asset in skill_entry.assets} == {
        ("references", "tips.md"),
        ("assets", "example.md"),
        ("scripts", "tool.py"),
        ("docs", "README.md"),
    }
    assert [script.filename for script in skill_entry.scripts] == ["tool.py"]

    context = query.search_skill_context(skill_entry.id, query="README", top_k=10)
    assert "docs/README.md" in context
    assert "Extra README context for operators." in context
    assert "scripts/tool.py" not in context
