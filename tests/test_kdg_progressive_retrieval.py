from __future__ import annotations

from know_do_graph import (
    EdgeRelation,
    EntryMetadata,
    EntryType,
    KnowDoGraph,
    SkillLevel,
    VerificationStatus,
)

from agents.MatCreator.knowledge import query, review


def _add(
    graph: KnowDoGraph,
    title: str,
    entry_type: EntryType,
    level: SkillLevel,
    *,
    tags: list[str] | None = None,
):
    return graph.add(
        title,
        content=f"{title} content",
        entry_type=entry_type,
        tags=tags or [],
        metadata=EntryMetadata(skill_level=level),
    )


def test_search_skill_context_only_returns_attached_sidecars(
    tmp_path, monkeypatch
) -> None:
    graph = KnowDoGraph(tmp_path / "know-do.db")
    selected = _add(
        graph,
        "Selected capability",
        EntryType.capability,
        SkillLevel.L1,
        tags=["matcreator-skill"],
    )
    other = _add(
        graph,
        "Other capability",
        EntryType.capability,
        SkillLevel.L1,
        tags=["matcreator-skill"],
    )
    attached_heuristic = _add(
        graph,
        "Attached heuristic",
        EntryType.heuristic,
        SkillLevel.L3,
    )
    attached_constraint = _add(
        graph,
        "Attached constraint",
        EntryType.constraint,
        SkillLevel.L4,
    )
    unrelated = _add(
        graph,
        "Unrelated heuristic",
        EntryType.heuristic,
        SkillLevel.L3,
    )
    graph.connect(
        attached_heuristic.id,
        selected.id,
        relation=EdgeRelation.heuristic_for,
    )
    graph.connect(
        attached_constraint.id,
        selected.id,
        relation=EdgeRelation.constraint_on,
    )
    graph.connect(unrelated.id, other.id, relation=EdgeRelation.heuristic_for)
    monkeypatch.setattr(query, "_get_kg", lambda: graph)

    result = query.search_skill_context(selected.id)

    assert "Attached heuristic" in result
    assert "Attached constraint" in result
    assert "Unrelated heuristic" not in result


class _ReviewerSession:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def review_nodes(self, instructions: str = "") -> dict:
        self.calls.append(("review_graph", instructions))
        return {"status": "completed", "summary": "graph reviewed"}

    def review_memory(
        self,
        *,
        session_id: str | None = None,
        instructions: str = "",
    ) -> dict:
        self.calls.append(("review_memory", session_id, instructions))
        return {"status": "completed", "summary": "memory reviewed"}


class _RecordingGraph:
    def __init__(self) -> None:
        self.options: dict | None = None
        self.session = _ReviewerSession()
        self.refreshed = False

    def chat(self, **options):
        self.options = options
        return self.session

    def refresh(self):
        self.refreshed = True
        return {}


def test_graph_agent_tool_routes_through_protected_reviewer_policy(
    monkeypatch,
) -> None:
    graph = _RecordingGraph()
    monkeypatch.setattr(query, "_get_kg", lambda: graph)
    monkeypatch.setenv("LLM_API_KEY", "test-key")

    result = review.talk_to_knowledge_graph_agent(
        "review_graph",
        instructions="Focus on duplicate unverified nodes.",
        batch_size=3,
    )

    assert result["status"] == "completed"
    assert graph.options is not None
    assert graph.options["agent"] == "reviewer"
    assert graph.options["batch_size"] == 3
    assert graph.options["policy"].protected_statuses == frozenset(
        {
            VerificationStatus.peer_reviewed,
            VerificationStatus.community_tested,
        }
    )
    assert graph.session.calls == [
        ("review_graph", "Focus on duplicate unverified nodes.")
    ]
    assert graph.refreshed is True


def test_graph_agent_tool_scopes_memory_review_to_current_session(
    monkeypatch,
) -> None:
    graph = _RecordingGraph()
    monkeypatch.setattr(query, "_get_kg", lambda: graph)
    monkeypatch.setenv("LLM_API_KEY", "test-key")

    result = review.talk_to_knowledge_graph_agent(
        "review_memory",
        instructions="Distill only reusable lessons.",
        session_id="session-123",
    )

    assert result["status"] == "completed"
    assert graph.session.calls == [
        ("review_memory", "session-123", "Distill only reusable lessons.")
    ]
