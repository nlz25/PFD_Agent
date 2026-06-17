from __future__ import annotations

import importlib
import sys
from pathlib import Path


def test_default_kdg_db_path_points_to_repo_local_adk(monkeypatch) -> None:
    monkeypatch.delenv("KDG_DB_PATH", raising=False)
    sys.modules.pop("src.matcreator.constants", None)
    sys.modules.pop("src.matcreator.config", None)

    constants = importlib.import_module("src.matcreator.constants")

    expected = Path.home() / ".matcreator" / ".adk" / "know_do_graph.db"

    assert constants.DEFAULT_KDG_DB_PATH == expected
    assert constants.KNOW_DO_GRAPH_DB == expected


def test_repo_local_kdg_db_is_treated_as_legacy_source(monkeypatch) -> None:
    monkeypatch.delenv("KDG_DB_PATH", raising=False)
    sys.modules.pop("src.matcreator.constants", None)
    sys.modules.pop("src.matcreator.config", None)

    constants = importlib.import_module("src.matcreator.constants")

    assert constants.LEGACY_UNIFIED_GRAPH_DB == (
        Path(__file__).resolve().parents[1] / "agents" / "MatCreator" / ".adk" / "know_do_graph.db"
    )


def test_legacy_minimax_env_is_normalized(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("KDG_DB_PATH", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.setenv("MINIMAX_API_KEY", "legacy-key")
    monkeypatch.setenv("MINIMAX_API_BASE", "https://legacy.example/v1")
    sys.modules.pop("src.matcreator.constants", None)
    sys.modules.pop("src.matcreator.config", None)

    constants = importlib.import_module("src.matcreator.constants")

    assert constants.LLM_API_KEY == "legacy-key"
    assert constants.LLM_BASE_URL == "https://legacy.example/v1"


def test_embedding_model_is_forwarded_to_kdg_embedder(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("KDG_DB_PATH", raising=False)
    monkeypatch.delenv("KDG_EMBED_PROVIDER", raising=False)
    monkeypatch.delenv("KDG_EMBED_MODEL", raising=False)
    monkeypatch.delenv("KDG_EMBED_API_KEY", raising=False)
    monkeypatch.delenv("KDG_EMBED_BASE_URL", raising=False)
    monkeypatch.setenv("EMBEDDING_MODEL", "minimax/qwen3-embedding-8b")
    monkeypatch.setenv("LLM_API_KEY", "embed-key")
    monkeypatch.setenv("LLM_BASE_URL", "https://embed.example/v1")
    sys.modules.pop("src.matcreator.constants", None)
    sys.modules.pop("src.matcreator.config", None)

    constants = importlib.import_module("src.matcreator.constants")

    assert constants.os.environ["KDG_EMBED_PROVIDER"] == "openai"
    assert constants.os.environ["KDG_EMBED_MODEL"] == "minimax/qwen3-embedding-8b"
    assert constants.os.environ["KDG_EMBED_API_KEY"] == "embed-key"
    assert constants.os.environ["KDG_EMBED_BASE_URL"] == "https://embed.example/v1"
