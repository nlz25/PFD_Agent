"""MatCreator agent configuration helpers."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Mapping

_CONFIG_ENV_VAR = "MATCREATOR_CONFIG_PATH"
_DEFAULT_CONFIG_PATH = Path(__file__).with_name("input.json")


def load_settings(config_path: str | os.PathLike[str] | None = None) -> Mapping[str, Any]:
	"""Load agent settings from JSON, falling back to env vars and sensible defaults."""

	candidate = config_path or os.environ.get(_CONFIG_ENV_VAR)
	path = Path(candidate) if candidate else _DEFAULT_CONFIG_PATH
	if not path.exists():
		return {}
	with path.open("r", encoding="utf-8") as handle:
		return json.load(handle)


_SETTINGS: Dict[str, Any] = dict(load_settings())

LLM_MODEL: str = _SETTINGS.get("LLM_MODEL", "")
LLM_API_KEY: str = _SETTINGS.get("LLM_API_KEY", "")
LLM_BASE_URL: str = _SETTINGS.get("LLM_BASE_URL", "")
BOHRIUM_USERNAME: str = _SETTINGS.get("BOHRIUM_USERNAME", "")
BOHRIUM_PASSWORD: str = _SETTINGS.get("BOHRIUM_PASSWORD", "")
BOHRIUM_PROJECT_ID: int | str = _SETTINGS.get("BOHRIUM_PROJECT_ID", "")
