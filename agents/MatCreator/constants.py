"""MatCreator agent configuration helpers."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

_script_dir = Path(__file__).parent
load_dotenv(_script_dir / ".env", override=True)

LLM_MODEL: str = os.environ.get("LLM_MODEL", "")
LLM_API_KEY: str = os.environ.get("LLM_API_KEY", "")
LLM_BASE_URL: str = os.environ.get("LLM_BASE_URL", "")
BOHRIUM_USERNAME: str = os.environ.get("BOHRIUM_USERNAME", "")
BOHRIUM_PASSWORD: str = os.environ.get("BOHRIUM_PASSWORD", "")
BOHRIUM_PROJECT_ID: int | str = os.environ.get("BOHRIUM_PROJECT_ID", 00000)
