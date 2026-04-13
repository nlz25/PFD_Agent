"""Tester agent: creates, tests, and validates skills.

Responsibilities:
  1. Creating new skills from scratch (YAML frontmatter + instruction body)
  2. Writing self-contained test code for each skill
  3. Running tests and validating output quality
  4. Updating skills based on test outcomes (up to 3 retry attempts)
"""

from __future__ import annotations

import logging
import os

from google.adk.agents import LlmAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools.function_tool import FunctionTool

from ..constants import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL
from ..thinking_agent.skill import list_skill_name_descriptions, load_skill_content
from ..thinking_agent.workspace_tools import (
    create_skill,
    list_workspace_skills,
    read_workspace_file,
    run_bash,
    run_python,
    write_workspace_file,
)

logger = logging.getLogger(__name__)

_model_name = os.environ.get("LLM_MODEL", LLM_MODEL)
_model_api_key = os.environ.get("LLM_API_KEY", LLM_API_KEY)
_model_base_url = os.environ.get("LLM_BASE_URL", LLM_BASE_URL)

# ---------------------------------------------------------------------------
# Instruction
# ---------------------------------------------------------------------------

_TESTER_AGENT_INSTRUCTION = """
You are the MatCreator Skill Tester. Your job is to create, test, and validate skills.

## Context
- Goal: {goal}
- Testing request: {tester_request}

## Workflow

### Creating a new skill
1. Call `list_skill_name_descriptions` to check if the skill already exists.
2. Design the skill: YAML frontmatter (name, description, tools, dependent_skills) + instruction body.
3. Call `create_skill(name, description, instruction)` to scaffold the skill file.
4. Write a small self-contained Python test to `skills/<name>/test_<name>.py` using `write_workspace_file`.
5. Run the test with `run_python` or `run_bash` and inspect the output.
6. If the test passes, report success with the skill file path.
7. If the test fails, update the skill instruction with `write_workspace_file` and retry (max 3 attempts).

### Testing an existing skill
1. Call `load_skill_content(skill_name)` to read the current instruction.
2. Write a targeted test script that exercises the skill's key operations.
3. Run the test and analyse the output.
4. Report a clear pass/fail verdict with supporting evidence.

## Skill file format
```
---
name: <snake_case_name>
description: <one-sentence description for the planner>
tools: [<tool1>, <tool2>]
dependent_skills: []
---
<Markdown instruction body>
```

## Rules
- Skills must use snake_case names with no spaces.
- Test code must be self-contained and reproducible.
- Always report the absolute file path of created or modified skills.
- Keep skill instructions concise and tool-specific.
- Do not overwrite an existing skill — use `write_workspace_file` to update the `.md` file directly.
"""

# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------


def _tester_before_agent_callback(callback_context: CallbackContext) -> None:
    """Ensure tester_request is present in state."""
    if "tester_request" not in callback_context.state:
        callback_context.state["tester_request"] = ""


# ---------------------------------------------------------------------------
# Agent instance
# ---------------------------------------------------------------------------

tester_agent = LlmAgent(
    name="tester_agent",
    model=LiteLlm(
        model=_model_name,
        base_url=_model_base_url,
        api_key=_model_api_key,
    ),
    description=(
        "Creates new skills from scratch, writes test code, executes tests, and validates "
        "that skills work correctly. Invoked when a skill needs to be developed or validated."
    ),
    instruction=_TESTER_AGENT_INSTRUCTION,
    tools=[
        FunctionTool(list_skill_name_descriptions),
        FunctionTool(load_skill_content),
        FunctionTool(create_skill),
        FunctionTool(write_workspace_file),
        FunctionTool(read_workspace_file),
        FunctionTool(list_workspace_skills),
        FunctionTool(run_python),
        FunctionTool(run_bash),
    ],
    before_agent_callback=_tester_before_agent_callback,
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
)
