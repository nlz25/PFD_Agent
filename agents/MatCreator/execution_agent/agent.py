"""Execution agent for the PlanningExecutionOrchestrator.

Executes a single plan step:
  1. Loads skill context for the step's skill
  2. Follows the injected skill instruction to run domain tools
  3. Summarises the outcome via summarize_agent
  4. Clears the skill context when done
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

from google.adk.agents import LlmAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools.agent_tool import AgentTool
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.function_tool import FunctionTool
from google.adk.tools.tool_context import ToolContext

from ..constants import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL
from ..thinking_agent.agent import load_skill_context, clear_current_skill
from ..thinking_agent.skill import load_guide_content, load_skill_content
from ..thinking_agent.trajectory import append_trajectory_entry
from ..thinking_agent.workspace_tools import (
    read_workspace_file,
    run_bash,
    run_python,
    write_workspace_file,
)
from ..tools import TOOLSETS

logger = logging.getLogger(__name__)

_model_name = os.environ.get("LLM_MODEL", LLM_MODEL)
_model_api_key = os.environ.get("LLM_API_KEY", LLM_API_KEY)
_model_base_url = os.environ.get("LLM_BASE_URL", LLM_BASE_URL)

# ---------------------------------------------------------------------------
# Summarise sub-agent (own instance — no coupling to thinking_agent's instance)
# ---------------------------------------------------------------------------

_SUMMARIZE_TOOL_INSTRUCTION = """
You summarize key outcomes and extract concrete artifacts from the most recent execution step.
Use absolute paths for artifacts.

Session state context:
- goal: {goal}
- plan: {plan}

Output ONLY a JSON object — no markdown fences, no extra text:
{{
  "key_results": "<concise summary of what was produced or learned>",
  "artifacts": ["<absolute path or ID of important generated files>"],
  "concise_summary": "<user-facing one-paragraph summary>"
}}
"""

_summarize_tool_agent = LlmAgent(
    name="exec_summarize_agent",
    model=LiteLlm(
        model=_model_name,
        base_url=_model_base_url,
        api_key=_model_api_key,
    ),
    description=(
        "Records the outcome of a completed execution step: key results, artifact paths, "
        "and a concise user-facing summary. Call after each significant step."
    ),
    instruction=_SUMMARIZE_TOOL_INSTRUCTION,
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
)

# ---------------------------------------------------------------------------
# Instruction
# ---------------------------------------------------------------------------

_EXECUTION_AGENT_INSTRUCTION = """
You are the MatCreator Executor. Your role is to execute a SINGLE plan step precisely.

## Context
- Goal: {goal}
- Plan: {plan}
- Current step: {current_step}
- Active skill: {active_skill}
- Skill instruction: {skill_instruction}

## Execution protocol
1. Read `current_step` carefully — note its `skill` and `action` fields.
2. Call `load_skill_context(skill_name)` using the skill from the current step.
3. Follow the injected `skill_instruction` to execute the step action using the available tools.
4. When the step is complete, call `exec_summarize_agent` to record the outcome.
5. Call `clear_current_skill` to release the skill context.

## Rules
- Execute ONLY the current step — do not attempt subsequent steps.
- If the required skill is not found, report the error and call `to_planner` with the reason.
- If execution fails unrecoverably, call `to_planner` with a clear reason before stopping.
- Always include absolute file paths in your summary.
- NEVER run code without user approval (already given when execution phase started).
"""

# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------


def to_planner(reason: str, tool_context: ToolContext) -> dict:
    """Signal the orchestrator to abort the current execution loop and return
    to the planning phase.

    Call this when the current step cannot proceed and replanning or user input
    is needed — e.g. a required skill is missing, a prerequisite step failed,
    the original plan is no longer valid, or a key parameter must be confirmed
    / provided by the user before execution can continue.

    Args:
        reason: A short human-readable explanation of why replanning is needed.

    Returns:
        A confirmation dict so the LLM knows the signal was recorded.
    """
    tool_context.state["return_to_planner"] = True
    tool_context.state["return_to_planner_reason"] = reason
    logger.info("[execution_agent] to_planner called — reason: %s", reason)
    return {"status": "ok", "message": f"Returning to planner: {reason}"}


def _exec_before_agent_callback(callback_context: CallbackContext) -> None:
    """Initialise execution-phase state keys if absent."""
    for key, default in [
        ("goal", None),
        ("plan", None),
        ("current_step", {}),
        ("active_skill", None),
        ("skill_instruction", None),
        ("summarize", None),
        ("trajectory_step", 0),
        ("return_to_planner", False),
        ("return_to_planner_reason", None),
    ]:
        if key not in callback_context.state:
            callback_context.state[key] = default


def _exec_after_tool_callback(
    tool: BaseTool,
    args: Dict[str, Any],
    tool_context: ToolContext,
    tool_response: Dict,
) -> Optional[Dict]:
    """Persist summarize result and append trajectory entry."""
    if tool.name == "exec_summarize_agent":
        tool_context.state["summarize"] = tool_response
        try:
            session_id = tool_context._invocation_context.session.id
            step_index = (tool_context.state.get("trajectory_step") or 0) + 1
            tool_context.state["trajectory_step"] = step_index
            log_path = append_trajectory_entry(
                session_id=session_id,
                step_index=step_index,
                goal=tool_context.state.get("goal"),
                active_skill=tool_context.state.get("active_skill"),
                summarize_response=tool_response,
            )
            logger.info("Trajectory entry %d written to %s", step_index, log_path)
        except Exception as exc:
            logger.warning("Failed to write trajectory entry: %s", exc)
    return None


# ---------------------------------------------------------------------------
# Agent instance
# ---------------------------------------------------------------------------

execution_agent = LlmAgent(
    name="execution_agent",
    model=LiteLlm(
        model=_model_name,
        base_url=_model_base_url,
        api_key=_model_api_key,
    ),
    description=(
        "Executes a single plan step by loading the relevant skill context and running the "
        "appropriate domain tools. Called by the orchestrator once per step in sequence."
    ),
    instruction=_EXECUTION_AGENT_INSTRUCTION,
    tools=[
        FunctionTool(load_skill_context),
        FunctionTool(clear_current_skill),
        FunctionTool(to_planner),
        AgentTool(_summarize_tool_agent),
        #FunctionTool(load_guide_content),
        #FunctionTool(load_skill_content),
        #FunctionTool(write_workspace_file),
        #FunctionTool(read_workspace_file),
        FunctionTool(run_python),
        FunctionTool(run_bash),
        *TOOLSETS,
    ],
    before_agent_callback=_exec_before_agent_callback,
    after_tool_callback=_exec_after_tool_callback,
)
