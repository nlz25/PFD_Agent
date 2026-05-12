from __future__ import annotations

import logging
import os

from google.adk.agents import LlmAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools.function_tool import FunctionTool
from google.adk.tools.tool_context import ToolContext

from ...constants import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL
from ...workspace import get_session_workdir
from ..thinking_agent.summarize import validate_summarize
from .step_executor_runner import run_step_executor

logger = logging.getLogger(__name__)

_model_name = os.environ.get("LLM_MODEL", LLM_MODEL)
_model_api_key = os.environ.get("LLM_API_KEY", LLM_API_KEY)
_model_base_url = os.environ.get("LLM_BASE_URL", LLM_BASE_URL)

# ---------------------------------------------------------------------------
# Instruction
# ---------------------------------------------------------------------------

_EXECUTION_ORCHESTRATOR_INSTRUCTION = """
You are the MatCreator Execution Orchestrator. You manage the full execution of all pending plan steps.

## Context
- Goal: {goal}
- Full plan: {plan}
- Resume from step number: {current_step_index} (execute steps with step_number > this value)
- Workspace directory: {workspace_dir}
- Prior trajectory: {summarize}

## Protocol
1. Identify all plan steps where step_number > {current_step_index}.
2. For each pending step, call `run_step_executor` with:
   - `step_number`: the step's number from the plan
   - `action`: the step's action text
   - `skill_name`: the name of the skill from the plan step
   - `workspace_dir`: the workspace directory above
   - `prior_context`: a brief summary of completed steps (from trajectory, if any)
3. **Parallelize independent steps**: if consecutive pending steps use different skills
   and do not share data, issue all their `run_step_executor` calls in a SINGLE response
   so ADK executes them concurrently.
4. After each `run_step_executor` result:
   - `status == "success"`: call `validate_summarize` with key_results, artifacts,
     concise_summary; then call `set_current_step_index` with the completed step_number.
   - `status == "needs_replanning"`: call `to_planner` with the replan_reason and STOP.
     Do not proceed to remaining steps.
5. Continue until all steps complete or `to_planner` is called.

## Rules
- NEVER run code directly — all execution must go through `run_step_executor`.
- Always use absolute file paths in artifact lists.
"""

# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


def to_planner(reason: str, tool_context: ToolContext) -> dict:
    """Signal the main orchestrator to abort execution and return to planning.

    Call when a step cannot proceed and replanning or user input is needed.

    Args:
        reason: Short explanation of why replanning is needed.
    """
    tool_context.state["return_to_planner"] = True
    tool_context.state["return_to_planner_reason"] = reason
    logger.info("[execution_orchestrator] to_planner — reason: %s", reason)
    return {"status": "ok", "message": f"Returning to planner: {reason}"}


def set_current_step_index(step_index: int, tool_context: ToolContext) -> dict:
    """Update the current step index for resumption tracking.

    Call after each successful step with that step's step_number so execution
    can resume from the correct point if interrupted later.

    Args:
        step_index: The 1-based step_number that just completed successfully.
    """
    tool_context.state["current_step_index"] = step_index
    logger.debug("[execution_orchestrator] current_step_index → %d", step_index)
    return {"status": "ok", "current_step_index": step_index}


# ---------------------------------------------------------------------------
# Callback
# ---------------------------------------------------------------------------


def _exec_before_agent_callback(callback_context: CallbackContext) -> None:
    """Initialise execution-phase state before the LLM runs."""
    state = callback_context.state

    for key, default in [
        ("goal", None),
        ("plan", None),
        ("current_step_index", 0),
        ("summarize", None),
        ("trajectory_step", 0),
        ("return_to_planner", False),
        ("return_to_planner_reason", None),
    ]:
        if key not in state:
            state[key] = default

    # Workspace directory (session_id is set by PlanningExecutionOrchestrator on first run)
    session_id = state.get("session_id", "default")
    state["workspace_dir"] = str(get_session_workdir(session_id))


# ---------------------------------------------------------------------------
# Agent instance
# ---------------------------------------------------------------------------

execution_agent = LlmAgent(
    name="execution_orchestrator",
    model=LiteLlm(
        model=_model_name,
        base_url=_model_base_url,
        api_key=_model_api_key,
    ),
    include_contents="none",
    description=(
        "Orchestrates execution of all plan steps by spawning isolated step_executor sub-agents. "
        "Handles step sequencing, optional parallelism, and error escalation to the planner."
    ),
    instruction=_EXECUTION_ORCHESTRATOR_INSTRUCTION,
    tools=[
        FunctionTool(run_step_executor),
        FunctionTool(to_planner),
        FunctionTool(validate_summarize),
        FunctionTool(set_current_step_index),
    ],
    before_agent_callback=_exec_before_agent_callback,
)
