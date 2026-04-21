"""MatCreator agent - a single LlmAgent that handles both planning and execution.

The agent dynamically loads skill context, runs tools, and manages its own
plan/execution loop in natural conversation. No separate execution agent or
phase state machine is needed.
"""

from __future__ import annotations

import os
import logging
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools.function_tool import FunctionTool
from google.adk.tools.agent_tool import AgentTool
from google.adk.tools.tool_context import ToolContext
from google.adk.agents.callback_context import CallbackContext
from google.adk.tools.base_tool import BaseTool

from ...constants import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL
from .trajectory import append_trajectory_entry
from .planning import validate_plan
from ...skill import ALL_SKILLS, ALL_SKILLS_TOOLSET, refresh_skills
from ...guide import ALL_GUIDES
from .memory import update_memory, read_memory
from ...tools.workspace_tools import (
    init_workspace_tool,
    run_bash,
    run_python
)
from ...tools.util_tools import (
    show_artifact,
    show_plot,
    show_structure
)


logger = logging.getLogger(__name__)

_model_name = os.environ.get("LLM_MODEL", LLM_MODEL)
_model_api_key = os.environ.get("LLM_API_KEY", LLM_API_KEY)
_model_base_url = os.environ.get("LLM_BASE_URL", LLM_BASE_URL)

# ---------------------------------------------------------------------------
# load_skill_context: dynamically injects skill instruction into session state
# ---------------------------------------------------------------------------

def load_skill_context(skill_name: str, tool_context: ToolContext) -> dict:
    """Load the instruction and tool list for a skill into session state.

    Call this BEFORE executing any step that belongs to a specific skill.
    The loaded instruction is injected into the agent's prompt via the
    {active_skill} and {skill_instruction} template variables.

    Args:
        skill_name: Exact skill name as listed in Available skills.
    """
    normalized = (skill_name or "").strip()
    if not normalized:
        return {
            "status": "error",
            "message": "skill_name is required.",
            "available_skills": sorted(s.name for s in ALL_SKILLS),
        }

    selected = next((s for s in ALL_SKILLS if s.name == normalized), None)
    if selected is None:
        lowered = normalized.lower()
        selected = next((s for s in ALL_SKILLS if s.name.lower() == lowered), None)

    if selected is None:
        return {
            "status": "error",
            "message": f"Skill '{skill_name}' not found.",
            "available_skills": sorted(s.name for s in ALL_SKILLS),
        }

    tool_context.state["active_skill"] = selected.name
    tool_context.state["skill_instruction"] = selected.instructions

    needed_tools = selected.frontmatter.metadata.get("tools", [])
    return {
        "status": "ok",
        "skill": selected.name,
        "instruction": selected.instructions,
        "needed_tools": needed_tools,
        "message": f"Loaded skill context for '{selected.name}'.",
    }


def clear_current_skill(tool_context: ToolContext) -> dict:
    """Clear the active skill context from session state.
    Call this after finishing a skill-specific step to avoid stale context.
    """
    tool_context.state["active_skill"] = None
    tool_context.state["skill_instruction"] = None
    return {"status": "ok", "message": "Active skill context cleared."}


def load_guide(guide_name: str) -> dict:
    """Return the full instruction content for a guide.

    Call this before planning when the user's goal matches a known guide.

    Args:
        guide_name: Exact guide name as listed in Available guides.
    """
    normalized = (guide_name or "").strip()
    if not normalized:
        return {
            "status": "error",
            "message": "guide_name is required.",
            "available_guides": sorted(g.name for g in ALL_GUIDES),
        }

    selected = next((g for g in ALL_GUIDES if g.name == normalized), None)
    if selected is None:
        lowered = normalized.lower()
        selected = next((g for g in ALL_GUIDES if g.name.lower() == lowered), None)

    if selected is None:
        return {
            "status": "error",
            "message": f"Guide '{guide_name}' not found.",
            "available_guides": sorted(g.name for g in ALL_GUIDES),
        }

    return {
        "status": "ok",
        "guide": selected.name,
        "instruction": selected.instructions,
    }


def confirm_plan_and_start_execution(tool_context: ToolContext) -> dict:
    """Signal that the user has approved the plan and execution should begin.

    Call this when the user explicitly confirms they want to proceed with the plan
    (e.g. "yes", "proceed", "go ahead").  The orchestrator will then delegate each
    plan step to the execution agent — do NOT execute steps yourself after calling this.
    """
    tool_context.state["execution_approved"] = True
    tool_context.state["current_step_index"] = 0
    return {
        "status": "ok",
        "message": "Execution approved. The orchestrator will now run each step via the execution agent.",
    }


def request_skill_testing(skill_or_description: str, tool_context: ToolContext) -> dict:
    """Request the tester agent to create or validate a skill.

    Call this when the user asks to create a new skill or test an existing one.
    The orchestrator will delegate to the tester agent on the next routing decision.

    Args:
        skill_or_description: Name of an existing skill to test, or a description
            of the new skill to create (e.g. "a skill for VASP band structure calculation").
    """
    tool_context.state["testing_requested"] = True
    tool_context.state["tester_request"] = skill_or_description
    return {
        "status": "ok",
        "message": f"Skill testing requested: {skill_or_description}",
    }


# ---------------------------------------------------------------------------
# Inline summarize_agent tool: records step outcomes into session state
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
    name="summarize_agent",
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
# Intent tool
# ---------------------------------------------------------------------------
class USERINTENT(BaseModel):
    """Classification of workflow type based on user intent."""
    goal: str = Field(
        ...,
        description="Single-sentence articulation of the user's goal/intent",
        max_length=300,
    )
    reasoning: str = Field(
        ...,
        description="Brief explanation of why you think this way",
        #max_length=300,
    )

_INTENT_INSTRUCTION = """
You are an agent that determines the user's goal.

## Task
- Infer the user's goal as one concise sentence using correct domain terminology.
- Provide short reasoning explaining why you interpreted the goal that way.

Output ONLY a JSON object — no markdown fences, no extra text:
{
  "goal": "<single-sentence statement of the user's goal>",
  "reasoning": "<brief explanation of why you interpreted it this way>"
}
"""

intent_tool_agent = LlmAgent(
    name="user_intent",
    model=LiteLlm(
        model=_model_name,
        base_url=_model_base_url,
        api_key=_model_api_key,
    ),
    description="Determine user's goal.",
    instruction=_INTENT_INSTRUCTION,
    output_schema=USERINTENT,
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
)


# ---------------------------------------------------------------------------
# Instruction
# ---------------------------------------------------------------------------

_MATCREATOR_INSTRUCTION = """
You are MatCreator, an AI assistant for computational materials science workflows.
Your role here is **planning only** — a dedicated execution agent handles the actual steps.

## Context
- Available skills: {skills}
- Available guides: {guides}
- Goal: {goal}
- Plan: {plan}
- Summarize: {summarize}

## Default workflow
1. Understand the user's goal with `user_intent`. Call `read_memory` to recall past context.
2. If the user's goal matches one of the Available guides, call `load_guide` to inject its workflow instructions before planning.
3. Draft a clear execution plan yourself, then call `validate_plan` to validate and commit it. Show the plan to the user.
{confirmation_instruction}
5. If the user asks to create or test a skill, call `request_skill_testing(description)`.

## Rules
- NEVER load skill context or run tools to execute plan steps — that is the execution agent's job.
- After `confirm_plan_and_start_execution`, simply inform the user that execution is starting.
- For skill creation/testing requests, always call `request_skill_testing` before responding.
- Keep responses concise; reference absolute file paths where relevant.
- When you encounter an error, quote the exact message and propose concrete solutions.
"""

# ---------------------------------------------------------------------------
# before_agent_callback: inject dynamic context into session state
# ---------------------------------------------------------------------------

def before_agent_callback(callback_context: CallbackContext) -> None:
    """Refresh memory, skills, and guides in session state each invocation."""
    state = callback_context._invocation_context.session.state
    for key, default in [
        ("plan", None),
        ("goal", None),
        ("skills", None),
        ("guides", None),
        ("active_skill", None),
        ("skill_instruction", None),
        ("summarize", None),
        ("trajectory_step", 0),
    ]:
        if key not in state:
            callback_context.state[key] = default
    
    callback_context.state["skills"] = "\n".join(
        f"- {s.name}: {s.description}" for s in ALL_SKILLS
    ) if ALL_SKILLS else "No skills available."

    callback_context.state["guides"] = "\n".join(
        f"- {g.name}: {g.description}" for g in ALL_GUIDES
    ) if ALL_GUIDES else "No guides available."

    if state.get("benchmark_mode", False):
        callback_context.state["confirmation_instruction"] = (
            "4. **Benchmark mode is active.** Immediately call `confirm_plan_and_start_execution` "
            "after `validate_plan` succeeds — do NOT wait for user confirmation."
        )
    else:
        callback_context.state["confirmation_instruction"] = (
            '4. **Wait for explicit user confirmation** (e.g. "yes", "ok", "proceed") before proceeding.\n'
            "   When the user confirms, call `confirm_plan_and_start_execution` — do NOT execute steps yourself."
        )

    return None

# ---------------------------------------------------------------------------
# after_tool_callback: persist plan and summarize updates
# ---------------------------------------------------------------------------

def after_tool_callback(
    tool: BaseTool,
    args: Dict[str, Any],
    tool_context: ToolContext,
    tool_response: Dict,
) -> Optional[Dict]:
    """Persist summarize updates; plan is persisted directly by validate_plan."""
    tool_name = tool.name

    if tool_name == "user_intent":
        if isinstance(tool_response, str):
            import json as _json
            import re as _re
            _m = _re.search(r"\{[\s\S]*\}", tool_response)
            try:
                tool_response = _json.loads(_m.group(0)) if _m else {}
            except _json.JSONDecodeError:
                pass
        if isinstance(tool_response, dict) and "goal" in tool_response:
            tool_context.state["goal"] = tool_response["goal"]

    elif tool_name == "summarize_agent":
        tool_context.state["summarize"] = tool_response

        # --- trajectory logging ---
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
        except Exception as _exc:  # never block the main flow
            logger.warning("Failed to write trajectory entry: %s", _exc)

    return None

# ---------------------------------------------------------------------------
# MatCreator agent instance
# ---------------------------------------------------------------------------
from ...tools.remoteagent_tool import load_remote_a2a_agents

thinking_agent = LlmAgent(
    name="MatCreator",
    model=LiteLlm(
        model=_model_name,
        base_url=_model_base_url,
        api_key=_model_api_key,
    ),
    description=(
        "MatCreator: plans and executes computational materials science workflows "
        "through natural conversation with the user."
    ),
    instruction=_MATCREATOR_INSTRUCTION,
    tools=[
        FunctionTool(validate_plan),
        AgentTool(_summarize_tool_agent),
        AgentTool(intent_tool_agent),
        FunctionTool(confirm_plan_and_start_execution),
        FunctionTool(request_skill_testing),
        FunctionTool(load_guide),
        FunctionTool(read_memory),
        FunctionTool(update_memory),
        FunctionTool(init_workspace_tool),
        FunctionTool(refresh_skills),
        FunctionTool(run_python),
        FunctionTool(run_bash),
        show_artifact,
        show_plot,
        show_structure,
        ALL_SKILLS_TOOLSET
    ],
    before_agent_callback=before_agent_callback,
    after_tool_callback=after_tool_callback,
    sub_agents=load_remote_a2a_agents()
)
