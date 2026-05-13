from __future__ import annotations

import os
import logging

from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools.function_tool import FunctionTool
from google.adk.tools.tool_context import ToolContext
from google.adk.agents.callback_context import CallbackContext

from ...constants import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL
from .planning import validate_plan
from .intent import validate_intent
from .summarize import validate_summarize
from .session_summary import write_session_summary
from ...skill import ALL_SKILLS, ALL_SKILLS_TOOLSET, refresh_skills
from ...guide import ALL_GUIDES
from .memory import query_knowledge_graph, save_to_knowledge_graph, update_memory, read_memory
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
        #"instruction": selected.instructions,
        #"needed_tools": needed_tools,
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


def resume_execution(tool_context: ToolContext) -> dict:
    """Resume execution from the current saved step index.

    Call this when the user explicitly requests to continue execution after an interruption.
    """
    plan = tool_context.state.get("plan")
    if not plan:
        return {
            "status": "error",
            "message": "No plan found in session state. Create/validate a plan first.",
        }

    current_step_index = tool_context.state.get("current_step_index", 0)
    if not isinstance(current_step_index, int) or current_step_index < 0:
        current_step_index = 0

    tool_context.state["return_to_planner"] = False
    tool_context.state["return_to_planner_reason"] = None
    tool_context.state["execution_approved"] = True
    tool_context.state["current_step_index"] = current_step_index

    return {
        "status": "ok",
        "message": (
            "Execution resume approved. "
            f"The orchestrator will continue from step index {current_step_index}."
        ),
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
# Instruction
# ---------------------------------------------------------------------------

_MATCREATOR_INSTRUCTION = """
You are MatCreator, an AI assistant for computational materials science workflows.
Your role here is **PLANNING ONLY** — a dedicated execution agent handles the actual steps.

## Context
- Available skills: {skills}
- Available guides: {guides}
- Goal: {goal}
- Plan: {plan}
- Summarize: {summarize}

## Default workflow
1. Determine the user's goal, then call `validate_intent` with your interpretation.
   Call `query_knowledge_graph` with the user's goal to retrieve relevant past knowledge,
   lessons, and related skills. Only fall back to `read_memory` when a broad context dump
   is explicitly needed.
2. If the user's goal matches one of the Available guides, call `load_guide` before planning.
3. Always draft an execution plan, then call `validate_plan` to validate and commit it. Show the plan to the user in Markdown table format.
{confirmation_instruction}
5. If the user asks to create or test a skill, call `request_skill_testing(description)`.
6. After completing a step, use `save_to_knowledge_graph` to persist key lessons or findings.
7. Once execution has fully completed and results are available, call `write_session_summary`
   with the global narrative: original goal, approach rationale, key decisions, lessons
   learned, any failed attempts, and the overall outcome.

## Rules
- NEVER execute plan steps.
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
# MatCreator agent instance
# ---------------------------------------------------------------------------


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
        FunctionTool(validate_intent),
        FunctionTool(validate_summarize),
        FunctionTool(write_session_summary),
        FunctionTool(confirm_plan_and_start_execution),
        FunctionTool(resume_execution),
        FunctionTool(request_skill_testing),
        FunctionTool(load_guide),
        FunctionTool(query_knowledge_graph),
        FunctionTool(save_to_knowledge_graph),
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
)
