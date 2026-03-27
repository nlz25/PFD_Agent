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

from ..constants import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL
from .planning_agent.agent import plan_builder_agent
from .skill import (
    list_skill_name_descriptions,
    list_guide_metadata,
    load_guide_content,
    load_skill_content,
    _load_skill_registry,
    Skill,
)
from .memory import load_memory, update_memory
from .workspace_tools import (
    write_workspace_file,
    read_workspace_file,
    list_workspace_skills,
    create_skill,
    run_python,
    run_bash,
    run_python_file,
    init_workspace_tool,
)
from ..tools import TOOLSETS

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
    skill_registry = _load_skill_registry()
    normalized = (skill_name or "").strip()
    if not normalized:
        return {
            "status": "error",
            "message": "skill_name is required.",
            "available_skills": sorted(skill_registry.keys()),
        }

    selected: Skill | None = skill_registry.get(normalized)
    if selected is None:
        lowered = normalized.lower()
        for name, skill in skill_registry.items():
            if name.lower() == lowered:
                selected = skill
                break

    if selected is None:
        return {
            "status": "error",
            "message": f"Skill '{skill_name}' not found.",
            "available_skills": sorted(skill_registry.keys()),
        }

    tool_context.state["active_skill"] = selected.name
    tool_context.state["skill_instruction"] = selected.instruction

    return {
        "status": "ok",
        "skill": selected.name,
        "instruction": selected.instruction,
        "needed_tools": selected.needed_tools,
        "message": f"Loaded skill context for '{selected.name}'.",
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
class WorkflowClassification(BaseModel):
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

_CLASSIFICATION_INSTRUCTION = f"""
You are an agent that determine user goal. 

## Task
- Infer the user's goal as one concise sentence using the correct domain terminology above.
- Provide short reasoning explaining which workflow type applies.

## Rule
Output in JSON format
"""

intent_tool_agent = LlmAgent(
    name="user_intent",
    model=LiteLlm(
        model=_model_name,
        base_url=_model_base_url,
        api_key=_model_api_key,
    ),
    description="Determine user's goal.",
    instruction=_CLASSIFICATION_INSTRUCTION,
    output_schema=WorkflowClassification,
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
)


# ---------------------------------------------------------------------------
# Instruction
# ---------------------------------------------------------------------------

_MATCREATOR_INSTRUCTION = """
You are MatCreator, an AI assistant for computational materials science workflows.

## Context
- Memory: {memory}
- Available skills: {skills}
- Available guides: {guides}
- Goal: {goal}
- Plan: {plan}
- Active skill: {active_skill}
- Skill instruction: {skill_instruction}
- Summarize: {summarize}

## Default workflow
1. Understand the user's goal by `user_intent`. Ask clarifying questions if needed.
2. Use `plan_builder_agent` to draft a clear plan. Show it to the user.
3. **Before executing the plan**, always wait for explicit confirmation (e.g. "yes", "ok", "proceed").
4. For each plan step, call `load_skill_context(skill_name)` first, then follow the
   injected skill instruction above.
5. After completing each step, call `summarize_agent` to record outcomes.
6. If a step fails, diagnose, propose a fix, and confirm with the user before retrying.

## Rules
- NEVER run code without explicit user approval.
- Always call `load_skill_context` before executing a domain-specific step.
- Keep responses concise; include key results with absolute paths when relevant.
- When you encounter an error, quote the exact message and propose concrete solutions.
"""

# ---------------------------------------------------------------------------
# before_agent_callback: inject dynamic context into session state
# ---------------------------------------------------------------------------

def before_agent_callback(callback_context: CallbackContext) -> None:
    """Refresh memory, skills, and guides in session state each invocation."""
    callback_context.state["memory"] = load_memory()

    skill_summaries = list_skill_name_descriptions()
    callback_context.state["skills"] = "\n".join(
        f"- {item['name']}: {item['description']}" for item in skill_summaries
    ) if skill_summaries else "No skills available."

    guide_meta = list_guide_metadata()
    callback_context.state["guides"] = "\n".join(
        f"- {g['name']}: {g['description']} (tags: {g['tags']})"
        for g in guide_meta
    ) if guide_meta else "No guides available."

    callback_context.state.setdefault("active_skill", "none")
    callback_context.state.setdefault("skill_instruction", "No skill loaded yet. Call load_skill_context before executing a domain step.")

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
    """Persist plan returned by plan_builder_agent and summarize from summarize_agent."""
    tool_name = tool.name

    if tool_name == "plan_builder_agent":
        if isinstance(tool_response, str):
            import json as _json
            import re as _re
            _m = _re.search(r"\{[\s\S]*\}", tool_response)
            try:
                tool_response = _json.loads(_m.group(0)) if _m else {}
            except _json.JSONDecodeError:
                pass
        tool_context.state["plan"] = tool_response

    elif tool_name == "summarize_agent":
        tool_context.state["summarize"] = tool_response

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
        AgentTool(plan_builder_agent),
        AgentTool(_summarize_tool_agent),
        AgentTool(intent_tool_agent),
        FunctionTool(load_skill_context),
        FunctionTool(load_guide_content),
        FunctionTool(load_skill_content),
        update_memory,
        FunctionTool(init_workspace_tool),
        FunctionTool(list_workspace_skills),
        FunctionTool(create_skill),
        FunctionTool(write_workspace_file),
        FunctionTool(read_workspace_file),
        FunctionTool(run_python),
        FunctionTool(run_bash),
        FunctionTool(run_python_file),
        *TOOLSETS,
    ],
    before_agent_callback=before_agent_callback,
    after_tool_callback=after_tool_callback,
)
