"""Thinking agent - orchestrates goal classification, plan creation, and approval gating.

Architecture mirrors database_agent / sql_agent:
  ThinkingAgent (this file)
    └─ plan_builder_agent  (thinking_agent/planning_agent/agent.py)
    └─ assessment_agent     (../assessment_agent.py)
        └─ summarize_agent      (thinking_agent/summarize_agent/agent.py)
"""

from __future__ import annotations

import os
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.models import llm_response
from google.adk.tools.function_tool import FunctionTool
from google.adk.tools.agent_tool import AgentTool
from google.adk.tools.tool_context import ToolContext
from google.adk.agents.callback_context import CallbackContext
from google.adk.tools.base_tool import BaseTool
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

from ..constants import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL
from ..prompts.subagents import format_subagent_descriptions
from .planning_agent.agent import plan_builder_agent
from .skill import _load_skill_registry
from .memory import load_memory
from .memory import update_memory
from ..constants import _KNOWLEDGE_PATH

def _load_glossary() -> str:
    """Load domain glossary from skills/glossary.md for injection into agent prompts."""
    glossary_path =  _KNOWLEDGE_PATH / "glossary.md"
    try:
        with open(os.path.abspath(glossary_path), "r") as f:
            return f.read()
    except FileNotFoundError:
        return ""

_DOMAIN_GLOSSARY = _load_glossary()

_model_name = os.environ.get("LLM_MODEL", LLM_MODEL)
_model_api_key = os.environ.get("LLM_API_KEY", LLM_API_KEY)
_model_base_url = os.environ.get("LLM_BASE_URL", LLM_BASE_URL)

# ---------------------------------------------------------------------------
# Workflow classification schema (tool agent)
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

class ApprovalAssessment(BaseModel):
    """Assessment of user's response to a presented goal/plan/decision."""
    
    approved: bool = Field(
        ...,
        description=(
            "True if user explicitly approves (yes, ok, proceed, looks good, etc.). "
            "False if user rejects, wants changes, or is uncertain."
        )
    )

_CLASSIFICATION_INSTRUCTION = f"""
You are an agent that determine user goal. 

## Task
- Infer the user's goal as one concise sentence using the correct domain terminology above.
- Provide short reasoning explaining which workflow type applies.

## Rule
Output in JSON format
"""

_APPROVAL_INSTRUCTIION ="""
You are the approveal agent that infers user approval for execution plan.

Goal: {goal}
Plan: {plan}
Execution summarize: {summarize}
 
Task:
Analyze the user's response and determine:
Did they approve the plan?
Do they want modifications?
Do they need clarification?

Output in json format
"""

_THINKING_INSTRUCTION = """
You are the central brain for planning and supervising the computational materials tasks.

You orchestrate planning through tool sub-agents:
- intent_tool_agent              : determine user's goal
- plan_builder_agent             : drafts and update ExecutionPlan
- approval_execution             : ask user permission to proceed to Execution
- update_memory(new_entries)     : appends new knowledge to MEMORY.md

You keep track of these state to decide what to do:
- Goal: {goal} 
- Execution plan: {plan}
- Execution summarize: {summarize}

Approval gate:
- Always check with user after running `intent_tool_agent`.
- Call `approval_execution` before moving to execution.
- If the user requests changes or asks questions, remain in thinking phase.
"""

# ---------------------------------------------------------------------------
# Sub-agents used as tools
# ---------------------------------------------------------------------------

intent_tool_agent = LlmAgent(
    name="intent_tool_agent",
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

assessment_tool_agent = LlmAgent(
    name="check_approval",
    model=LiteLlm(
        model=_model_name,
        base_url=_model_base_url,
        api_key=_model_api_key,
    ),
    description="Check user approval for EXECUTION. REQUIRE explicit user approval to use this tool",
    instruction=_APPROVAL_INSTRUCTIION,
    output_schema=ApprovalAssessment,
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
)

# before_agent_callback
def before_agent_callback(callback_context: CallbackContext):
    """Set environment variables and initialize session state for MatCreator agent."""
    callback_context.state['approval'] = False
    callback_context.state.setdefault("summarize",default=None)
    return None

def before_tool_callback(
    tool: BaseTool,
    args: dict[str, Any],
    tool_context: ToolContext,
) -> Optional[dict]:
    """Inject state variables before specific agent tools are called."""
    if tool.name == "plan_builder_agent":
        tool_context.state["agents"] = format_subagent_descriptions()
        tool_context.state["memory"] = load_memory()

    if tool.name == "intent_tool_agent":
        tool_context.state["memory"] = load_memory()

    if tool.name == "plan_builder_agent":
        registry = _load_skill_registry()
        lines = []
        for skill in registry.values():
            #tags_str = ", ".join(skill.tags) if skill.tags else "none"
            lines.append(f"- {skill.name}: {skill.description} - Instruction {skill.instruction})")
        skills_text = "\n\n".join(lines) if lines else "No skills available."
        tool_context.state["skills"] = skills_text
    return None  # always return None to let the tool proceed normally

# After tool modifier
def after_tool_modifier(
    tool: BaseTool, args: Dict[str, Any], tool_context: ToolContext, tool_response: Dict
) -> Optional[Dict]:
    """Inspects/modifies the tool result after execution."""
    agent_name = tool_context.agent_name
    tool_name = tool.name
    print(f"[Callback] After tool call for tool '{tool_name}' in agent '{agent_name}'")
    print(f"[Callback] Args used: {args}")
    print(f"[Callback] Original tool_response: {tool_response}")

    # --- Modification Example ---
    # If the tool was 'get_capital_city' and result is 'Washington, D.C.'
    if tool_name == 'intent_tool_agent':
        # schema output is already in dict format
        tool_context.state['goal'] = tool_response.get('goal')

    elif tool_name == 'skill_search_tool_agent':
        selected_skills = tool_response.get('selected_skills',[])
        registry = _load_skill_registry()
        for name in selected_skills:
            if skill:=registry.get(name):
                lines=[]
                tags_str = ", ".join(skill.tags) if skill.tags else "none"
                lines.append(f"- {skill.name}: {skill.description} (tags: {tags_str})")
        skills_text = "\n".join(lines) if lines else "No skills available."
        tool_context.state["skills"] = skills_text
        
    elif tool_name == 'plan_builder_agent':
        tool_context.state['plan'] = tool_response
        tool_context.state['detailed_steps'] = tool_response.get('steps')
        tool_context.state['agents_needed'] = list({
            step['agent'] for step in (tool_response.get('steps') or [])
            if isinstance(step, dict) and 'agent' in step
        })
        
    
    elif tool_name == 'summarize_agent':
        tool_context.state['summarize'] = tool_response

    return None

# After model modfier
def after_model_modifier(
    callback_context: CallbackContext, 
    llm_response: llm_response
) -> Optional[Dict]:
    """Inspects/modifies the tool result after execution."""
    
    k_list=["phase"]
    for k in k_list:
        print(f"[Callback] The current states are {k}:{callback_context.state.get(k)}")
    return None

# ---------------------------------------------------------------------------
# ThinkingAgent instance
# ---------------------------------------------------------------------------

def approval_execution(tool_context: ToolContext) -> dict:
    """Transition to execution phase. Only call this AFTER the user has explicitly
    approved the plan in natural language (e.g. 'yes', 'ok', 'proceed', 'looks good').
    Do NOT call this if the user is still asking questions or requesting changes."""
    #"""Call this function to ask for user approval for execution"""
    tool_context.state["approval"] = True
    return {
        "status": "ok",
        "message": "Execution approved",
    }

thinking_agent = LlmAgent(
    name="thinking_agent",
    model=LiteLlm(
        model=_model_name,
        base_url=_model_base_url,
        api_key=_model_api_key,
    ),
    description=(
        "Thinking-phase orchestrator. Classifies workflow, creates structured execution plans, "
        "and gates transition to execution on explicit user approval."
    ),
    instruction=_THINKING_INSTRUCTION,
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
    tools=[
        AgentTool(intent_tool_agent),
        AgentTool(plan_builder_agent),
        update_memory,
        FunctionTool(approval_execution,
                     )
    ],
    before_agent_callback=before_agent_callback,
    before_tool_callback=before_tool_callback,
    after_tool_callback=after_tool_modifier,
    after_model_callback=after_model_modifier
)
