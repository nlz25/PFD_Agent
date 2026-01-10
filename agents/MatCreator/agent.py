from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.agents.callback_context import CallbackContext
import os
from .pfd_agent.agent import pfd_agent
from .database_agent.agent import database_agent
from .abacus_agent.agent import abacus_agent
from .dpa_agent.agent import dpa_agent
from .vasp_agent.agent import vasp_agent
from .structure_agent.agent import structure_agent
from .constants import LLM_MODEL, LLM_API_KEY, LLM_BASE_URL
from .callbacks import (
    before_agent_callback,
    set_session_metadata,
    get_session_context
)

model_name = os.environ.get("LLM_MODEL", LLM_MODEL)
model_api_key = os.environ.get("LLM_API_KEY", LLM_API_KEY)
model_base_url = os.environ.get("LLM_BASE_URL", LLM_BASE_URL)

description="""
You are the MatCreator Agent. You route user intents to the right capability: either plan and direct sub-agents
for simple tasks, or utilize specialized coordinator agents for certain complex workflows.
""" 

global_instruction = """
General rules for all agents:
- Only call tools that are available in your current context.
- Keep responses concise; include key artifacts with absolute paths and relevant metrics.
- When encountering errors, quote the exact error message and propose concrete solutions.
"""

instruction ="""
Routing logic
- Use `set_session_metadata` to record/update user goals, and relevant context. Update if needed.
- Simple, specific tasks, orchestrate and directly TRANSFER to the matching sub-agent: database_agent | abacus_agent | dpa_agent |vasp_agent
- For complex, multi-stage workflows, delegate to specialized coordinator agent if available. 
 
You have one specialized coordinator agent:
1. 'pfd_agent': Handles complex, multi-stage PFD workflows (mix of MD exploration, configuration filtering, labeling and model training).

Planning and execution rules (must follow)
1. Always make a minimal plan (1–3 bullets) before executing calculation tasks.
2. ALWAYS seek explicit user confirmation before delegating complex workflows to coordinator agents (e.g., pfd-agent).
3. Never mix tool calls from different sub-agents in the same step; each execution transfers to one agent only.
4. For coordinator (e.g., pfd_agent) transfers: mention that session metadata will be created/updated and that detailed step-by-step planning happens inside that specialized agent.
5. Review session context with 'get_session_context' when resuming disrupted workflows to understand what has already been completed.

Outputs
- Always surface absolute artifact paths, key metrics (ids count, entropy gain, energies, model/log paths).
- After coordinator steps: summarize the step’s outputs and the next planned phase.

Errors & blocking inputs
- If required inputs (db path, structure file, model path, config) are missing: ask exactly one
    concise question, then proceed.
- On failure: quote the error, impact, and offer a concrete adjustment (smaller limit, different head, fix path).

Response format (strict)
- Plan: 1–3 bullets (intent + rationale).
- Action: Immediately transfer control to the appropriate agent by invoking it. Do NOT just write text about transferring.
- Result: (after agent returns) concise artifacts + metrics (absolute paths).
- Next: immediate follow-up step or final recap.

IMPORTANT: To transfer to a sub-agent, you must actually invoke/call that agent - do not just mention the agent name in text.
Never fabricate agent or tool names. Always transfer to agents for actions.
"""

def before_agent_callback_root(callback_context: CallbackContext):
    """Set environment variables and initialize session metadata for MatCreator agent."""
    session_id = callback_context._invocation_context.session.id
    user_id = callback_context._invocation_context.session.user_id
    app_name = callback_context._invocation_context.session.app_name
    
    # Set environment variables for session context
    os.environ["CURRENT_SESSION_ID"] = session_id
    os.environ["CURRENT_USER_ID"] = user_id
    os.environ["CURRENT_APP_NAME"] = app_name
    
    # Initialize session metadata in database if not already exists
    try:
        from .callbacks import get_session_metadata
        existing_metadata = get_session_metadata(session_id)
        if not existing_metadata:
            set_session_metadata(
                session_id=session_id,
                additional_metadata={"initialized_by": "root_agent"}
            )
    except Exception as e:
        print(f"Warning: Failed to initialize session metadata: {e}")
    
    return None


tools = [
    set_session_metadata, 
    get_session_context
    ]

root_agent = LlmAgent(
    name='MatCreator_agent',
    model=LiteLlm(
        model=model_name,
        base_url=model_base_url,
        api_key=model_api_key
    ),
    description=description,
    instruction=instruction,
    global_instruction=global_instruction,
    before_agent_callback=before_agent_callback_root,
    tools=tools,
    sub_agents=[
        pfd_agent,
        database_agent,
        abacus_agent,
        dpa_agent,
        vasp_agent,
        structure_agent
    ]
    )