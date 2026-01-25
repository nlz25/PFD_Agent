from google.adk.agents import LlmAgent, InvocationContext
from google.adk.models.lite_llm import LiteLlm
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.remote_a2a_agent import RemoteA2aAgent
from google.adk.events import Event, EventActions
from google.genai.types import Content, Part

import os
import logging
from .planning_agent import planning_agent
from .assessment_agent import assessment_agent
from .execution_agent import execution_agent
from .constants import LLM_MODEL, LLM_API_KEY, LLM_BASE_URL
from .callbacks import (
    set_session_metadata,
    get_session_context,
    get_session_metadata
)

AGENT_CARD_WELL_KNOWN_PATH=".well-known/agent-card.json"

model_name = os.environ.get("LLM_MODEL", LLM_MODEL)
model_api_key = os.environ.get("LLM_API_KEY", LLM_API_KEY)
model_base_url = os.environ.get("LLM_BASE_URL", LLM_BASE_URL)

logger = logging.getLogger(__name__)

description="""
You are the MatCreator Agent. You orchestrate computational materials science workflows through 
a structured plan-confirm-execute cycle, ensuring user visibility and approval before expensive operations.
""" 

global_instruction = """
General rules for all agents:
- Only call tools that are available in your current context.
- Keep responses concise; include key artifacts with absolute paths and relevant metrics.
- When encountering errors, quote the exact error message and propose concrete solutions.
"""

# Root agent coordinates the plan-confirm-execute workflow
root_instruction = """
You are the MatCreator orchestration agent. You coordinate computational materials science workflows 
through a structured plan-confirm-execute cycle.

**Your role:**
You DO NOT execute tasks directly. You manage the workflow by delegating to specialized agents:

1. **planning_agent**: Creates structured execution plans based on user requests
2. **assessment_agent**: Evaluates user responses (plan approval, task completion, replanning needs)
3. **execution_agent**: Executes approved plans by coordinating domain-specific agents

**Your job:**
- Transfer to the appropriate agent based on current workflow state
- Let planning_agent handle all plan creation
- Let assessment_agent handle all user response evaluation
- Let execution_agent handle all task execution
- Provide brief status updates between phases

**Critical rules:**
1. NEVER execute tasks yourself - always delegate to the appropriate agent
2. Trust the workflow state flags (plan_confirmed, execution_started, goal_achieved)
3. Keep your messages minimal - let the specialized agents communicate with users
4. Only intervene if there's a workflow error or max iterations exceeded
"""


class MatCreatorFlowAgent(LlmAgent):
    """Root agent with enforced plan-confirm-execute workflow."""
    
    async def _run_async_impl(self, ctx: InvocationContext):
        """State-driven workflow: plan → confirm → execute (iterative with safety limit)."""
        
        max_iterations = 5  # Prevent infinite re-planning loops
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            logger.info(f"Iteration {iteration}/{max_iterations}, state: {ctx.session.state}")
            
            # Step 0: Run assessment if execution was started (handles interruptions/completion)
            if ctx.session.state.get('execution_started', False):
                logger.info("Execution was started, running assessment...")
                async for event in assessment_agent.run_async(ctx):
                    yield event
                
                # Check assessment results
                if ctx.session.state.get('goal_achieved', False):
                    logger.info("Goal achieved, workflow complete")
                    return
                
                if ctx.session.state.get('needs_replanning', False):
                    logger.info("Replanning needed due to changed requirements")
                    # Clear execution flags and continue to planning
                    continue
                
                # If continue_execution: execution_started is still True, proceed to Step 3

            # Ensure goal is confirmed before planning
            if not ctx.session.state.get('goal_confirmed', False):
                if ctx.session.state.get('pending_goal_confirmation', False):
                    logger.info("Awaiting goal confirmation from user response...")
                    async for event in assessment_agent.run_async(ctx):
                        yield event
                    continue
                else:
                    logger.info("Understanding user intent and proposing goal...")
                    async for event in planning_agent.run_async(ctx):
                        yield event
                    return  # Wait for user to confirm goal
                
            
            # Step 1: Create plan if needed
            if not ctx.session.state.get('plan_confirmed',False) and not ctx.session.state.get('pending_confirmation',False):
                logger.info("Creating plan...")
                async for event in planning_agent.run_async(ctx):
                    yield event
                return  # Wait for user response
            
            # Step 2: Get confirmation if plan not yet approved
            if not ctx.session.state.get('plan_confirmed', False):
                logger.info("Awaiting plan confirmation...")
                async for event in assessment_agent.run_async(ctx):
                    yield event
                
                # If still not confirmed (questions/unclear), wait for user
                if not ctx.session.state.get('plan_confirmed', False):
                    logger.info("Plan not confirmed-reformulate")
                    continue
            
            # Step 3: Execute approved plan
            if ctx.session.state.get('plan_confirmed', False):
                logger.info("Executing approved plan via execution_agent")
                
                # Delegate to execution agent - completes in single invocation
                async for event in execution_agent.run_async(ctx):
                    yield event
                
                logger.info("Plan execution completed, will assess on next user input")
                return  # Execution complete, wait for user input to trigger assessment
        
        # Safety: exceeded max iterations
        logger.warning(f"Exceeded max iterations ({max_iterations}) - stopping")
        yield Event(
            content=Content(parts=[Part(text=f"⚠️ Reached maximum re-planning attempts ({max_iterations}). Please start a new conversation if you need further assistance.")]),
            author=self.name
        )


def before_agent_callback_root(callback_context: CallbackContext):
    """Set environment variables and initialize session state for MatCreator agent."""
    session_id = callback_context._invocation_context.session.id
    user_id = callback_context._invocation_context.session.user_id
    app_name = callback_context._invocation_context.session.app_name
    
    # Set environment variables for session context
    os.environ["CURRENT_SESSION_ID"] = session_id
    os.environ["CURRENT_USER_ID"] = user_id
    os.environ["CURRENT_APP_NAME"] = app_name
    
    # Initialize session state variables if not present
    state = callback_context._invocation_context.session.state
    if 'plan' not in state:
        state['plan'] = None
    if 'goal' not in state:
        state['goal'] = None
    if 'goal_confirmed' not in state:
        state['goal_confirmed'] = False
    if 'pending_goal_confirmation' not in state:
        state['pending_goal_confirmation'] = False
    if 'plan_confirmed' not in state:
        state['plan_confirmed'] = False
        
    if 'pending_confirmation' not in state:
        state['pending_confirmation'] = False
    
    if 'execution_started' not in state:
        state['execution_started'] = False
    
    if 'execution_complete' not in state:
        state['execution_complete'] = False
    
    if 'goal_achieved' not in state:
        state['goal_achieved'] = False
    
    if 'needs_replanning' not in state:
        state['needs_replanning'] = False
    
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
    get_session_context,
    get_session_metadata
    ]

remote_a2a_url="http://localhost:8001/"
structure_builder_agent = RemoteA2aAgent(
    name="structure_builder_agent",
    description="",
    agent_card=f"{remote_a2a_url}{AGENT_CARD_WELL_KNOWN_PATH}",
)


root_agent = MatCreatorFlowAgent(
    name='MatCreator_agent',
    model=LiteLlm(
        model=model_name,
        base_url=model_base_url,
        api_key=model_api_key
    ),
    description=description,
    instruction=root_instruction, 
    global_instruction=global_instruction,
    before_agent_callback=before_agent_callback_root,
    tools=[set_session_metadata, get_session_context, get_session_metadata],
    sub_agents=[
        planning_agent,
        assessment_agent,
        execution_agent,
    ]
    )