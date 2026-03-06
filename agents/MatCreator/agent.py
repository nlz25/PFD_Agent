from google.adk.agents import InvocationContext, BaseAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.remote_a2a_agent import RemoteA2aAgent
from google.adk.apps import App, ResumabilityConfig
from google.adk.apps.app import EventsCompactionConfig
from google.adk.events import Event, EventActions
from google.genai.types import Content,Part
from pydantic import PrivateAttr
import os
import logging
from typing import Optional, Any
from .thinking_agent import thinking_agent
from .execution_agent import build_execution_agent
from .summarize_agent import summarize_agent
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
a structured thinking-execution cycle, ensuring user visibility and approval before expensive operations.
""" 

global_instruction = """
General rules for all agents:
- Only call tools that are available in your current context.
- Keep responses concise; include key artifacts with absolute paths and relevant metrics.
- When encountering errors, quote the exact error message and propose concrete solutions.
"""

# Root agent coordinates the phase-based thinking-execution workflow
root_instruction = """
You are the MatCreator orchestration agent. You coordinate computational materials science workflows 
through a phase-based thinking-execution cycle.

**Your role:**
You DO NOT execute tasks directly. You route work using the session phase:

1. **phase=thinking** → delegate to **thinking_agent**
2. **phase=execution** → delegate to **execution_agent**

**Your job:**
- Transfer to the appropriate agent based on phase only
- Let thinking_agent handle intent clarification, plan drafting, and plan approval handling
- Let execution_agent handle all approved task execution

**Critical rules:**
1. NEVER execute tasks yourself - always delegate to the appropriate agent
2. Trust the phase state variable (thinking/execution)
3. Keep your messages minimal - let the specialized agents communicate with users
4. If phase is invalid or missing, normalize to thinking
"""

class MatCreatorFlowAgent(BaseAgent):
    """Root agent with enforced phase-based routing."""

    _execution_agent: Optional[Any] = PrivateAttr(default=None)

    @property
    def execution_agent(self) -> Optional[Any]:
        """The current execution agent instance, built from the active plan."""
        return self._execution_agent

    @execution_agent.setter
    def execution_agent(self, agent: BaseAgent) -> None:
        self._execution_agent = agent

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    async def _run_async_impl_cycle(self, ctx: InvocationContext):
        """Route to the correct workflow phase, persisting step in session.state."""
        state = ctx.session.state
        if ctx.session.state["phase"]=="thinking":
            logger.info(f"[{self.name}]: Starting thinking")
            async for event in thinking_agent.run_async(ctx):
                yield event

            if not state.get("approval", False):
                return
            event_action = EventActions(state_delta={"phase":"execution"})
            event = Event(
                content=Content(parts=[Part(text=f"🚀 Starting execution")]),
                author=self.name,
                actions=event_action
        )
            yield event
            
            
            
        if ctx.session.state["phase"]=="execution":
            logger.info(f"[{self.name}]: Starting execution loop")
            if self.execution_agent == None:
                logger.info(f"[{self.name}]: Building execution agent from approved plan")
                self.execution_agent = build_execution_agent(ctx.session.state.get("plan", {}))
                
            while True:
                async for event in self._run_async_execution(ctx):
                    yield event
                if ctx.session.state.get("execution_complete", False):
                    break    
                
            if ctx.session.state.get("recommended_next_action", "") == "replan" or ctx.session.state.get("recommended_next_action", "") == "mark_complete":
                logger.info(f"[{self.name}]: Execution complete with recommended next action {ctx.session.state.get('recommended_next_action','')}, routing back to thinking agent.")
                event_action = EventActions(state_delta={
                    "phase":"thinking",
                    "approval": True,
                    })
                event = Event(
                    content=Content(parts=[Part(text=f"thinking...")]),
                    author=self.name,
                    actions=event_action
                    )
                yield event
                
            elif ctx.session.state.get("recommended_next_action", "") == "request_user_input":
                logger.info(f"[{self.name}]: Execution agent recommends requesting user input, remain in execution phase")
                event_action = EventActions(state_delta={"approval":False})
                event = Event(
                    content=Content(parts=[Part(text=f"Waiting for user input...")]),
                    author=self.name,
                    actions=event_action
                    )
                yield event
        return
            
    async def _run_async_execution(self, ctx: InvocationContext):
        """Override to prevent automatic phase changes after each tool call."""
        async for event in self.execution_agent.run_async(ctx):
                yield event
                
        logger.info(f"[{self.name}]: Summarizing result")
        async for event in summarize_agent.run_async(ctx):
                yield event
    
    async def _run_async_impl(self, ctx: InvocationContext):
        '''Main loop to continuously route between phases until session ends.'''
        while True:
            async for event in self._run_async_impl_cycle(ctx):
                yield event
            if not ctx.session.state.get("approval",False):
                break   
        return     
            

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
    if 'phase' not in state:
        callback_context.state['phase']='thinking'

    # Legacy state defaults kept temporarily for migration compatibility
    if 'plan' not in state:
        callback_context.state['plan'] = None
        
    if 'detailed_steps' not in state:
        callback_context.state['detailed_steps'] = None
        
    if 'goal' not in state:
        callback_context.state['goal'] = None
    
    if 'approval' not in state:
        callback_context.state['approval'] = False
        
    if 'summarize' not in state:
        callback_context.state['summarize'] = None
        
    if 'skills' not in state:
        callback_context.state['skills'] = None
    
    if 'execution_started' not in state:
        callback_context.state['execution_started'] = False
    
    if 'execution_complete' not in state:
        callback_context.state['execution_complete'] = False

    if 'execution_history' not in state:
        callback_context.state['execution_history'] = []

    if 'execution_last_output' not in state:
        callback_context.state['execution_last_output'] = ""
    
    if 'goal_achieved' not in state:
        callback_context.state['goal_achieved'] = False
    
    if 'needs_replanning' not in state:
        callback_context.state['needs_replanning'] = False
    
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


_root_agent = MatCreatorFlowAgent(
    name='MatCreator_agent',
    before_agent_callback=before_agent_callback_root,
    sub_agents=[
        thinking_agent
    ]
    )

app = App(
    name="MatCreator",
    root_agent=_root_agent,
    resumability_config=ResumabilityConfig(
        is_resumable=True,
    ),
    events_compaction_config=EventsCompactionConfig(
        compaction_interval=3,  # Trigger compaction every 3 new invocations.
        overlap_size=1          # Include last invocation from the previous window.
    ),
    # Optionally include App-level features:
    # plugins, context_cache_config, resumability_config
)