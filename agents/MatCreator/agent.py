"""Root orchestration agent for MatCreator.

Defines ``MatCreatorFlowAgent``, a custom ``BaseAgent`` that routes between
the *thinking* phase (intent clarification, plan drafting, approval gating)
and the *execution* phase (approved-plan step execution) using a session
state variable ``phase``.  Also wires together the ADK ``App`` with event
compaction and resumability support.
"""

from google.adk.agents import InvocationContext, BaseAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.remote_a2a_agent import RemoteA2aAgent
from google.adk.apps.llm_event_summarizer import LlmEventSummarizer
from google.adk.apps.app import App, EventsCompactionConfig
from google.adk.apps import ResumabilityConfig
from google.adk.events import Event, EventActions
from google.genai.types import Content,Part
from google.adk.models.lite_llm import LiteLlm
from pydantic import PrivateAttr
import os
import logging
from typing import Optional, Any
from .thinking_agent import thinking_agent
from .execution_agent import execution_agent
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

    _execution_agent: Optional[Any] = PrivateAttr(default_factory=lambda: execution_agent)

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
                if state.get("approval", False):
                    break

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
            _max_execution_iterations = int(os.environ.get("MAX_EXECUTION_ITERATIONS", "20"))
            _execution_iteration = 0
            while True:
                _execution_iteration += 1
                if _execution_iteration > _max_execution_iterations:
                    logger.warning(
                        f"[{self.name}]: Execution circuit-breaker triggered after "
                        f"{_max_execution_iterations} iterations. Forcing replan."
                    )
                    ctx.session.state["execution_complete"] = True
                    ctx.session.state["recommended_next_action"] = "replan"
                    ctx.session.state["completion_status"] = "blocked"
                    break
                async for event in self._run_async_execution(ctx):
                    yield event
                if ctx.session.state.get("execution_complete", False):
                    break    
            
            if ctx.session.state.get("completion_status") == "completed" or ctx.session.state.get("recommended_next_action", "") == "mark_complete" or ctx.session.state.get("recommended_next_action", "") == "replan":
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
                
            elif ctx.session.state.get("recommended_next_action", "") == "request_user_input_but_no_need_to_replan":
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
        if ctx.session.state.get("force_break", False):
            logger.info(f"[{self.name}]: Skipping summarize step due to force_break")
            ctx.session.state["force_break"] = False
            return
                
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
        
    if 'guides' not in state:
        callback_context.state['guides'] = None
    
    if 'execution_started' not in state:
        callback_context.state['execution_started'] = False
    
    if 'execution_complete' not in state:
        callback_context.state['execution_complete'] = False

    if 'force_break' not in state:
        callback_context.state['force_break'] = False

    if 'execution_history' not in state:
        callback_context.state['execution_history'] = []

    if 'execution_last_output' not in state:
        callback_context.state['execution_last_output'] = ""
    
    if 'goal_achieved' not in state:
        callback_context.state['goal_achieved'] = False
    
    if 'needs_replanning' not in state:
        callback_context.state['needs_replanning'] = False
        
    if 'completion_status' not in state:
        callback_context.state['completion_status'] = "in_progress"
    
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

compaction_summarizer = LlmEventSummarizer(
    #llm=Gemini(model="gemini-1.5-flash") # Or another model of your choice
    llm=LiteLlm(
        model=model_name,
        base_url=model_base_url,
        api_key=model_api_key
    ),
)

app = App(
    name="MatCreator",
    root_agent=_root_agent,
    resumability_config=ResumabilityConfig(
        is_resumable=True,
    ),
    events_compaction_config=EventsCompactionConfig(
        compaction_interval=3,
        overlap_size=1,
        summarizer=compaction_summarizer  # Pass the summarizer here
    )
)