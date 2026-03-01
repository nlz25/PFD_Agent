"""Execution agent for MatCreator - executes approved plans by delegating to domain agents."""

from __future__ import annotations

import os
import logging
from typing import Dict, Any, List
from google.adk.agents import LlmAgent, InvocationContext
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.lite_llm import LiteLlm
from google.adk.events import Event, EventActions
from google.genai.types import Content, Part
from .constants import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL
from .callbacks import after_tool_callback
from .prompts.subagents import (
    SUBAGENTS,
)

_model_name = os.environ.get("LLM_MODEL", LLM_MODEL)
_model_api_key = os.environ.get("LLM_API_KEY", LLM_API_KEY)
_model_base_url = os.environ.get("LLM_BASE_URL", LLM_BASE_URL)

logger = logging.getLogger()

_EXECUTION_INSTRUCTION = """
You are the execution agent. Your sole responsibility is to execute an approved plan by delegating to domain agents.

Steps: {detailed_steps}

**Your task:**
- Read the approved plan from session state
- Execute steps sequentially in order
- For each step: delegate to the specified domain agent
- After each step: report results with absolute paths and metrics
- Collect all results and provide final summary


**Execution rules:**
1. Transfer to ONE agent at a time based on current step
2. Wait for agent completion before proceeding to next step
3. Summarize results after each step
4. On errors: report exact error message, propose solution, and STOP execution
5. Do not deviate from the plan - follow it precisely

**Workflow-specific guidance will be provided in the context.**
"""


class ExecutionAgent(LlmAgent):
    """Agent that executes approved plans by orchestrating domain agents."""

    @staticmethod
    def _extract_event_text(event: Event) -> str:
        """Extract plain text content from an ADK event."""
        content = getattr(event, "content", None)
        if content is None:
            return ""
        parts = getattr(content, "parts", None) or []
        chunks: List[str] = []
        for part in parts:
            text = getattr(part, "text", None)
            if isinstance(text, str) and text.strip():
                chunks.append(text.strip())
        return "\n".join(chunks)

    @classmethod
    def _save_key_execution_event(cls, state: Dict[str, Any], event: Event) -> None:
        """Save key execution event text to state['execution_history'] as a list."""
        text = cls._extract_event_text(event)
        if not text:
            return

        author = getattr(event, "author", "")
        entry = f"[{author}] {text}" if author else text

        history = state.get("execution_history")
        if not isinstance(history, list):
            history = []

        if not history or history[-1] != entry:
            history.append(entry)

        if len(history) > 200:
            history = history[-200:]

        state["execution_history"] = history
        state["execution_last_output"] = entry
    
    async def _run_async_impl(self, ctx: InvocationContext):
        """Execute the approved plan step by step."""
        
        # Read plan and guidance from session state
        plan = ctx.session.state.get('plan')
        goal = ctx.session.state.get('goal', '')
        
        if not plan:
            logger.error("No plan found in session state")
            yield Event(
                content=Content(parts=[Part(text="❌ Error: No execution plan found. Please create a plan first.")]),
                author=self.name
            )
            return
        
        # Mark execution as started
        state_update = {
            "phase": "execution",
            "execution_started": True,
            "execution_complete": False,
            "goal_achieved": False,
            "execution_history": [],
            "execution_last_output": "",
        }
        ctx.session.state.update(state_update)
        event_action = EventActions(state_delta=state_update)
        start_event = Event(
            content=Content(parts=[Part(text=f"🚀 Starting execution: {goal}")]),
            author=self.name,
            actions=event_action
        )
        self._save_key_execution_event(ctx.session.state, start_event)
        yield start_event
        
        logger.info(f"Starting execution of plan: {goal}")
        
        # Build execution context with workflow-specific guidance
        
        # Inject execution context into instruction
        original_instruction = self.instruction
        self.instruction = _EXECUTION_INSTRUCTION #+ "\n\n" + execution_context
        
        # Execute by delegating to LLM with domain agents available
        try:
            async for event in super()._run_async_impl(ctx):
                self._save_key_execution_event(ctx.session.state, event)
                yield event

            ctx.session.state["execution_complete"] = True
            ctx.session.state["goal_achieved"] = True
            logger.info("Plan execution completed successfully")
            
        except Exception as e:
            logger.error(f"Execution failed: {e}")
            error_event = Event(
                content=Content(parts=[Part(text=f"❌ Execution error: {str(e)}")]),
                author=self.name
            )
            self._save_key_execution_event(ctx.session.state, error_event)
            ctx.session.state["execution_complete"] = True
            ctx.session.state["goal_achieved"] = False
            yield error_event
        finally:
            # Restore original instruction
            self.instruction = original_instruction
            logger.info("Execution agent finished (assessment agent will determine next steps)")
    
    def _format_plan(self, plan: Dict[str, Any]) -> str:
        """Format plan steps for display."""
        lines = []
        for step in plan.get('steps', []):
            lines.append(
                f"{step['step_number']}. [{step['agent']}] {step['action']}\n"
                f"   Expected output: {step['expected_output']}"
            )
        return '\n'.join(lines)

# After agent callback
def after_agent_callback(callback_context: CallbackContext):
    """Reset phase to thinking after execution completes."""
    callback_context.state['phase'] = 'thinking'
    return None


def build_execution_agent(plan: dict) -> ExecutionAgent:
    """Factory that instantiates an ExecutionAgent scoped to the agents named in *plan*.

    The set of required domain agents is derived from the ``agent`` field of each
    ``PlanStep`` in *plan*. Only agents present in the global SUBAGENTS registry are
    attached; unknown names are logged as warnings. If the plan is empty or malformed
    every registered sub-agent is attached as a safe fallback.
    """
    steps = plan.get("steps", []) if isinstance(plan, dict) else []
    needed_names: set[str] = {step["agent"] for step in steps if isinstance(step, dict) and "agent" in step}

    if needed_names:
        unknown = needed_names - SUBAGENTS.keys()
        if unknown:
            logger.warning(f"[ExecutorFactory] Plan references unknown agents: {unknown} — they will be skipped.")
        sub_agents = [SUBAGENTS[name] for name in needed_names if name in SUBAGENTS]
        logger.info(f"[ExecutorFactory] Attaching sub-agents: {[a.name for a in sub_agents]}")
        print(f"[ExecutorFactory] Attaching sub-agents: {[a.name for a in sub_agents]}")
        print(f"Name{__name__}")
    else:
        # Fallback: attach all registered agents when plan provides no agent names
        logger.warning("[ExecutorFactory] No agent names found in plan steps — attaching all sub-agents as fallback.")
        sub_agents = list(SUBAGENTS.values())
    return ExecutionAgent(
        name="execution_agent",
        model=LiteLlm(
            model=_model_name,
            base_url=_model_base_url,
            api_key=_model_api_key,
        ),
        description="Executes approved plans by delegating to domain-specific agents in sequence.",
        instruction=_EXECUTION_INSTRUCTION,
        after_tool_callback=after_tool_callback,
        disallow_transfer_to_parent=True,
        disallow_transfer_to_peers=True,
        after_agent_callback=after_agent_callback,
        sub_agents=sub_agents,
    )


# Export helpers
__all__ = [
    "build_execution_agent",
]
