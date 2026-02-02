"""Execution agent for MatCreator - executes approved plans by delegating to domain agents."""

from __future__ import annotations

import os
import logging
from typing import Dict, Any, List
from google.adk.agents import LlmAgent, InvocationContext
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

logger = logging.getLogger(__name__)

_EXECUTION_INSTRUCTION = """
You are the execution agent. Your sole responsibility is to execute an approved plan by delegating to domain agents.

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
    
    async def _run_async_impl(self, ctx: InvocationContext):
        """Execute the approved plan step by step."""
        
        # Read plan and guidance from session state
        plan = ctx.session.state.get('plan')
        goal = ctx.session.state.get('goal', '')
        workflow_type = plan.get('workflow_type', 'default') if plan else 'default'
        workflow_guidance = ctx.session.state.get('workflow_guidance', '')
        
        if not plan:
            logger.error("No plan found in session state")
            yield Event(
                content=Content(parts=[Part(text="❌ Error: No execution plan found. Please create a plan first.")]),
                author=self.name
            )
            return
        
        # Mark execution as started
        state_update = {"execution_started": True, "execution_complete": False}
        event_action = EventActions(state_delta=state_update)
        yield Event(
            content=Content(parts=[Part(text=f"🚀 Starting execution: {goal}")]),
            author=self.name,
            actions=event_action
        )
        
        logger.info(f"Starting execution of plan: {goal}")
        
        # Build execution context with workflow-specific guidance
        execution_context = f"""
**Workflow Type:** {workflow_type}
**Goal:** {goal}

**Approved Plan to Execute:**
{self._format_plan(plan)}

{workflow_guidance}

**Instructions:**
Execute each step in sequence by transferring to the appropriate domain agent.
After each agent completes, summarize the results before moving to the next step.
"""
        
        # Inject execution context into instruction
        original_instruction = self.instruction
        self.instruction = _EXECUTION_INSTRUCTION + "\n\n" + execution_context
        
        # Execute by delegating to LLM with domain agents available
        try:
            async for event in super()._run_async_impl(ctx):
                yield event
            
            logger.info("Plan execution completed successfully")
            
        except Exception as e:
            logger.error(f"Execution failed: {e}")
            yield Event(
                content=Content(parts=[Part(text=f"❌ Execution error: {str(e)}")]),
                author=self.name
            )
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


# Create execution agent instance with domain agents as sub-agents
execution_agent = ExecutionAgent(
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
    sub_agents=list(SUBAGENTS.values())
)


# Export helper functions
__all__ = [
    "execution_agent"
]
