"""Planning agent for MatCreator - generates structured execution plans."""

from __future__ import annotations

import os
from typing import List, Literal, Optional

from google.adk.agents import LlmAgent
from google.adk.agents import InvocationContext
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.lite_llm import LiteLlm
from google.adk.events import Event,EventActions
from google.genai.types import Content, Part
from pydantic import BaseModel, Field
import logging
from .constants import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL
from .prompts.workflow import search_workflow_guidance,list_workflow_descriptions, get_allowed_agents_for_workflow
from .prompts.subagents import format_subagent_descriptions, format_subagent_descriptions_for_agents


class PlanStep(BaseModel):
    """Single step in the execution plan."""
    
    step_number: int = Field(..., description="Sequential step number (1, 2, 3, ...)")
    agent: str = Field(
        ...,
        description=(
            "Which agent will execute this step. Must be one of: "
            "database_agent, structure_agent, abacus_agent, vasp_agent, dpa_agent, pfd_agent"
        )
    )
    action: str = Field(
        ...,
        description="Clear, concise description of what this step does (1-2 sentences)",
        max_length=500
    )
    inputs_required: List[str] = Field(
        default_factory=list,
        description="List of required inputs (paths, parameters, etc.)"
    )
    expected_output: str = Field(
        ...,
        description="What artifact or result this step produces",
        max_length=200
    )

class WorkflowClassification(BaseModel):
    """Classification of workflow type based on user intent."""
    
    workflow_type: Literal["pfd", "default"] = Field(
        ...,
        description="Type of workflow that best matches user's request"
    )
    goal: str = Field(
        ...,
        description="Single-sentence articulation of the user's goal/intent",
        max_length=300
    )
    reasoning: str = Field(
        ...,
        description="Brief explanation of why this workflow type was chosen",
        max_length=200
    )


class ExecutionPlan(BaseModel):
    """Structured execution plan for user approval."""
    
    workflow_type: Literal["pfd", "default"] = Field(
        ...,
        description=(
            "Type of workflow"
        )
    )
    goal: str = Field(
        ...,
        description="High-level objective this plan achieves (1 sentence)",
        max_length=300
    )
    steps: List[PlanStep] = Field(
        ...,
        description="Ordered list of execution steps",
        min_items=1,
        max_items=10
    )
    fallback_strategy: str = Field(
        ...,
        description=(
            "If any step fails or is not feasible, describe an alternative approach "
            "or contingency plan (1-2 sentences)"
        ),
        max_length=500
    )
    additional_notes: str= Field(
        ...,
        description="Any extra information or considerations for the user",
        max_length=500
    )


def _format_workflow_descriptions() -> str:
    """Format workflow descriptions for classification."""
    workflow_descs = list_workflow_descriptions()
    lines = []
    for workflow_type, info in workflow_descs.items():
        lines.append(f"- **{workflow_type}**: {info['description']}")
        lines.append(f"  Tags: {', '.join(info['tags'])}")
    return '\n'.join(lines)


_CLASSIFICATION_INSTRUCTION = """
You are analyzing a user's request to determine the appropriate workflow type.

{workflow_descriptions}

**Your task:**
1) Infer the user's goal as a concise single sentence
2) Determine which workflow type best matches their intent
3) Explain your reasoning briefly

**Classification criteria:**
- **pfd**: Iterative workflows involving fine-tuning, distillation, active learning, convergence checks, or model training cycles
- **default**: General tasks, single-step operations, or multi-step workflows without iteration

Return a JSON object with fields: workflow_type, goal, reasoning.
"""

_PLANNING_INSTRUCTION = """
You are the planning agent for MatCreator. Your job is to create a clear, structured execution plan 
based on the user's request, available sub-agents and workflow types.

**Planning rules:**

1. **Maximize efficiency**: 
   - For ML potentials: ALWAYS check database_agent first for existing datasets/models
   - Prefer reusing/fine-tuning existing models over training from scratch
   - Only generate new DFT data if existing data is insufficient

2. **Be specific**:
   - Each step should clearly state which agent and what action
   - Include required inputs (file paths, parameters)
   - State expected outputs (structure files, database paths, model checkpoints)

3. **Handle uncertainty**:
   - If user request is vague, include a clarification step
   - Provide fallback strategies for risky steps

**Output format:**
Return a JSON object conforming to ExecutionPlan schema. Be concise but clear.

Focus on clarity and actionability. Users need to understand what will happen before approving.
"""


_model_name = os.environ.get("LLM_MODEL", LLM_MODEL)
_model_api_key = os.environ.get("LLM_API_KEY", LLM_API_KEY)
_model_base_url = os.environ.get("LLM_BASE_URL", LLM_BASE_URL)

logger = logging.getLogger(__name__)


class PlanningAgent(LlmAgent):
    """Planning agent that enforces state persistence."""
    
    def _format_plan(self, plan_data: ExecutionPlan) -> str:
        """Format the ExecutionPlan into user-friendly presentation."""
        lines = ["📋 **Execution Plan**\n"]
        lines.append(f"**Goal:** {plan_data.goal}\n")
        lines.append("\n**Steps:**")
        
        for step in plan_data.steps:
            lines.append(f"\n{step.step_number}. **[{step.agent}]** {step.action}")
            if step.inputs_required:
                inputs = ', '.join(step.inputs_required)
                lines.append(f"   - Inputs: {inputs}")
            lines.append(f"   - Output: {step.expected_output}")
        
        if plan_data.workflow_type:
            lines.append(f"\n\n**Workflow Type:** {plan_data.workflow_type}")
        
        if plan_data.fallback_strategy:
            lines.append(f"\n\n**Fallback Strategy:** {plan_data.fallback_strategy}")
            
        if plan_data.additional_notes:
            lines.append(f"\n\n💬 **Agent Message:**\n> {plan_data.additional_notes}")
        
        lines.append("\n\n⚠️ **Do you approve this plan?**")
        lines.append("Please reply 'yes' to proceed, or describe any changes you'd like.")
        
        return '\n'.join(lines)
    
    async def _classify_workflow(self, ctx: InvocationContext):
        """Phase 1: Classify workflow type from user request and update ctx."""
        workflow_descriptions = _format_workflow_descriptions()
        classification_instruction = _CLASSIFICATION_INSTRUCTION.format(
            workflow_descriptions=workflow_descriptions
        )
        
        
        self.instruction = classification_instruction
        self.output_schema = WorkflowClassification
        
        classification_data = None
        
        async for event in super()._run_async_impl(ctx):
            if event.is_final_response() and event.content:
                try:
                    import json
                    text_data = event.content.parts[0].text.strip()
                    if text_data.startswith('{'):
                        parsed_json = json.loads(text_data)
                        classification_data = WorkflowClassification(**parsed_json)
                        logger.info(
                            "Workflow classified as '%s' with goal '%s'", 
                            classification_data.workflow_type,
                            classification_data.goal
                        )
                        break
                except Exception as e:
                    logger.error(f"Failed to parse workflow classification: {e}")
                continue
            yield event
        
        if not classification_data:
            logger.warning("Workflow classification failed, defaulting to 'default' type")
            classification_data = WorkflowClassification(
                workflow_type="default",
                goal=ctx.session.state.get('goal', 'Not specified'),
                reasoning="Classification failed, using default workflow"
            )
        
        
        workflow_guidance = search_workflow_guidance(classification_data.workflow_type)
        state_update = {
            "goal": classification_data.goal,
            "goal_confirmed": False,
            "pending_goal_confirmation": True,
            "plan": None,
            "plan_confirmed": False,
            "pending_confirmation": False,
            "workflow_type": classification_data.workflow_type,
            "workflow_guidance": workflow_guidance,
        }

        confirmation_prompt = (
            "I interpreted your goal as:\n"
            f"- Goal: {classification_data.goal}\n"
            f"- Workflow type: {classification_data.workflow_type}\n\n"
            "Please reply 'yes' to confirm, or rewrite the goal/clarify changes."
        )

        event_action = EventActions(state_delta=state_update)
        yield Event(
            content=Content(parts=[Part(text=confirmation_prompt)]),
            author=self.name,
            actions=event_action
            )

    
    async def _create_detailed_plan(self, ctx: InvocationContext):
        """Phase 2: Create detailed execution plan with workflow guidance and present to user."""
        # Get workflow info from ctx
        workflow_type = ctx.session.state.get('workflow_type', 'default')
        workflow_guidance = search_workflow_guidance(workflow_type)
        goal = ctx.session.state.get('goal', 'Not specified')
        agent_descriptions = ctx.session.state.get('agent_descriptions_formatted')
        allowed_agents = get_allowed_agents_for_workflow(workflow_type)
        if allowed_agents:
            agent_descriptions = format_subagent_descriptions_for_agents(allowed_agents)
        if not agent_descriptions:
            agent_descriptions = format_subagent_descriptions_for_agents(allowed_agents) if allowed_agents else format_subagent_descriptions()

        base_instruction = _PLANNING_INSTRUCTION.strip()
        guidance_section = workflow_guidance.strip() if workflow_guidance else "(no additional guidance)"
        enhanced_instruction = f"""{base_instruction}

**Available sub-agents (use exactly these names):**
{agent_descriptions}

**Workflow context:**
- Workflow type: {workflow_type}
- Goal: {goal}
- Guidance:\n{guidance_section}
"""

        self.instruction = enhanced_instruction
        self.output_schema = ExecutionPlan

        plan_data = None
        
        async for event in super()._run_async_impl(ctx):
                # Don't yield the raw JSON event - we'll format it nicely
            if event.is_final_response() and event.content:
                try:
                    import json
                    text_data = event.content.parts[0].text.strip()
                    if text_data.startswith('{'):
                        parsed_json = json.loads(text_data)
                        plan_data = ExecutionPlan(**parsed_json)
                        break
                except Exception as e:
                    logger.error(f"Failed to parse execution plan: {e}")
                
                # Yield intermediate events (thinking, etc.)
            if not (event.is_final_response() and event.content):
                yield event
        
        # Restore default planning instruction for subsequent runs
        self.instruction = _PLANNING_INSTRUCTION

        
        # After plan creation, persist and present the plan
        if plan_data:
            logger.info(f"Phase 2 complete: Plan created for goal '{plan_data.goal}'")
            
            # State to be updated in a persistent way
            state_update = {
                "goal": plan_data.goal,
                "plan": plan_data.model_dump(),
                "workflow_type": workflow_type,
                "workflow_guidance": workflow_guidance,
                "goal_confirmed": True,
                "pending_goal_confirmation": False,
                "pending_confirmation": True,
                "plan_confirmed": False,
                "execution_started": False,
                "execution_complete": False,
                "goal_achieved": False,
                "needs_replanning": False
            }
            event_action = EventActions(state_delta=state_update)
            
            # Format and present the plan nicely
            formatted_plan = self._format_plan(plan_data)
            yield Event(
                content=Content(parts=[Part(text=formatted_plan)]),
                author=self.name,
                actions=event_action
            )
            logger.info(f"Plan created and persisted: {plan_data.goal}")
        else:
            logger.error("Phase 2 failed: No ExecutionPlan was extracted")
            yield Event(
                content=Content(parts=[Part(text="❌ Failed to create execution plan. Please try rephrasing your request.")]),
                author=self.name
            )
    
    async def _run_async_impl(self, ctx: InvocationContext):
        """Two-phase planning: classify workflow → create detailed plan with guidance."""
        
        # If goal is not yet confirmed, stay in intent classification/confirmation
        if not ctx.session.state.get("goal_confirmed", False):
            logger.info("Phase 1: Understanding user intent and proposing goal...")
            async for event in self._classify_workflow(ctx):
                yield event
            return

        # PHASE 2: Create detailed plan with workflow guidance
        else:
            logger.info("Phase 2: Creating detailed execution plan with guidance...")
            async for event in self._create_detailed_plan(ctx):
                yield event



def after_agent_callback(callback_context: CallbackContext):
    """Callback to handle planning events if needed."""
    #callback_context._invocation_context.
    if 'plan' in callback_context.state:
        logger.info("Plan has been set in session state.")

planning_agent = PlanningAgent(
    name="planning_agent",
    model=LiteLlm(
        model=_model_name,
        base_url=_model_base_url,
        api_key=_model_api_key,
    ),
    description="Creates structured execution plans for user approval before proceeding with tasks.",
    instruction=_PLANNING_INSTRUCTION,
    output_schema=ExecutionPlan,
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
    after_agent_callback=after_agent_callback,
)
