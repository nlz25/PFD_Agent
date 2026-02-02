"""Assessment agent for MatCreator - evaluates task completion and determines next actions."""

from __future__ import annotations

import os
import logging
from typing import Literal, Optional

from google.adk.agents import LlmAgent
from google.adk.agents import InvocationContext
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.lite_llm import LiteLlm
from google.adk.events import Event, EventActions
from google.genai.types import Content, Part
from pydantic import BaseModel, Field

from .constants import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL


_model_name = os.environ.get("LLM_MODEL", LLM_MODEL)
_model_api_key = os.environ.get("LLM_API_KEY", LLM_API_KEY)
_model_base_url = os.environ.get("LLM_BASE_URL", LLM_BASE_URL)

logger = logging.getLogger(__name__)


class PlanApprovalAssessment(BaseModel):
    """Assessment of user's response to a presented plan."""
    
    plan_approved: bool = Field(
        ...,
        description=(
            "True if user explicitly approves the plan (yes, ok, proceed, looks good, etc.). "
            "False if user rejects, wants changes, or is uncertain."
        )
    )
    
    needs_modification: bool = Field(
        ...,
        description=(
            "True if user requests specific changes to the plan. "
            "False if approved as-is or rejected entirely."
        )
    )
    
    modifications_requested: str = Field(
        default="",
        description=(
            "If needs_modification=True, describe what changes the user wants. "
            "Empty string if no modifications requested."
        ),
        max_length=500
    )
    
    clarification_needed: str = Field(
        default="",
        description=(
            "If user asked questions or seems uncertain, what clarification do they need? "
            "Empty string if no clarification needed."
        ),
        max_length=500
    )


class GoalConfirmationAssessment(BaseModel):
    """Assessment of whether the user confirms the inferred goal."""

    goal_confirmed: bool = Field(
        ...,
        description="True if the user explicitly or implicitly confirmed the proposed goal"
    )
    revised_goal: str = Field(
        default="",
        description="Updated goal if the user corrected/rewrote it. Empty if unchanged.",
        max_length=400
    )
    clarification_needed: str = Field(
        default="",
        description="If the user asked for more info or the goal is unclear, capture what is needed",
        max_length=400
    )


class ExecutionAssessment(BaseModel):
    """Assessment of execution progress and completion status."""
    
    goal_achieved: bool = Field(
        ...,
        description=(
            "True if the original goal has been fully achieved based on the plan and "
            "conversation history. False if work is incomplete or interrupted."
        )
    )
    
    needs_replanning: bool = Field(
        ...,
        description=(
            "True if user's recent input indicates changed requirements that require "
            "a new plan. False if we should continue with the current plan."
        )
    )
    
    reason: str = Field(
        ...,
        description=(
            "Clear explanation of why this assessment was made. What evidence from "
            "the conversation supports this conclusion?"
        ),
        max_length=500
    )
    
    next_action: Literal["continue_execution", "replan", "complete"] = Field(
        ...,
        description=(
            "'continue_execution': Resume executing the current plan\n"
            "'replan': User has changed requirements, create new plan\n"
            "'complete': Goal achieved, task is done"
        )
    )


_PLAN_APPROVAL_INSTRUCTION = """
You are assessing the user's response to a proposed execution plan.

**Your job:**
Determine whether the user:
1. Approves the plan (yes, ok, proceed, looks good, let's do it, etc.)
2. Requests specific modifications (change step X, use different approach, etc.)
3. Needs clarification (asks questions about the plan)
4. Rejects the plan entirely

**Assessment guidelines:**

**plan_approved = True:**
- Explicit approval only: "yes", "ok", "proceed", "looks good", "approved", "let's go", "do it", "I approve"
- Conditional approval: "yes, but..." (set needs_modification=True if conditions given)
- If approval is not explicit, treat as plan_approved=False

**plan_approved = False:**
- Rejection: "no", "I don't think so", "that won't work"
- Uncertainty: "I'm not sure", "maybe", "hmm"
- Questions about feasibility: "will this work?", "is this the right approach?"
- Any other responses without explicit approval: "sounds great", "perfect", "that works", "thanks" (no approval phrase)

**needs_modification = True:**
- Explicit changes: "change step 2 to use X instead of Y"
- Additions: "add a step to validate the results"
- Different approach: "use method A instead of method B"
- Parameter changes: "increase the dataset size"

**clarification_needed:**
- User asks questions: "what does step 2 do?", "why VASP instead of ABACUS?"
- Expresses confusion: "I don't understand step 3"
- Seeks more info: "what parameters will be used?"

**Output format:**
Return JSON conforming to PlanApprovalAssessment schema.

**Examples:**

User: "yes, let's proceed"
→ plan_approved=True, needs_modification=False

User: "looks good, but use 500 structures instead of 100"
→ plan_approved=False, needs_modification=True, modifications_requested="Increase dataset size from 100 to 500 structures"

User: "what temperature will step 2 use?"
→ plan_approved=False, needs_modification=False, clarification_needed="User wants to know what temperature will be used in step 2"

User: "no, I want to use ABACUS instead of VASP"
→ plan_approved=False, needs_modification=True, modifications_requested="Use ABACUS instead of VASP for DFT calculations"
"""


_EXECUTION_ASSESSMENT_INSTRUCTION = """
You are assessing execution progress and task completion after execution has started.

**Your job:**
1. Review the original goal and approved plan
2. Examine the conversation history and execution progress
3. Assess whether the goal has been achieved
4. Determine if user input indicates changed requirements
5. Recommend next action

**Assessment criteria:**

**Goal Achieved (complete):**
- All plan steps were executed successfully OR
- The goal was accomplished even if some steps were skipped/modified
- User has not requested additional work

**Needs Replanning (replan):**
- User explicitly requested changes to the approach
- User asked for different/additional functionality
- User indicated the current plan won't work
- Execution revealed fundamental issues with the plan

**Continue Execution (continue_execution):**
- Execution was interrupted mid-way (user question, clarification needed)
- User provided requested information to continue
- Some steps remain incomplete but plan is still valid
- No fundamental changes to requirements

**Critical rules:**
- Be conservative: if unclear whether done, recommend continue_execution
- User asking clarifying questions ≠ changed requirements
- Errors in execution don't automatically require replanning
- Focus on the ORIGINAL goal, not intermediate steps

**Output format:**
Return JSON conforming to ExecutionAssessment schema with clear reasoning.

**Example 1 - Mid-execution clarification:**
Original goal: "Train DeepMD model for silicon"
User just said: "What temperature should I use for MD?"
→ goal_achieved=False, needs_replanning=False, next_action="continue_execution"
Reason: "User asking clarification question, execution can continue once answered"

**Example 2 - Goal completed:**
Original goal: "Build diamond Si structure"
Last execution output: "Structure saved to /path/Si.xyz"
→ goal_achieved=True, needs_replanning=False, next_action="complete"
Reason: "Structure was successfully created and saved as requested"

**Example 3 - Changed requirements:**
Original goal: "Fine-tune model with 100 structures"
User just said: "Actually, use 500 structures instead and add temperature ramping"
→ goal_achieved=False, needs_replanning=True, next_action="replan"
Reason: "User changed dataset size and added new temperature ramping requirement"
"""


_GOAL_CONFIRMATION_INSTRUCTION = """
You are confirming whether the proposed goal matches the user's intent.

**Your job:**
1. Decide if the user confirmed the goal
2. Capture any revised goal wording they provided
3. Note if they asked for clarification

**Rules:**
- Confirmed if they say yes/agree/that's right or restate the same goal
- If they rewrite or adjust the goal, treat that as the revised_goal and goal_confirmed=True unless they express uncertainty
- If they ask questions or seem unsure, set goal_confirmed=False and describe the clarification_needed

Return JSON with: goal_confirmed (bool), revised_goal (string), clarification_needed (string).
"""


class AssessmentAgent(LlmAgent):
    """Agent that assesses user responses for both plan approval and execution status."""
    
    async def _run_async_impl(self, ctx: InvocationContext):
        """Assess based on current context: plan approval or execution status."""
        
        # Determine assessment mode based on state
        pending_goal_confirmation = ctx.session.state.get('pending_goal_confirmation', False)
        pending_confirmation = ctx.session.state.get('pending_confirmation', False)
        execution_started = ctx.session.state.get('execution_started', False)

        # Mode 0: Goal confirmation
        if pending_goal_confirmation and not execution_started:
            async for event in self._assess_goal_confirmation(ctx):
                yield event
            return
        
        # Mode 1: Plan Approval Assessment
        if pending_confirmation and not execution_started:
            async for event in self._assess_plan_approval(ctx):
                yield event
        
        # Mode 2: Execution Assessment
        elif execution_started:
            async for event in self._assess_execution(ctx):
                yield event
        
        else:
            logger.warning("Assessment agent called but no clear context (no pending_confirmation or execution_started)")
            yield Event(
                content=Content(parts=[Part(text="ℹ️ Nothing to assess at this time.")]),
                author=self.name
            )

    async def _assess_goal_confirmation(self, ctx: InvocationContext):
        """Assess whether the user confirmed the inferred goal."""
        proposed_goal = ctx.session.state.get('goal', 'Not specified')
        workflow_type = ctx.session.state.get('workflow_type', 'default')

        goal_context = f"""
Proposed goal: {proposed_goal}
Workflow type: {workflow_type}

The user's most recent reply is in the conversation above.

Respond with JSON per the schema.
"""

        original_instruction = self.instruction
        original_schema = self.output_schema

        self.instruction = _GOAL_CONFIRMATION_INSTRUCTION + "\n\n" + goal_context
        self.output_schema = GoalConfirmationAssessment

        assessment_data = None

        try:
            async for event in super()._run_async_impl(ctx):
                if event.is_final_response() and event.content:
                    try:
                        import json
                        text_data = event.content.parts[0].text.strip()
                        if text_data.startswith('{'):
                            parsed_json = json.loads(text_data)
                            assessment_data = GoalConfirmationAssessment(**parsed_json)
                            logger.info(
                                "Goal confirmation=%s, revised_goal='%s'",
                                assessment_data.goal_confirmed,
                                assessment_data.revised_goal,
                            )
                            continue
                    except Exception as e:
                        logger.debug(f"Could not parse goal confirmation as JSON: {e}")

                if not (event.is_final_response() and event.content):
                    yield event

        finally:
            self.instruction = original_instruction
            self.output_schema = original_schema

        if assessment_data:
            new_goal = assessment_data.revised_goal.strip() or proposed_goal

            if assessment_data.goal_confirmed:
                state_update = {
                    "goal": new_goal,
                    "goal_confirmed": True,
                    "pending_goal_confirmation": False,
                    "plan": None,
                    "plan_confirmed": False,
                    "pending_confirmation": False,
                    "needs_replanning": False,
                }
                message = (
                    "✅ Goal confirmed.\n\n"
                    f"Goal: {new_goal}\n"
                    "I'll draft a plan next."
                )
            else:
                state_update = {
                    "goal": new_goal,
                    "goal_confirmed": False,
                    "pending_goal_confirmation": False,
                    "workflow_type": None,
                    "workflow_guidance": None,
                    "plan": None,
                    "plan_confirmed": False,
                    "pending_confirmation": False,
                    "needs_replanning": False,
                }
                clarification = assessment_data.clarification_needed or "Please restate the goal in one sentence."
                message = (
                    "❓ Goal not yet confirmed.\n\n"
                    f"{clarification}"
                )

            event_action = EventActions(state_delta=state_update)
            yield Event(
                content=Content(parts=[Part(text=message)]),
                author=self.name,
                actions=event_action
            )
        else:
            logger.warning("Goal confirmation assessment completed but no result extracted")
    
    async def _assess_plan_approval(self, ctx: InvocationContext):
        """Assess user's response to a proposed plan."""
        plan = ctx.session.state.get('plan')
        goal = ctx.session.state.get('goal', 'No goal specified')
        
        # Build approval assessment context
        approval_context = f"""
**Proposed Plan:**
Goal: {goal}

{self._format_plan(plan) if plan else 'No plan available'}

**User's most recent response is in the conversation history above.**

**Your task:**
Analyze the user's response and determine:
1. Did they approve the plan?
2. Do they want modifications?
3. Do they need clarification?
4. If the user does NOT explicitly approve, set plan_approved=False by default.
"""
        
        # Inject context and run assessment
        original_instruction = self.instruction
        original_schema = self.output_schema
        
        self.instruction = _PLAN_APPROVAL_INSTRUCTION + "\n\n" + approval_context
        self.output_schema = PlanApprovalAssessment
        
        approval_data = None
        
        try:
            async for event in super()._run_async_impl(ctx):
                # Capture the approval assessment result
                if event.is_final_response() and event.content:
                    try:
                        import json
                        text_data = event.content.parts[0].text.strip()
                        if text_data.startswith('{'):
                            parsed_json = json.loads(text_data)
                            approval_data = PlanApprovalAssessment(**parsed_json)
                            logger.info(f"Plan approval: {approval_data.plan_approved}, modifications: {approval_data.needs_modification}")
                            continue
                    except Exception as e:
                        logger.debug(f"Could not parse approval as JSON: {e}")
                
                # Yield other events
                if not (event.is_final_response() and event.content):
                    yield event
        
        finally:
            self.instruction = original_instruction
            self.output_schema = original_schema
        
        # Update state based on approval assessment
        if approval_data:
            if approval_data.plan_approved:
                # Plan approved as-is
                state_update = {
                    "plan_confirmed": True,
                    "pending_confirmation": False,
                }
                message = "✅ **Plan Approved**\n\nProceeding with execution..."
            
            elif approval_data.needs_modification:
                # User wants changes - need to replan
                state_update = {
                    "plan_confirmed": False,
                    "pending_confirmation": False,
                    "plan_modifications": approval_data.modifications_requested,
                }
                message = f"🔄 **Modifications Requested**\n\n{approval_data.modifications_requested}\n\nCreating updated plan..."
            
            elif approval_data.clarification_needed:
                # User has questions - stay in pending state
                state_update = {
                    "plan_confirmed": False,
                    "pending_confirmation": False,
                    "plan_clarification": approval_data.clarification_needed,
                }
                message = f"💬 **Clarification Needed**\n\n{approval_data.clarification_needed}\n\nLet me address your questions..."
            
            else:
                # Rejected without modifications - need user to clarify what they want
                state_update = {
                    "plan_confirmed": False,
                    "pending_confirmation": False,
                }
                message = "❌ **Plan Not Approved**\n\nCould you tell me what changes you'd like or what approach you prefer?"
            
            event_action = EventActions(state_delta=state_update)
            yield Event(
                content=Content(parts=[Part(text=message)]),
                author=self.name,
                actions=event_action
            )
        else:
            logger.warning("Plan approval assessment completed but no result extracted")
    
    async def _assess_execution(self, ctx: InvocationContext):
        """Assess current execution state and determine next actions."""
        
        # Read current state
        plan = ctx.session.state.get('plan')
        goal = ctx.session.state.get('goal', 'No goal specified')
        execution_started = ctx.session.state.get('execution_started', False)
        
        if not execution_started:
            logger.info("Execution never started, no assessment needed")
            yield Event(
                content=Content(parts=[Part(text="ℹ️ No execution in progress to assess.")]),
                author=self.name
            )
            return
        
        # Build assessment context
        assessment_context = f"""
**Original Goal:** {goal}

**Approved Plan:**
{self._format_plan(plan) if plan else 'No plan available'}

**Execution Status:**
- Execution started: {execution_started}
- Plan confirmed: {ctx.session.state.get('plan_confirmed', False)}

**Recent conversation history is available in the context above.**

**Your task:**
Assess whether:
1. The goal has been achieved
2. User input requires replanning
3. What action should happen next

Provide clear reasoning based on the conversation and execution progress.
"""
        
        # Inject context and run assessment
        original_instruction = self.instruction
        original_schema = self.output_schema
        
        self.instruction = _EXECUTION_ASSESSMENT_INSTRUCTION + "\n\n" + assessment_context
        self.output_schema = ExecutionAssessment
        
        assessment_data = None
        
        try:
            async for event in super()._run_async_impl(ctx):
                # Capture the assessment result
                if event.is_final_response() and event.content:
                    try:
                        import json
                        text_data = event.content.parts[0].text.strip()
                        if text_data.startswith('{'):
                            parsed_json = json.loads(text_data)
                            assessment_data = ExecutionAssessment(**parsed_json)
                            logger.info(f"Assessment: {assessment_data.next_action} - {assessment_data.reason}")
                            continue
                    except Exception as e:
                        logger.debug(f"Could not parse assessment as JSON: {e}")
                
                # Yield other events
                if not (event.is_final_response() and event.content):
                    yield event
        
        finally:
            self.instruction = original_instruction
            self.output_schema = original_schema
        
        # Update state based on assessment
        if assessment_data:
            state_update = {
                "goal_achieved": assessment_data.goal_achieved,
                "needs_replanning": assessment_data.needs_replanning,
                "assessment_reason": assessment_data.reason,
            }
            
            # Set specific flags based on next action
            if assessment_data.next_action == "complete":
                state_update["execution_complete"] = True
                state_update["execution_started"] = False
                state_update["plan_confirmed"] = False
                state_update["pending_confirmation"] = False
                message = f"✅ **Task Complete**\n\n{assessment_data.reason}"
            
            elif assessment_data.next_action == "replan":
                state_update["goal_achieved"] = False
                #state_update["goal_confirmed"] = False
                state_update["execution_started"] = False
                state_update["plan_confirmed"] = False
                state_update["pending_confirmation"] = False
                state_update["plan"] = None  # Clear old plan
                message = f"🔄 **Replanning Required**\n\n{assessment_data.reason}\n\nI'll create a new plan based on your updated requirements."
            
            else:  # continue_execution
                state_update["execution_complete"] = False
                message = f"▶️ **Continuing Execution**\n\n{assessment_data.reason}\n\nResuming work on the current plan..."
            
            event_action = EventActions(state_delta=state_update)
            yield Event(
                content=Content(parts=[Part(text=message)]),
                author=self.name,
                actions=event_action
            )
        else:
            logger.warning("Assessment agent completed but no ExecutionAssessment was extracted")
    
    def _format_plan(self, plan: dict) -> str:
        """Format plan for display in assessment context."""
        if not plan:
            return "No plan available"
        
        lines = []
        for step in plan.get('steps', []):
            lines.append(
                f"{step['step_number']}. [{step['agent']}] {step['action']}"
            )
        return '\n'.join(lines)


# Create assessment agent instance
assessment_agent = AssessmentAgent(
    name="assessment_agent",
    model=LiteLlm(
        model=_model_name,
        base_url=_model_base_url,
        api_key=_model_api_key,
    ),
    description="Assesses user responses for plan approval and evaluates execution progress to determine task completion.",
    instruction=_EXECUTION_ASSESSMENT_INSTRUCTION,  # Default instruction (will be overridden based on context)
    output_schema=ExecutionAssessment,  # Default schema (will be overridden based on context)
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
)
