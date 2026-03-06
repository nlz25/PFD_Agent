"""Plan-builder sub-agent for the ThinkingAgent.

This agent is the structural equivalent of `database_agent/sql_agent`:
it accepts a structured planning request via `PlanBuilderInput` and returns
a fully-formed `ExecutionPlan` JSON without any custom orchestration logic.
"""

from __future__ import annotations

import os
from typing import List, Literal

from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from pydantic import BaseModel, Field

from ...constants import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL

_model_name = os.environ.get("LLM_MODEL", LLM_MODEL)
_model_api_key = os.environ.get("LLM_API_KEY", LLM_API_KEY)
_model_base_url = os.environ.get("LLM_BASE_URL", LLM_BASE_URL)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class PlanStep(BaseModel):
    """Single step in the execution plan."""

    step_number: int = Field(..., description="Sequential step number (1, 2, 3, ...)")
    agent: str = Field(
        ...,
        description=(
            "Agent or tool that will execute this step."
        ),
    )
    action: str = Field(
        ...,
        description="Clear, concise description of what this step does (1-2 sentences)",
        max_length=500,
    )
    #transfer_back: bool = Field(
    #    ...,
    #    description="Default to False. Transfer back to the thinking agent if not sure whether to execute next step or not."
    #)
    #inputs_required: str = Field(
    #    ...,
    #    description="Expected inputs (models, parameters, etc.)",
    #    max_length=300,
    #)
    #expected_output: str = Field(
    #    ...,
    #    description="What result this step produces",
    #    max_length=300,
    #)


class ExecutionPlan(BaseModel):
    """Structured execution plan for user approval."""
    stages: List[str] = Field(
        ..., description="Stages of execution, be general. Example: ['Evaluate the pre-trained model','Proceed to fine-tuning only if neccesary']"
    )
    current_stage: int = Field(..., description="The current stage of execution")
    steps: List[PlanStep] = Field(
        ...,
        #description="Ordered list of detailed execution steps, ONLY includes DETERMINED steps",
        description="Ordered list of detailed steps in the CURRENT stage, ONLY includes DETERMINED steps",
        min_items=1,
        max_items=10,
    )
    additional_notes: str = Field(
        ...,
        description="Any extra information or considerations for the user",
        max_length=500,
    )


class PlanBuilderInput(BaseModel):
    """Structured request passed from the ThinkingAgent to the plan-builder."""

    goal: str = Field(
        ...,
        description="Immediate goal in one sentence.",
    )


# ---------------------------------------------------------------------------
# Instruction
# ---------------------------------------------------------------------------

_PLAN_BUILDER_INSTRUCTION = """
You are a plan-builder sub-agent. You produce a
detailed, actionable execution plan.

Input:
- goal: {goal}
- skills: {skills}
- agent_descriptions: {agents}
- memory: {memory}
- current plan {plan}

Requirements:
- Use ONLY agent names that appear in agent_descriptions.
- Keep each step specific and concise.
- List steps ONLY for the current stage.

Strictly Follow the JSON output.
"""


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------
output_schema=ExecutionPlan.model_json_schema()
output_schema.update({"additional_properties":False})
print(output_schema)
plan_builder_agent = LlmAgent(
    name="plan_builder_agent",
    model=LiteLlm(
        model=_model_name,
        base_url=_model_base_url,
        api_key=_model_api_key
    ),
    description=(
        "Produces a detailed ExecutionPlan JSON. ALWAYS call it when creating/updating plans."
    ),
    instruction=_PLAN_BUILDER_INSTRUCTION,
    input_schema=PlanBuilderInput,
    output_schema=ExecutionPlan,
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
)
