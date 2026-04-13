"""Plan validation tool for the ThinkingAgent.

Provides a Pydantic-backed ``validate_plan`` function tool that the thinking
agent calls after generating a plan to validate schema conformance and
persist it to session state.
"""

from __future__ import annotations

import logging
from typing import List

from google.adk.tools.tool_context import ToolContext
from pydantic import BaseModel, Field, ValidationError, field_validator

from ..skill import _load_skill_registry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class PlanStep(BaseModel):
    """Single step in the execution plan."""

    step_number: int = Field(..., description="Sequential step number (1, 2, 3, ...)")
    skill: str = Field(
        ...,
        description="Skill name used by the executor to load relevant tools and instruction.",
        min_length=1,
    )
    action: str = Field(
        ...,
        description="Clear, concise description of what this step does (1-2 sentences)",
        max_length=500,
    )

    @field_validator("skill")
    @classmethod
    def _validate_skill_name(cls, value: str) -> str:
        allowed_names = set(_load_skill_registry().keys())
        if value not in allowed_names:
            allowed = ", ".join(sorted(allowed_names)) or "<none loaded>"
            raise ValueError(
                f"Invalid skill '{value}'. Allowed skills are: {allowed}"
            )
        return value


class ExecutionPlan(BaseModel):
    """Structured execution plan for user approval."""
    steps: List[PlanStep] = Field(
        ...,
        description="Ordered list of detailed steps in the CURRENT stage, ONLY includes DETERMINED steps",
        min_items=1,
        max_items=10,
    )
    additional_notes: str = Field(
        ...,
        description="Any extra information or considerations for the user",
        max_length=500,
    )




# ---------------------------------------------------------------------------
# validate_plan tool
# ---------------------------------------------------------------------------

def validate_plan(plan: dict, tool_context: ToolContext) -> dict:
    """Validate and commit a plan to session state.

    Call this after drafting a plan to validate it against the schema
    and persist it. On success the plan is stored under the 'plan' session
    state key and returned. On failure the validation errors are returned so
    you can fix and retry.

    Args:
        plan: Dict with 'steps' (list of {step_number, skill, action}) and
              'additional_notes' (str).
    """
    try:
        validated = ExecutionPlan(**plan)
        tool_context.state["plan"] = validated.model_dump()
        return {
            "status": "ok",
            "plan": validated.model_dump(),
            "message": f"Plan validated and saved with {len(validated.steps)} steps.",
        }
    except ValidationError as exc:
        return {
            "status": "error",
            "errors": exc.errors(),
            "message": "Plan validation failed. Fix the errors and re-call validate_plan.",
        }
    except Exception as exc:
        return {
            "status": "error",
            "message": f"Unexpected error: {exc}",
        }
