from __future__ import annotations

import logging
import os
from typing import List, Literal, Optional

from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools.function_tool import FunctionTool
from google.adk.tools.tool_context import ToolContext
from pydantic import BaseModel, Field, ValidationError, model_validator

from ...constants import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL
from ...skill import ALL_SKILLS_TOOLSET
from ...tools.remoteagent_tool import load_remote_a2a_agents
from ...tools.util_tools import show_artifact, show_plot, show_structure
from ...tools.workspace_tools import run_bash, run_python

logger = logging.getLogger(__name__)

_model_name = os.environ.get("LLM_MODEL", LLM_MODEL)
_model_api_key = os.environ.get("LLM_API_KEY", LLM_API_KEY)
_model_base_url = os.environ.get("LLM_BASE_URL", LLM_BASE_URL)


class StepExecutorInput(BaseModel):
    step_number: int = Field(description="1-based index of this step in the plan")
    action: str = Field(description="Action description from the plan step")
    suggested_skills: List[str] = Field(description="Ordered list of skill names suggested by the planner for this step")
    workspace_dir: str = Field(description="Absolute path to the session workspace directory")
    prior_context: Optional[str] = Field(
        default=None,
        description="Condensed summaries of prior completed steps for context",
    )


class StepExecutorResult(BaseModel):
    status: Literal["success", "needs_replanning"]
    key_results: str = Field(
        default="",
        description="Bullet-point list of key findings, values, and produced files",
    )
    artifacts: list[str] = Field(
        default_factory=list,
        description="Absolute paths of generated files or artifacts",
    )
    concise_summary: str = Field(
        default="",
        description="Short user-facing paragraph describing what was done",
    )
    replan_reason: Optional[str] = Field(
        default=None,
        description="Why replanning is needed (only set when status=needs_replanning)",
    )

    @model_validator(mode="after")
    def _fill_missing_fields(self) -> "StepExecutorResult":
        # If the LLM returned only one of the two summary fields, mirror it to the other.
        if not self.key_results and self.concise_summary:
            self.key_results = self.concise_summary
        elif not self.concise_summary and self.key_results:
            self.concise_summary = self.key_results
        return self


_STEP_EXECUTOR_INSTRUCTION = """
You are a focused step executor. Execute the single plan step provided in your input.

## Your task
1. Review `suggested_skills` from your input. Call `load_skill` for each skill you deem
   relevant to the action. You may also load additional skills.
2. Decide whether to execute directly or decompose:
   - **Execute directly** if the action is atomic (single tool call or single well-defined operation).
   - **Decompose** by calling `run_sub_steps` if the action requires multiple distinct phases
     or sequential decisions..
   - **Depth limit**: if `recursion_depth` in your context is already at the maximum,
     `run_sub_steps` will refuse — execute directly or submit needs_replanning.
3. If you called `run_sub_steps`, inspect the returned `sub_results` list. Each entry
   contains the result of one sub-step. Aggregate their `key_results` and `artifacts`
   then call `submit_step_result` with the combined outcome.

## Reporting results (REQUIRED)
When done, call `submit_step_result` with:
- `status`: "success" or "needs_replanning"
- `key_results`: bullet-point list of key findings, values, and produced files
- `concise_summary`: short user-facing paragraph describing what was done
- `artifacts`: list of absolute paths of all generated files
- `replan_reason`: why replanning is needed (only when status=needs_replanning, else omit)

If `submit_step_result` returns a validation error, fix the fields and call it again.
Do NOT write JSON text — always use the `submit_step_result` tool.

## Execution rules
- Work inside `workspace_dir` for all file operations.
- Use `run_python` or `run_bash` for computation. Do not fabricate outputs.
- Include ALL generated files with their absolute paths in `artifacts`.
- Do not retry indefinitely on failure — call `submit_step_result` with needs_replanning.
- **Long-running background jobs (e.g., dpdisp via tmux):** You MUST poll until the
  background process exits AND the expected output files exist locally before calling
  `submit_step_result`. Submitting a job is NOT the same as completing it. Poll with
  `run_bash` every 30–60 seconds (e.g., check `tmux has-session`). Only report success
  after outputs are downloaded. If the process exits but outputs are missing, attempt
  recovery before reporting needs_replanning.

## Prior context
If `prior_context` is provided, use it to understand what prior steps produced
and avoid repeating completed work.
"""


async def run_sub_steps(
    sub_steps: List[dict],
    tool_context: ToolContext,
) -> dict:
    """Decompose this step into sub-steps and spawn child step executors.

    Call this when the current action requires multiple distinct phases or
    sequential decisions. Sub-steps run concurrently when independent.
    After this returns, inspect `sub_results` and aggregate them into
    a single `submit_step_result` call.

    Args:
        sub_steps: List of sub-step dicts, each containing:
            - action (str, required): what to do
            - suggested_skills (list[str], optional): skills to load
            - prior_context (str, optional): context from earlier sub-steps
    """
    from .step_executor_runner import _run_sub_steps_impl  # lazy import avoids circular dep

    return await _run_sub_steps_impl(sub_steps, tool_context=tool_context)


def submit_step_result(
    status: str,
    key_results: str,
    concise_summary: str,
    tool_context: ToolContext,
    artifacts: Optional[List[str]] = None,
    replan_reason: Optional[str] = None,
) -> dict:
    """Submit the result of this step execution.

    Call this tool when you have finished executing the step. Replaces writing
    a JSON response. If validation fails, fix the reported fields and retry.

    Args:
        status: "success" or "needs_replanning"
        key_results: Bullet-point list of key findings and produced files
        concise_summary: Short user-facing paragraph describing what was done
        artifacts: Absolute paths of all generated files (empty list if none)
        replan_reason: Why replanning is needed (only when status=needs_replanning)
    """
    try:
        result = StepExecutorResult(
            status=status,  # type: ignore[arg-type]
            key_results=key_results,
            concise_summary=concise_summary,
            artifacts=artifacts or [],
            replan_reason=replan_reason,
        )
        tool_context.state["_step_result"] = result.model_dump()
        return {"status": "ok", "message": "Step result submitted successfully."}
    except ValidationError as exc:
        return {
            "status": "error",
            "errors": exc.errors(),
            "message": "Validation failed. Fix the errors and call submit_step_result again.",
        }
    except Exception as exc:
        return {"status": "error", "message": f"Unexpected error: {exc}"}


step_executor_agent = LlmAgent(
    name="step_executor",
    model=LiteLlm(
        model=_model_name,
        base_url=_model_base_url,
        api_key=_model_api_key,
    ),
    description=(
        "Executes a single plan step in an isolated session. "
        "Receives structured input with action and skill name; loads skill instructions autonomously."
    ),
    instruction=_STEP_EXECUTOR_INSTRUCTION,
    input_schema=StepExecutorInput,
    tools=[
        FunctionTool(run_sub_steps),
        FunctionTool(submit_step_result),
        FunctionTool(run_python),
        FunctionTool(run_bash),
        ALL_SKILLS_TOOLSET,
        FunctionTool(show_plot),
        FunctionTool(show_structure),
        FunctionTool(show_artifact),
    ],
    sub_agents=load_remote_a2a_agents(),
)
