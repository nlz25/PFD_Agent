"""Summarize-agent submodule for thinking phase post-execution synthesis.

This agent summarizes execution outcomes using:
- confirmed goal (session state)
- approved execution plan (session state)
- execution history / results (session history passed by caller)
"""

from __future__ import annotations

import os
import json
import logging
from typing import Any, Dict, List, Literal

from google.adk.agents import LlmAgent, InvocationContext
from google.adk.models.lite_llm import LiteLlm
from google.adk.events import Event, EventActions
from google.genai.types import Content, Part
from pydantic import BaseModel, Field

from ..constants import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL

_model_name = os.environ.get("LLM_MODEL", LLM_MODEL)
_model_api_key = os.environ.get("LLM_API_KEY", LLM_API_KEY)
_model_base_url = os.environ.get("LLM_BASE_URL", LLM_BASE_URL)

logger = logging.getLogger()


class ExecutionSummaryInput(BaseModel):
    """Input for execution-result summarization."""

    goal: str = Field(
        ...,
        description="Confirmed user goal from session state.",
        max_length=500,
    )
    plan: Dict[str, Any] = Field(
        ...,
        description="Approved execution plan object from session state.",
    )
    execution_history: List[str] = Field(
        default_factory=list,
        description=(
            "Optional execution key-event history override. "
            "Primary source is session state variable `execution_history` (list of strings)."
        ),
    )


class ExecutionSummary(BaseModel):
    """Structured summary of execution outcomes for decision making."""

    goal: str = Field(
        ...,
        description="Goal that was evaluated.",
        max_length=500,
    )
    completion_status: Literal["completed", "in_progress", "blocked"] = Field(
        ...,
        description="Overall execution status against the approved plan and goal.",
    )
    completed_steps: List[int] = Field(
        default_factory=list,
        description="Step numbers completed successfully.",
    )
    pending_steps: List[int] = Field(
        default_factory=list,
        description="Step numbers not yet completed.",
    )
    failed_steps: List[int] = Field(
        default_factory=list,
        description="Step numbers that failed or were blocked.",
    )
    key_results: str = Field(
        ...,
        description="Concise summary of what was produced or learned during execution.",
        max_length=1200,
    )
    artifacts: List[str] = Field(
        default_factory=list,
        description="Important generated files/paths/IDs observed in execution outputs.",
    )
    blockers: str = Field(
        default="",
        description="Main blockers or errors preventing completion. Empty if none.",
        max_length=800,
    )
    recommended_next_action: Literal[
        "continue_execution",
        "request_user_input",
        "replan",
        "mark_complete",
    ] = Field(
        ...,
        description="Recommended next action for the orchestrator.",
    )
    concise_summary: str = Field(
        ...,
        description="User-facing one-paragraph execution summary.",
        max_length=800,
    )


_SUMMARIZE_INSTRUCTION = """
You are a summarize-agent used by the thinking agent after or during execution.

Session state context:
- goal: {goal}
- plan: {plan}
- execution_history: {execution_history}

Task:
1) Compare execution history against plan steps and goal.
2) Determine completion status and infer completed/pending/failed step numbers.
3) Summarize key outcomes and extract concrete artifacts (absolute paths/IDs when present).
4) Provide blockers and recommend one next action.

Source-of-truth rule:
- Use session state `execution_history` as the default execution transcript.

Rules:
- Be conservative: if evidence is incomplete, use in_progress.
- Do not invent artifacts or metrics; only include what appears in history.
- concise_summary must be one short paragraph (2-4 sentences).
- If all required steps are complete and goal achieved -> mark_complete.
- If blocked by errors or missing prerequisites -> replan or request_user_input as appropriate.
"""


class SummarizeAgent(LlmAgent):
    """Summarize execution history and render final output as readable text."""

    @staticmethod
    def _extract_text(event: Event) -> str:
        """Extract text from event content parts."""
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

    @staticmethod
    def _render_summary_text(summary: ExecutionSummary) -> str:
        """Render a user-friendly text block from structured summary."""
        completed = ", ".join(str(i) for i in summary.completed_steps) or "none"
        pending = ", ".join(str(i) for i in summary.pending_steps) or "none"
        failed = ", ".join(str(i) for i in summary.failed_steps) or "none"
        artifacts = "\n".join(f"- {item}" for item in summary.artifacts) if summary.artifacts else "- none"
        blockers = summary.blockers.strip() or "none"

        return (
            "📋 Execution Summary\n"
            f"Goal: {summary.goal}\n"
            f"Status: {summary.completion_status}\n"
            f"Completed steps: {completed}\n"
            f"Pending steps: {pending}\n"
            f"Failed steps: {failed}\n"
            f"Next action: {summary.recommended_next_action}\n\n"
            f"Key results:\n{summary.key_results}\n\n"
            f"Artifacts:\n{artifacts}\n\n"
            f"Blockers: {blockers}\n\n"
            f"Concise summary: {summary.concise_summary}"
        )

    async def _run_async_impl(self, ctx: InvocationContext):
        """Run summarization, then post-process final event into proper text."""
        final_event: Event | None = None

        try:
            async for event in super()._run_async_impl(ctx):
                if event.is_final_response() and event.content:
                    final_event = event
                    continue
                yield event

            if final_event is None:
                logger.warning("SummarizeAgent: no final response event found")
                yield Event(
                    content=Content(parts=[Part(text="⚠️ Unable to generate execution summary.")]),
                    author=self.name,
                )
                return

            raw_text = self._extract_text(final_event)
            summary_data: ExecutionSummary | None = None

            try:
                parsed = json.loads(raw_text)
                summary_data = ExecutionSummary(**parsed)
            except Exception:
                logger.debug("SummarizeAgent: final response was not strict JSON, trying fallback parse")
                try:
                    summary_data = ExecutionSummary.model_validate_json(raw_text)
                except Exception as parse_error:
                    logger.warning(f"SummarizeAgent: failed to parse final summary: {parse_error}")

            if summary_data is None:
                yield final_event
                return

            rendered_text = self._render_summary_text(summary_data)
            
            state_update = {
                "summarize": summary_data.model_dump(),
                "execution_summary_text": summary_data.concise_summary,
                "recommended_next_action": summary_data.recommended_next_action,
            }
            logger.info(f"[{self.name}]: Updating session state with summary and recommended action: {state_update['recommended_next_action']}")
            logger.info(f"[{self.name}]: Updating session state with summary and completion status: {summary_data.completion_status}")
            # Update session state "stop execution"
            if summary_data.completion_status == "completed" or "blocked" or len(summary_data.pending_steps) == 0:
                state_update["execution_complete"] = True
                
            elif summary_data.completion_status == "in_progress":
                state_update["execution_complete"] = False
            
            event_action = EventActions(state_delta=state_update)
            yield Event(
                content=Content(parts=[Part(text=rendered_text)]),
                author=self.name,
                actions=event_action,
            )

        except Exception as exc:
            logger.error(f"SummarizeAgent runtime error: {exc}")
            yield Event(
                content=Content(parts=[Part(text=f"❌ Summarization error: {str(exc)}")]),
                author=self.name,
            )


summarize_agent = SummarizeAgent(
    name="summarize_agent",
    model=LiteLlm(
        model=_model_name,
        base_url=_model_base_url,
        api_key=_model_api_key,
    ),
    description=(
        "Summarizes execution progress/results from goal, approved plan, and execution history, "
        "returning a structured ExecutionSummary."
    ),
    instruction=_SUMMARIZE_INSTRUCTION,
    #input_schema=ExecutionSummaryInput,
    output_schema=ExecutionSummary,
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
)
