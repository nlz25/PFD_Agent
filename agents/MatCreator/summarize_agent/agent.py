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
import re
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
_summarize_agent_max_attempts = int(os.environ.get("SUMMARIZE_AGENT_MAX_ATTEMPTS", "5"))

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
    recommended_next_action: Literal[
        "continue_execution",
        "request_user_input_but_no_need_to_replan",
        "replan",
        "mark_complete",
    ] = Field(
        ...,
        description="Recommended next action for the orchestrator. ",
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
"""

_RETRY_INSTRUCTION_TEMPLATE = """

Previous output failed schema validation:
{validation_error}

Retry now and output ONLY a valid JSON object that conforms to the ExecutionSummary schema.
Do not include markdown fences or explanatory text.
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
    def _extract_json_candidate(raw_text: str) -> str:
        """Extract JSON candidate from raw text, tolerating fenced responses."""
        text = raw_text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text).strip()

        match = re.search(r"\{[\s\S]*\}", text)
        return match.group(0) if match else text

    @classmethod
    def _validate_execution_summary(
        cls, event: Event | None
    ) -> tuple[dict[str, Any] | None, str | None]:
        """Validate final event payload against ExecutionSummary schema."""
        if event is None:
            return None, "No final response event was produced."

        raw_text = cls._extract_text(event)
        if not raw_text:
            return None, "Final response event is empty."

        candidate = cls._extract_json_candidate(raw_text)

        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError as exc:
            return None, f"JSON decode error: {exc}"

        try:
            from pydantic import ValidationError
            validated = ExecutionSummary.model_validate(parsed)
        except ValidationError as exc:
            return None, f"Schema validation error: {exc}"

        return validated.model_dump(), None

    @staticmethod
    def _render_summary_text(summary: ExecutionSummary) -> str:
        """Render a user-friendly text block from structured summary."""
        completed = ", ".join(str(i) for i in summary.completed_steps) or "none"
        pending = ", ".join(str(i) for i in summary.pending_steps) or "none"
        failed = ", ".join(str(i) for i in summary.failed_steps) or "none"
        artifacts = "\n".join(f"- {item}" for item in summary.artifacts) if summary.artifacts else "- none"
        #blockers = summary.blockers.strip() or "none"

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
            f"Concise summary: {summary.concise_summary}"
        )

    @staticmethod
    def _make_fallback_state_event(author: str, message: str) -> Event:
        """Yield a state-delta event that terminates the execution loop for replan."""
        state_update = {
            "execution_complete": True,
            "recommended_next_action": "replan",
            "completion_status": "blocked",
        }
        return Event(
            content=Content(parts=[Part(text=message)]),
            author=author,
            actions=EventActions(state_delta=state_update),
        )

    async def _run_async_impl(self, ctx: InvocationContext):
        """Run summarization with retry until output matches ExecutionSummary schema."""
        base_instruction = self.instruction
        last_error = ""
        validated_data: dict[str, Any] | None = None
        final_event: Event | None = None

        try:
            for attempt in range(1, _summarize_agent_max_attempts + 1):
                logger.info(f"SummarizeAgent attempt {attempt}/{_summarize_agent_max_attempts}")
                if attempt == 1:
                    self.instruction = base_instruction
                else:
                    self.instruction = (
                        base_instruction
                        + _RETRY_INSTRUCTION_TEMPLATE.format(validation_error=last_error)
                    )

                buffered_events: List[Event] = []
                candidate_final: Event | None = None

                async for event in super()._run_async_impl(ctx):
                    if event.is_final_response() and event.content:
                        candidate_final = event
                    else:
                        buffered_events.append(event)

                validated_data, validation_error = self._validate_execution_summary(candidate_final)
                if validation_error is None:
                    for event in buffered_events:
                        yield event
                    final_event = candidate_final
                    break

                last_error = validation_error
                logger.warning(
                    "SummarizeAgent schema validation failed on attempt %d/%d: %s",
                    attempt,
                    _summarize_agent_max_attempts,
                    validation_error,
                )

            if validated_data is None or final_event is None:
                logger.warning("SummarizeAgent: all attempts exhausted without valid output")
                yield self._make_fallback_state_event(
                    self.name,
                    f"⚠️ Unable to generate valid execution summary after {_summarize_agent_max_attempts} attempts. Routing back for replanning.",
                )
                return

            summary_data = ExecutionSummary.model_validate(validated_data)

            rendered_text = self._render_summary_text(summary_data)
            
            
            ## check and determine the execution complete status
            if len(summary_data.failed_steps) == 0 and len(summary_data.pending_steps) == 0:
                logger.info(f"Execution appears fully completed. Marking execution_complete=True in session state.")
                completion_status = "completed"
                recommended_next_action = "mark_complete"
            else:
                completion_status = summary_data.completion_status or "in_progress"
                recommended_next_action = summary_data.recommended_next_action
                logger.info(f"Execution not fully completed. Marking execution_complete=False in session state. Reported completion_status: {completion_status}")
                
            state_update = {
                "summarize": summary_data.model_dump(),
                "completion_status": completion_status,
                "execution_summary_text": summary_data.concise_summary,
                "recommended_next_action": recommended_next_action,
            }
            logger.info(f"[{self.name}]: Updating session state with summary and recommended action: {state_update['recommended_next_action']}")
            logger.info(f"[{self.name}]: Updating session state with summary and completion status: {completion_status}")
            
            # Update session state "stop execution"
            if recommended_next_action == "continue_execution":
                state_update["execution_complete"] = False
            else:
                state_update["execution_complete"] = True
            
            event_action = EventActions(state_delta=state_update)
            yield Event(
                content=Content(parts=[Part(text=rendered_text)]),
                author=self.name,
                actions=event_action,
            )

        except Exception as exc:
            logger.error(f"SummarizeAgent runtime error: {exc}")
            yield self._make_fallback_state_event(
                self.name,
                f"❌ Summarization error: {str(exc)}. Routing back for replanning.",
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
