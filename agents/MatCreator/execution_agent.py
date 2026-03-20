"""Execution agent for MatCreator - executes approved plans using skill-scoped toolsets."""

from __future__ import annotations

import hashlib
import json
import os
import logging
from typing import Dict, Any, List
from google.adk.agents import LlmAgent, InvocationContext
from google.adk.agents.callback_context import CallbackContext
from google.adk.apps.llm_event_summarizer import LlmEventSummarizer
from google.adk.models.lite_llm import LiteLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.tools.function_tool import FunctionTool
from google.adk.tools.agent_tool import AgentTool
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from google.adk.events import Event, EventActions
from google.genai.types import Content, Part
from pydantic import Field, BaseModel
from .constants import (
    LLM_API_KEY, LLM_BASE_URL, 
    LLM_MODEL, 
    EXECUTION_ENABLE_WITHIN_INVOCATION_COMPACTION,
    EXECUTION_COMPACT_KEEP_TAIL,
    EXECUTION_COMPACT_EVERY_EVENTS,
)
from .callbacks import after_tool_callback
from .tools import TOOLSETS
from .thinking_agent.skill import (
    SKILL_LIBRARY,
    Skill,
)
from .thinking_agent.workspace_tools import (
    run_python,
    run_bash,
    run_python_file
)

_model_name = os.environ.get("LLM_MODEL", LLM_MODEL)
_model_api_key = os.environ.get("LLM_API_KEY", LLM_API_KEY)
_model_base_url = os.environ.get("LLM_BASE_URL", LLM_BASE_URL)
_EXECUTION_COMPACT_EVERY_EVENTS = int(os.environ.get("EXECUTION_COMPACT_EVERY_EVENTS", str(EXECUTION_COMPACT_EVERY_EVENTS)))
_EXECUTION_COMPACT_KEEP_TAIL = int(os.environ.get("EXECUTION_COMPACT_KEEP_TAIL", str(EXECUTION_COMPACT_KEEP_TAIL)))



def _env_flag(name: str, default: bool) -> bool:
    """Parse common truthy/falsey env var values with a safe default."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in {"0", "true", "t", "yes", "y", "on"}:
        return True
    if value in {"1", "false", "f", "no", "n", "off"}:
        return False
    return default


_EXECUTION_ENABLE_WITHIN_INVOCATION_COMPACTION = _env_flag(
    "EXECUTION_ENABLE_WITHIN_INVOCATION_COMPACTION",
    EXECUTION_ENABLE_WITHIN_INVOCATION_COMPACTION,
)

logger = logging.getLogger()

_EXECUTION_INSTRUCTION = """
You are the execution agent. Your sole responsibility is to execute an approved plan by selecting the right skill and using direct tools.

Plans: {detailed_steps}

Skill catalog: {skills}

Current skill: {execution_active_skill}

Current skill instruction: {execution_active_skill_instruction}

Previous execution history: {summarize}

Critical rule:
- Before executing EACH plan step, call `load_skill_context(skill_name=<step.skill>)`.
- After loading context, follow the returned skill instruction and restrict tool usage to current skill tools.
- After completing EACH plan stage, call `summarize_agent` to record what was done and the outcome.
- Stop execution if you can't find solution for multiple consecutive steps, and wait for replanning.
"""


def break_execution(tool_context: ToolContext) -> dict[str, str]:
    """If user want to stop execution immediately, call THIS to route control back to replanning."""
    tool_context.state["execution_complete"] = True
    tool_context.state["recommended_next_action"] = "replan"
    tool_context.state["force_break"] = True
    return {
        "status": "ok",
        "message": "Execution stopped. Control will return for replanning.",
    }



def load_skill_context(skill_name: str, tool_context: ToolContext) -> dict[str, Any]:
    """Load execution context for a skill by name from SKILL_LIBRARY.

    IMPORTANT: Call this tool BEFORE executing each plan step.
    Use the step's `skill` value as `skill_name` so the executor can load
    skill-specific instruction and the related tool scope for that step.
    """
    normalized = (skill_name or "").strip()
    if not normalized:
        return {
            "status": "error",
            "message": "skill_name is required.",
            "available_skills": sorted(SKILL_LIBRARY.keys()),
        }

    selected: Skill | None = SKILL_LIBRARY.get(normalized)
    if selected is None:
        lowered = normalized.lower()
        for name, skill in SKILL_LIBRARY.items():
            if name.lower() == lowered:
                selected = skill
                break

    if selected is None:
        return {
            "status": "error",
            "message": f"Skill '{skill_name}' not found in SKILL_LIBRARY.",
            "available_skills": sorted(SKILL_LIBRARY.keys()),
        }


    selected_tool_names: List[str] = []
    for tool_name in selected.needed_tools:
        if tool_name not in selected_tool_names:
            selected_tool_names.append(tool_name)


    tool_context.state["execution_active_skill"] = selected.name
    tool_context.state["execution_active_skill_instruction"] = selected.instruction
    tool_context.state["execution_selected_tool_names"] = selected_tool_names
    tool_context.state["execution_selected_tool_names_text"] = ", ".join(selected_tool_names) if selected_tool_names else "No explicit tools listed."
    tool_context.state["execution_active_skill_source"] = selected.source_path

    return {
        "status": "ok",
        "skill": selected.name,
        "instruction": tool_context.state["execution_active_skill_instruction"],
        "selected_tool_names": selected_tool_names,
        "message": f"Loaded skill context for '{selected.name}'.",
    }

def execution_after_tool_callback(
    tool: BaseTool,
    args: Dict[str, Any],
    tool_context: ToolContext,
    tool_response: Dict[str, Any],
):
    """Execution callback wrapper: log all tools and inject skill context after lookup."""
    after_tool_callback(tool=tool, args=args, tool_context=tool_context, tool_response=tool_response)

    tool_name = getattr(tool, "name", "")
    if tool_name != "load_skill_context":
        return None

    if not isinstance(tool_response, dict):
        return None

    if tool_response.get("status") != "ok":
        return None

    selected = tool_response.get("selected_tool_names") or []
    if isinstance(selected, list):
        tool_context.state["execution_selected_tool_names"] = selected
        tool_context.state["execution_selected_tool_names_text"] = ", ".join(selected) if selected else "No explicit tools listed."

    instruction = tool_response.get("instruction")
    if isinstance(instruction, str) and instruction.strip():
        tool_context.state["execution_active_skill_instruction"] = instruction

    return None


# Tools that must always remain available regardless of active skill
_ALWAYS_AVAILABLE_EXECUTION_TOOLS: set[str] = {
    "load_skill_context",
    "break_execution",
    "summarize_agent",
    "plot_agent",
}


def execution_before_model_callback(
    callback_context: CallbackContext,
    llm_request: LlmRequest,
) -> None:
    """Filter LlmRequest tool declarations to the active skill's allowed set.

    ``_process_agent_tools`` in ADK iterates ``agent.tools`` directly and
    populates ``llm_request`` before this callback fires, so this is the
    only reliable place to restrict what tools the model actually sees.
    """
    state = callback_context.state
    active_names: list[str] = list(state.get("execution_selected_tool_names") or [])

    # No skill loaded yet — allow all tools so the LLM can call load_skill_context.
    if not active_names:
        return None

    allowed = set(active_names) | _ALWAYS_AVAILABLE_EXECUTION_TOOLS

    # Filter the dispatch dictionary so unknown tool calls are rejected.
    llm_request.tools_dict = {
        name: tool
        for name, tool in llm_request.tools_dict.items()
        if name in allowed
    }

    # Filter the function declarations that are sent to the model.
    if llm_request.config and llm_request.config.tools:
        for tool in llm_request.config.tools:
            if getattr(tool, "function_declarations", None):
                tool.function_declarations = [
                    decl
                    for decl in tool.function_declarations
                    if getattr(decl, "name", None) in allowed
                ]

    logger.info(
        "before_model_callback: filtered tools to allowed=%s for skill '%s'",
        allowed,
        state.get("execution_active_skill", "unknown"),
    )
    return None


class ExecutionAgent(LlmAgent):
    """Agent that executes approved plans by orchestrating domain agents."""

    # Keep for reference / logging; real filtering is in execution_before_model_callback.
    _ALWAYS_AVAILABLE_TOOLS: set[str] = _ALWAYS_AVAILABLE_EXECUTION_TOOLS

    @staticmethod
    def _is_compaction_event(event: Event) -> bool:
        return bool(event.actions and event.actions.compaction)

    @classmethod
    def _events_to_compact_for_current_invocation(
        cls,
        *,
        events: List[Event],
        invocation_id: str,
        keep_tail: int,
    ) -> List[Event]:
        """Pick raw events for mid-invocation compaction for a single invocation id."""
        raw_events = [
            event
            for event in events
            if event.invocation_id == invocation_id and not cls._is_compaction_event(event)
        ]
        if len(raw_events) <= keep_tail:
            return []

        last_compacted_end_ts = 0.0
        for event in events:
            if event.invocation_id != invocation_id:
                continue
            if not cls._is_compaction_event(event):
                continue
            compaction = event.actions.compaction
            if compaction and compaction.end_timestamp is not None:
                last_compacted_end_ts = max(last_compacted_end_ts, compaction.end_timestamp)

        new_raw_events = [e for e in raw_events if e.timestamp > last_compacted_end_ts]
        if len(new_raw_events) <= keep_tail:
            return []

        return new_raw_events[:-keep_tail]

    @classmethod
    async def _compact_invocation_context_events(
        cls,
        *,
        ctx: InvocationContext,
        invocation_id: str,
        summarizer: LlmEventSummarizer,
    ) -> bool:
        """Compact current invocation events and append a compaction event to the session."""
        keep_tail = max(2, _EXECUTION_COMPACT_KEEP_TAIL)
        session_events = ctx.session.events
        new_session_events=[]
        candidate_events = cls._events_to_compact_for_current_invocation(
            events=session_events,
            invocation_id=invocation_id,
            keep_tail=keep_tail,
        )
        logger.info(
            "Mid-invocation compaction check: invocation_id=%s total_session_events=%d candidates=%d",
            invocation_id,
            len(session_events),
            len(candidate_events),
        )
        if not candidate_events:
            return False

        compaction_event = await summarizer.maybe_summarize_events(events=candidate_events)
        if compaction_event is None:
            return False

        compaction_event.invocation_id = invocation_id
        compaction_event.branch = getattr(ctx, "branch", None)

        await ctx.session_service.append_event(session=ctx.session, event=compaction_event)

        new_compaction = compaction_event.actions.compaction if compaction_event.actions else None
        if not new_compaction:
            return False

        start_ts = new_compaction.start_timestamp
        end_ts = new_compaction.end_timestamp

        # Rebuild the in-memory session events to discard compacted raw events.
        # Keep non-target invocation events as-is, keep compaction events, and
        # for target invocation keep only raw events outside the compacted range.
        new_session_events: List[Event] = []
        for existing_event in session_events:
            if existing_event.invocation_id != invocation_id:
                new_session_events.append(existing_event)
                continue

            #if cls._is_compaction_event(existing_event):
            #    new_session_events.append(existing_event)
            #    continue

            if start_ts <= existing_event.timestamp <= end_ts:
                continue

            new_session_events.append(existing_event)

        # Ensure current compaction summary exists exactly once.
        has_same_range = any(
            cls._is_compaction_event(event)
            and event.actions
            and event.actions.compaction
            and event.invocation_id == invocation_id
            and event.actions.compaction.start_timestamp == start_ts
            and event.actions.compaction.end_timestamp == end_ts
            for event in new_session_events
        )
        if not has_same_range:
            # Insert summary at the start of the current invocation segment.
            insertion_index = next(
                (
                    index
                    for index, event in enumerate(new_session_events)
                    if event.invocation_id == invocation_id
                ),
                len(new_session_events),
            )
            new_session_events.insert(insertion_index, compaction_event)

        # Replace the live list object in place so downstream code sees compacted context.
        #session_events[:] = new_session_events
        ctx.session.events = new_session_events
        logger.info(
            "Mid-invocation compaction completed: invocation_id=%s compacted_events=%d new_total_events=%d",
            invocation_id,
            len(candidate_events),
            len(ctx.session.events),
        )

        ctx.session.state["execution_mid_invocation_compactions"] = int(
            ctx.session.state.get("execution_mid_invocation_compactions", 0)
        ) + 1
        compaction_text = cls._extract_event_text(compaction_event)
        if compaction_text:
            ctx.session.state["execution_mid_invocation_last_summary"] = compaction_text
        return True

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
            "execution_history": [],
            "execution_last_output": "",
            "execution_mid_invocation_event_count": 0,
            "execution_mid_invocation_compactions": 0,
            "execution_mid_invocation_last_summary": "",
            "execution_within_invocation_compaction_enabled": _EXECUTION_ENABLE_WITHIN_INVOCATION_COMPACTION,
            "execution_active_skill": "",
            "execution_active_skill_instruction": "",
            "execution_selected_tool_names": [],
            "execution_selected_tool_names_text": "Call load_skill_context before each step.",
            "skills": "\n".join(
                [
                    f"- {skill_id}: {skill.description}"
                    for skill_id, skill in SKILL_LIBRARY.items()
                ]
            ),
            "force_break": False,
        }
        ctx.session.state.update(state_update)
        compaction_summarizer = LlmEventSummarizer(
            llm=LiteLlm(
                model=_model_name,
                base_url=_model_base_url,
                api_key=_model_api_key,
            )
        )
        current_invocation_id = ""
        event_action = EventActions(state_delta=state_update)
        start_event = Event(
            #content=Content(parts=[Part(text=f"🚀 Starting execution: {goal}")]),
            author=self.name,
            actions=event_action
        )
        self._save_key_execution_event(ctx.session.state, start_event)
        yield start_event
        
        logger.info(f"Starting execution of plan: {goal}")
        logger.info(
            "Within-invocation compaction enabled: %s",
            _EXECUTION_ENABLE_WITHIN_INVOCATION_COMPACTION,
        )
    
        try:
            async for event in super()._run_async_impl(ctx):
                self._save_key_execution_event(ctx.session.state, event)
                if not current_invocation_id and getattr(event, "invocation_id", None):
                    current_invocation_id = str(event.invocation_id)
                ctx.session.state["execution_mid_invocation_event_count"] = int(
                    ctx.session.state.get("execution_mid_invocation_event_count", 0)
                ) + 1
                event_count = int(ctx.session.state["execution_mid_invocation_event_count"])
                if (
                    _EXECUTION_ENABLE_WITHIN_INVOCATION_COMPACTION
                    and current_invocation_id
                    and event_count % max(4, _EXECUTION_COMPACT_EVERY_EVENTS) == 0
                ):
                    try:
                        did_compact = await self._compact_invocation_context_events(
                            ctx=ctx,
                            invocation_id=current_invocation_id,
                            summarizer=compaction_summarizer,
                        )
                        if did_compact:
                            logger.info(
                                "Mid-invocation compaction appended to session events "
                                f"(invocation_id={current_invocation_id}, event_count={event_count})"
                            )
                    except Exception as compaction_exc:
                        logger.warning(f"Mid-invocation compaction skipped: {compaction_exc}")
                yield event
            logger.info("Plan execution completed")
            
        except Exception as e:
            logger.error(f"Execution failed: {e}")
            error_event = Event(
                content=Content(parts=[Part(text=f"❌ Execution error: {str(e)}")]),
                author=self.name
            )
            self._save_key_execution_event(ctx.session.state, error_event)
            yield error_event
        finally:
            logger.info("Execution agent finished (assessment agent will determine next steps)")

_SUMMARIZE_INSTRUCTION = """
You summarize key outcomes and extract concrete artifacts.
Use absolute paths for artifacts.

Session state context:
- goal: {goal}
- plan: {plan}

Output ONLY a JSON object — no markdown fences, no extra text:
{{
  "key_results": "<concise summary of what was produced or learned during execution>",
  "artifacts": ["<absolute path or ID of important generated files/paths/IDs>"],
  "concise_summary": "<user-facing one-paragraph execution summary>"
}}
"""

class ExecutionSummary(BaseModel):
    """Structured summary of execution outcomes for decision making."""
    key_results: str = Field(
        ...,
        description="Concise summary of what was produced or learned during execution.",
        max_length=1200,
    )
    artifacts: List[str] = Field(
        default_factory=list,
        description="Important generated files/paths/IDs observed in execution outputs.",
    )
    concise_summary: str = Field(
        ...,
        description="User-facing one-paragraph execution summary.",
        max_length=800,
    )

summarize_agent = LlmAgent(
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
)



execution_agent = ExecutionAgent(
        name="execution_agent",
        model=LiteLlm(
            model=_model_name,
            base_url=_model_base_url,
            api_key=_model_api_key,
        ),
        description="Executes approved plans by delegating to domain-specific agents in sequence.",
        instruction=_EXECUTION_INSTRUCTION,
        before_model_callback=execution_before_model_callback,
        after_tool_callback=execution_after_tool_callback,
        disallow_transfer_to_parent=True,
        disallow_transfer_to_peers=True,
        tools=[
            FunctionTool(load_skill_context), 
            FunctionTool(break_execution), 
            FunctionTool(run_python),
            FunctionTool(run_bash),
            FunctionTool(run_python_file),
            AgentTool(summarize_agent),
            *TOOLSETS],
        sub_agents=[],
    )

