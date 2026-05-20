from __future__ import annotations

import asyncio
import logging
from contextlib import aclosing
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from google.adk.tools.agent_tool import ForwardingArtifactService
from google.adk.tools.tool_context import ToolContext
from google.genai import types

from ...workspace import get_session_workdir
from .step_executor import StepExecutorInput, StepExecutorResult, step_executor_agent
from ..graph_logger import AgentGraphLogger
from ..cancellation import (
    is_cancellation_requested,
    get_cancellation_reason,
    is_step_cancellation_requested,
    clear_step_cancellation,
)

logger = logging.getLogger(__name__)

# Time between flag-file polls in the watcher task.
# Event-count polling is unreliable when events are sparse (e.g. mid-LLM-call),
# so we use a wall-clock interval instead.
_CANCEL_POLL_INTERVAL = 0.5  # seconds


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _response_dict(response) -> dict:
    """Best-effort conversion for ADK function response payloads."""
    if isinstance(response, dict):
        return response
    try:
        return dict(response or {})
    except (TypeError, ValueError):
        return {}


def _append_unique(values: list[str], value) -> None:
    if isinstance(value, str) and value and value not in values:
        values.append(value)


async def _watch_for_cancellation(
    target: asyncio.Task,
    session_id: str,
    step_number: int,
) -> None:
    """Poll cancellation flags at fixed intervals; call target.cancel() when flagged.

    Runs concurrently with the step execution task.  Using task.cancel() raises
    CancelledError at the target's current await point (including in-flight HTTP
    calls to the LLM), which is the only reliable way to stop a running coroutine.
    """
    try:
        while not target.done():
            await asyncio.sleep(_CANCEL_POLL_INTERVAL)
            if (
                is_cancellation_requested(session_id)
                or is_step_cancellation_requested(session_id, step_number)
            ):
                logger.warning(
                    "[CANCEL] Step %d watcher triggered task cancellation (session=%s)",
                    step_number, session_id,
                )
                target.cancel()
                return
    except asyncio.CancelledError:
        pass


async def _stream_step_events(
    runner: Runner,
    session,
    content,
    step_id: str,
    tool_context: ToolContext,
    graph: AgentGraphLogger,
) -> tuple[dict, list[str]]:
    """Stream events from the step executor sub-agent and collect results.

    Returns (step_state_delta, plot_paths).  Raises CancelledError if the task
    is cancelled mid-stream.
    """
    step_state_delta: dict = {}
    pending_tool_calls: dict[str, dict] = {}
    plot_paths: list[str] = []

    async with aclosing(
        runner.run_async(
            user_id=session.user_id, session_id=session.id, new_message=content
        )
    ) as agen:
        async for event in agen:
            if event.actions and event.actions.state_delta:
                tool_context.state.update(event.actions.state_delta)
                step_state_delta.update(event.actions.state_delta)

            if event.content:
                for part in event.content.parts:
                    fc = getattr(part, "function_call", None)
                    fr = getattr(part, "function_response", None)
                    is_thought = getattr(part, "thought", False)
                    text = getattr(part, "text", None)

                    if is_thought and text:
                        graph.log_conversation_event(step_id, {
                            "timestamp": _now(),
                            "author": event.author,
                            "type": "thought",
                            "content": text,
                        })
                    elif text and not fc and not fr:
                        graph.log_conversation_event(step_id, {
                            "timestamp": _now(),
                            "author": event.author,
                            "type": "text",
                            "content": text,
                        })
                    elif fc and not is_thought:
                        pending_tool_calls[fc.name] = {
                            "name": fc.name,
                            "args_summary": str(dict(fc.args or {}))[:300],
                            "start_time": _now(),
                        }
                        graph.log_conversation_event(step_id, {
                            "timestamp": _now(),
                            "author": event.author,
                            "type": "function_call",
                            "content": f"{fc.name}({str(dict(fc.args or {}))[:500]})",
                        })
                    elif fr:
                        response = _response_dict(fr.response)
                        _append_unique(plot_paths, response.get("plot_path"))

                        record = pending_tool_calls.pop(fr.name, {"name": fr.name, "start_time": _now()})
                        record["result_summary"] = str(fr.response)[:300]
                        record["end_time"] = _now()
                        graph.log_tool_call(step_id, record)
                        graph.log_conversation_event(step_id, {
                            "timestamp": _now(),
                            "author": event.author,
                            "type": "function_response",
                            "content": f"{fr.name} → {str(fr.response)[:500]}",
                        })

    return step_state_delta, plot_paths


async def run_step_executor(
    step_number: int,
    action: str,
    suggested_skills: list[str],
    workspace_dir: str,
    prior_context: Optional[str] = None,
    *,
    tool_context: ToolContext,
) -> dict:
    """Run step_executor as an isolated sub-agent, following AgentTool logic.

    Each invocation gets its own workspace subdirectory under the session workdir,
    overriding the LLM-supplied workspace_dir with a per-step path.
    """
    # Per-step workspace: {session_workdir}/{plan_exec_id}/step_{step_number}/
    # plan_exec_id is generated once per plan execution to avoid collisions across plans.
    session_id = tool_context.state.get("session_id", "default")
    plan_exec_id = tool_context.state.get("plan_exec_id")
    if not plan_exec_id:
        plan_exec_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        tool_context.state["plan_exec_id"] = plan_exec_id
    step_workspace = get_session_workdir(session_id) / plan_exec_id / f"step_{step_number}"
    step_workspace.mkdir(parents=True, exist_ok=True)
    logger.debug("[step_executor_runner] step %d workspace: %s", step_number, step_workspace)

    graph = AgentGraphLogger(session_id)
    parent_id = tool_context.state.get("_graph_exec_node_id", "orchestrator")
    step_id = f"{parent_id}__step_{step_number}"
    graph.log_node_start(step_id, "step", f"Step {step_number}", parent_id)

    # Serialize input as user message (matches AgentTool input_schema path)
    step_input = StepExecutorInput(
        step_number=step_number,
        action=action,
        suggested_skills=suggested_skills,
        workspace_dir=str(step_workspace),
        prior_context=prior_context,
    )
    content = types.Content(
        role="user",
        parts=[types.Part.from_text(text=step_input.model_dump_json(exclude_none=True))],
    )

    # Log input parameters
    graph.log_node_input(step_id, {
        "step_number": step_number,
        "action": action,
        "workspace_dir": str(step_workspace),
        "prior_context": prior_context,
        "suggested_skills": suggested_skills,
    })

    # Pre-step cancellation check — abort before creating the runner if flagged
    if is_cancellation_requested(session_id) or is_step_cancellation_requested(session_id, step_number):
        reason = get_cancellation_reason(session_id) or "user_requested"
        logger.warning(
            "[CANCEL] Step %d aborted before start (session=%s, reason=%s)",
            step_number, session_id, reason,
        )
        graph.log_node_complete(step_id, "failed", summary=f"Cancelled before start: {reason}")
        clear_step_cancellation(session_id, step_number)
        return {
            "status": "cancelled",
            "message": f"Step {step_number} skipped: execution cancellation was requested ({reason}).",
        }

    # Create runner with isolated session (mirrors AgentTool)
    invocation_context = tool_context._invocation_context
    child_app_name = (
        invocation_context.app_name if invocation_context else step_executor_agent.name
    )
    runner = Runner(
        app_name=child_app_name,
        agent=step_executor_agent,
        artifact_service=ForwardingArtifactService(tool_context),
        session_service=InMemorySessionService(),
        memory_service=InMemoryMemoryService(),
        credential_service=invocation_context.credential_service,
    )

    # Inherit parent state, override workspace_dir with per-step path
    state_dict = {
        k: v
        for k, v in tool_context.state.to_dict().items()
        if not k.startswith("_adk")
    }
    state_dict["workspace_dir"] = str(step_workspace)
    state_dict["_graph_exec_node_id"] = step_id

    session = await runner.session_service.create_session(
        app_name=child_app_name,
        user_id=invocation_context.user_id,
        state=state_dict,
    )

    # Wrap event streaming in a Task so task.cancel() can raise CancelledError at
    # the current await point (including in-flight LLM HTTP calls).  A sibling
    # watcher task polls cancellation flags on a wall-clock interval and cancels
    # the inner task when a flag is detected.
    inner_task = asyncio.create_task(
        _stream_step_events(runner, session, content, step_id, tool_context, graph)
    )
    watcher = asyncio.create_task(
        _watch_for_cancellation(inner_task, session_id, step_number)
    )

    cancelled = False
    step_state_delta: dict = {}
    plot_paths: list[str] = []
    try:
        step_state_delta, plot_paths = await inner_task
    except asyncio.CancelledError:
        cancelled = True
    finally:
        if not watcher.done():
            watcher.cancel()
        await asyncio.gather(watcher, return_exceptions=True)
        try:
            await asyncio.wait_for(runner.close(), timeout=5.0)
        except Exception:
            pass

    if cancelled:
        reason = get_cancellation_reason(session_id) or "user_requested"
        logger.warning(
            "[CANCEL] Step %d task cancelled (session=%s, reason=%s)",
            step_number, session_id, reason,
        )
        graph.log_node_complete(
            step_id, "failed", summary=f"Cancelled mid-step ({reason})"
        )
        clear_step_cancellation(session_id, step_number)
        return {
            "status": "cancelled",
            "message": f"Step {step_number} cancelled ({reason}).",
        }

    if step_state_delta:
        graph.log_state_delta(step_id, step_state_delta)

    step_result_data = step_state_delta.get("_step_result")
    if step_result_data:
        tool_context.state["_step_result"] = None  # State has no pop(); reset instead
        result = StepExecutorResult.model_validate(step_result_data)
        graph.log_node_complete(
            step_id,
            result.status,
            summary=result.concise_summary,
            artifacts=result.artifacts,
        )
        payload = result.model_dump(exclude_none=True)
        if plot_paths:
            payload["plot_paths"] = plot_paths
            payload["plot_path"] = plot_paths[0]
        clear_step_cancellation(session_id, step_number)
        return payload

    # submit_step_result was never called — surface as replanning signal
    logger.warning("[step_executor_runner] step %d: submit_step_result was never called", step_number)
    result = StepExecutorResult(
        status="needs_replanning",
        replan_reason="step executor did not call submit_step_result — no result captured",
    )
    graph.log_node_complete(step_id, "needs_replanning")
    clear_step_cancellation(session_id, step_number)
    return result.model_dump()


MAX_RECURSION_DEPTH = 3


async def _run_single_sub_step(
    sub_step_index: int,
    action: str,
    suggested_skills: list[str],
    prior_context: Optional[str],
    parent_workspace: str,
    parent_step_id: str,
    tool_context: ToolContext,
    recursion_depth: int,
    graph: AgentGraphLogger,
) -> dict:
    session_id = tool_context.state.get("session_id", "default")

    sub_workspace = Path(parent_workspace) / f"sub_{sub_step_index}"
    sub_workspace.mkdir(parents=True, exist_ok=True)

    step_id = f"{parent_step_id}__sub_{sub_step_index}"
    graph.log_node_start(step_id, "step", f"Sub-step {sub_step_index}", parent_step_id)

    step_input = StepExecutorInput(
        step_number=sub_step_index,
        action=action,
        suggested_skills=suggested_skills,
        workspace_dir=str(sub_workspace),
        prior_context=prior_context,
    )
    content = types.Content(
        role="user",
        parts=[types.Part.from_text(text=step_input.model_dump_json(exclude_none=True))],
    )

    graph.log_node_input(step_id, {
        "sub_step_index": sub_step_index,
        "action": action,
        "workspace_dir": str(sub_workspace),
        "suggested_skills": suggested_skills,
    })

    if is_cancellation_requested(session_id):
        reason = get_cancellation_reason(session_id) or "user_requested"
        graph.log_node_complete(step_id, "failed", summary=f"Cancelled before start: {reason}")
        return {"status": "cancelled", "message": f"Sub-step {sub_step_index} cancelled before start."}

    invocation_context = tool_context._invocation_context
    child_app_name = (
        invocation_context.app_name if invocation_context else step_executor_agent.name
    )
    runner = Runner(
        app_name=child_app_name,
        agent=step_executor_agent,
        artifact_service=ForwardingArtifactService(tool_context),
        session_service=InMemorySessionService(),
        memory_service=InMemoryMemoryService(),
        credential_service=invocation_context.credential_service,
    )

    state_dict = {
        k: v
        for k, v in tool_context.state.to_dict().items()
        if not k.startswith("_adk")
    }
    state_dict["workspace_dir"] = str(sub_workspace)
    state_dict["recursion_depth"] = recursion_depth + 1
    state_dict["_graph_exec_node_id"] = step_id

    session = await runner.session_service.create_session(
        app_name=child_app_name,
        user_id=invocation_context.user_id,
        state=state_dict,
    )

    inner_task = asyncio.create_task(
        _stream_step_events(runner, session, content, step_id, tool_context, graph)
    )
    watcher = asyncio.create_task(
        _watch_for_cancellation(inner_task, session_id, sub_step_index)
    )

    cancelled = False
    step_state_delta: dict = {}
    plot_paths: list[str] = []
    try:
        step_state_delta, plot_paths = await inner_task
    except asyncio.CancelledError:
        cancelled = True
    finally:
        if not watcher.done():
            watcher.cancel()
        await asyncio.gather(watcher, return_exceptions=True)
        try:
            await asyncio.wait_for(runner.close(), timeout=5.0)
        except Exception:
            pass

    if cancelled:
        reason = get_cancellation_reason(session_id) or "user_requested"
        graph.log_node_complete(step_id, "failed", summary=f"Cancelled mid-step ({reason})")
        return {"status": "cancelled", "message": f"Sub-step {sub_step_index} cancelled ({reason})."}

    if step_state_delta:
        graph.log_state_delta(step_id, step_state_delta)

    step_result_data = step_state_delta.get("_step_result")
    if step_result_data:
        result = StepExecutorResult.model_validate(step_result_data)
        graph.log_node_complete(
            step_id, result.status,
            summary=result.concise_summary,
            artifacts=result.artifacts,
        )
        payload = result.model_dump(exclude_none=True)
        if plot_paths:
            payload["plot_paths"] = plot_paths
            payload["plot_path"] = plot_paths[0]
        return payload

    logger.warning("[step_executor_runner] sub-step %d: submit_step_result was never called", sub_step_index)
    result = StepExecutorResult(
        status="needs_replanning",
        replan_reason="sub-step executor did not call submit_step_result — no result captured",
    )
    graph.log_node_complete(step_id, "needs_replanning")
    return result.model_dump()


async def _run_sub_steps_impl(
    sub_steps: list[dict],
    *,
    tool_context: ToolContext,
) -> dict:
    """Spawn child step executors for sub-steps. Called from run_sub_steps in step_executor.py."""
    recursion_depth = tool_context.state.get("recursion_depth", 0)
    if recursion_depth >= MAX_RECURSION_DEPTH:
        return {
            "status": "error",
            "message": (
                f"Maximum recursion depth ({MAX_RECURSION_DEPTH}) reached. "
                "Execute this action directly or call submit_step_result with needs_replanning."
            ),
        }

    if not sub_steps:
        return {"status": "error", "message": "sub_steps list is empty."}

    session_id = tool_context.state.get("session_id", "default")
    parent_workspace = tool_context.state.get("workspace_dir", "")
    parent_step_id = tool_context.state.get("_graph_exec_node_id", "step_executor")
    graph = AgentGraphLogger(session_id)

    tasks = [
        _run_single_sub_step(
            sub_step_index=i + 1,
            action=sub.get("action", ""),
            suggested_skills=sub.get("suggested_skills", []),
            prior_context=sub.get("prior_context"),
            parent_workspace=parent_workspace,
            parent_step_id=parent_step_id,
            tool_context=tool_context,
            recursion_depth=recursion_depth,
            graph=graph,
        )
        for i, sub in enumerate(sub_steps)
    ]

    raw_results = await asyncio.gather(*tasks, return_exceptions=True)

    sub_results = []
    for i, result in enumerate(raw_results):
        if isinstance(result, Exception):
            sub_results.append({
                "sub_step_index": i + 1,
                "status": "needs_replanning",
                "replan_reason": f"Sub-step raised an exception: {result}",
            })
        else:
            sub_results.append({"sub_step_index": i + 1, **result})

    return {"sub_results": sub_results}
