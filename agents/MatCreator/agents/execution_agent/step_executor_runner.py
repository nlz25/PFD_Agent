from __future__ import annotations

import logging
from contextlib import aclosing
from datetime import datetime, timezone
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

logger = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


async def run_step_executor(
    step_number: int,
    action: str,
    skill_name: str,
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
    step_id = f"step_{step_number}"
    parent_id = tool_context.state.get("_graph_exec_node_id", "orchestrator")
    graph.log_node_start(step_id, "step", f"Step {step_number}", parent_id)

    # Serialize input as user message (matches AgentTool input_schema path)
    step_input = StepExecutorInput(
        step_number=step_number,
        action=action,
        skill_name=skill_name,
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
        "skill_name": skill_name,
    })

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

    session = await runner.session_service.create_session(
        app_name=child_app_name,
        user_id=invocation_context.user_id,
        state=state_dict,
    )

    # Run sub-agent and forward state deltas to parent (mirrors AgentTool loop)
    step_state_delta: dict = {}
    pending_tool_calls: dict[str, dict] = {}  # tool name -> partial record

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

    await runner.close()

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
        return result.model_dump(exclude_none=True)

    # submit_step_result was never called — surface as replanning signal
    logger.warning("[step_executor_runner] step %d: submit_step_result was never called", step_number)
    result = StepExecutorResult(
        status="needs_replanning",
        replan_reason="step executor did not call submit_step_result — no result captured",
    )
    graph.log_node_complete(step_id, "needs_replanning")
    return result.model_dump()
