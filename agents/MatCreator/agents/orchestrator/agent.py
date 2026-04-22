"""Planning-Execution Orchestrator for MatCreator.

Every invocation runs a planning-first loop:

  1. Planning phase  — always runs first. planning_agent (thinking_agent) handles
                       user intent and sets one of two routing flags:
                         • state["execution_approved"] = True  → execution phase
                         • state["testing_requested"]  = True  → testing phase
                       If neither flag is set the turn was conversational; the loop
                       exits and control returns to the user.

  2. Execution phase — loops over plan steps, calling execution_agent for each one.
                       Terminates early if execution_agent calls `to_planner`
                       (sets state["return_to_planner"] = True).
                       After all steps complete (or on early exit), loops back to
                       the planning phase within the same invocation.

  3. Testing phase   — delegates to tester_agent for skill creation/validation, then
                       loops back to the planning phase.
"""

from __future__ import annotations

import json
import logging
from typing import AsyncGenerator, List, Optional

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from pydantic import Field

from ...workspace import init_session_workdir

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_steps(plan) -> List[dict]:
    """Return the list of step dicts from a plan value (dict or JSON string)."""
    if not plan:
        return []
    if isinstance(plan, str):
        try:
            plan = json.loads(plan)
        except json.JSONDecodeError:
            return []
    if isinstance(plan, dict):
        return plan.get("steps", [])
    return []


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class PlanningExecutionOrchestrator(BaseAgent):
    """Orchestrates the planning → execution loop.

    Attributes:
        planning_agent: Handles user intent, plan creation, and approval signalling.
        execution_agent: Executes one plan step at a time (skill load + tool calls).
        tester_agent: Creates and validates skills on demand (optional).
    """

    planning_agent: BaseAgent
    execution_agent: BaseAgent
    tester_agent: Optional[BaseAgent] = Field(default=None)

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        state = ctx.session.state

        if not state.get("session_workdir_initialized"):
            init_session_workdir(ctx.session.id)
            state["session_id"] = ctx.session.id
            state["session_workdir_initialized"] = True

        while True:
            # ── Planning phase (always runs first) ───────────────────────────
            state["execution_approved"] = False
            logger.info("[orchestrator] entering planning phase")
            async for event in self.planning_agent.run_async(ctx):
                yield event

            testing_requested: bool = state.get("testing_requested", False)
            execution_approved: bool = state.get("execution_approved", False)

            # ── Testing phase ─────────────────────────────────────────────────
            if testing_requested and self.tester_agent is not None:
                logger.info("[orchestrator] entering testing phase")
                async for event in self.tester_agent.run_async(ctx):
                    yield event
                state["testing_requested"] = False
                continue  # loop back to planner

            # ── Execution phase ───────────────────────────────────────────────
            if execution_approved:
                plan = state.get("plan")
                steps = _extract_steps(plan)
                if not steps:
                    logger.warning("[orchestrator] execution approved but plan has no steps")
                    state["execution_approved"] = False
                    continue  # loop back to planner

                current_step_index: int = state.get("current_step_index", 0)
                total_steps = len(steps)
                logger.info(
                    "[orchestrator] executing %d/%d remaining steps",
                    total_steps - current_step_index,
                    total_steps,
                )

                interrupted = False
                while current_step_index < total_steps:
                    step = steps[current_step_index]
                    state["current_step"] = step
                    logger.info(
                        "[orchestrator] step %d/%d — skill=%s | action=%s",
                        current_step_index + 1,
                        total_steps,
                        step.get("skill", "?"),
                        step.get("action", "?"),
                    )
                    async for event in self.execution_agent.run_async(ctx):
                        yield event

                    if state.get("return_to_planner", False):
                        reason = state.get("return_to_planner_reason", "unspecified")
                        logger.info(
                            "[orchestrator] execution interrupted at step %d/%d — returning to planner (reason: %s)",
                            current_step_index + 1,
                            total_steps,
                            reason,
                        )
                        state["return_to_planner"] = False
                        state["return_to_planner_reason"] = None
                        interrupted = True
                        break

                    current_step_index += 1
                    state["current_step_index"] = current_step_index

                if not interrupted:
                    logger.info("[orchestrator] all %d steps complete", total_steps)

                # Reset execution flags; only reset step index after full completion.
                state["execution_approved"] = False
                state["current_step"] = None
                if not interrupted:
                    state["current_step_index"] = 0
                else:
                    # Preserve the index so a resume command can continue from here.
                    state["current_step_index"] = current_step_index
                continue  # loop back to planner

            # ── No flag set — planner handled the turn conversationally ───────
            # In benchmark mode, auto-approve if a valid plan exists
            if state.get("benchmark_mode", False):
                plan = state.get("plan")
                steps = _extract_steps(plan)
                if steps:
                    logger.info(
                        "[orchestrator] benchmark mode: auto-approving plan with %d steps",
                        len(steps),
                    )
                    state["execution_approved"] = True
                    state["current_step_index"] = 0
                    continue
                logger.warning(
                    "[orchestrator] benchmark mode: no valid plan to auto-approve, exiting"
                )
            break
