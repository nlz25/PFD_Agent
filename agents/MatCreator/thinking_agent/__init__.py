"""Thinking-agent sub-package.

Exports ``thinking_agent``, the planning-phase orchestrator responsible for
goal classification, execution-plan drafting, and the user-approval gate
before handing control to the execution agent.
"""

from .agent import thinking_agent

__all__ = ["thinking_agent"]
