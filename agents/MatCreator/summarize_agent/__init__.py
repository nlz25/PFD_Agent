"""Summarize-agent sub-package.

Exports ``summarize_agent``, an ``LlmAgent`` that synthesises execution
outcomes into a structured ``ExecutionSummary`` (key results, artifact paths,
user-facing paragraph) after each execution cycle.
"""

from .agent import summarize_agent, ExecutionSummary, ExecutionSummaryInput

__all__ = ["summarize_agent", "ExecutionSummary", "ExecutionSummaryInput"]
