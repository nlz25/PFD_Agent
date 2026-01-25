"""Sub-agent registry and formatting helpers for planning/execution prompts."""

from __future__ import annotations

from typing import Dict, List

from ..database_agent.agent import database_agent
from ..abacus_agent.agent import abacus_agent
from ..dpa_agent.agent import dpa_agent
from ..vasp_agent.agent import vasp_agent
from ..structure_agent.agent import structure_agent

# Master registry of sub-agents available to the MatCreator workflow
SUBAGENTS: Dict[str, object] = {
    "database_agent": database_agent,
    "structure_agent": structure_agent,
    "abacus_agent": abacus_agent,
    "vasp_agent": vasp_agent,
    "dpa_agent": dpa_agent,
}

def _get_subagent_descriptions() -> Dict[str, str]:
    """Get descriptions of all sub-agents."""
    return {
        name: agent.description if hasattr(agent, "description") and agent.description
        else f"{name} - description not available"
        for name, agent in SUBAGENTS.items()
    }


def format_subagent_descriptions() -> str:
    """Formatted list of all sub-agents for injection into prompts."""
    descriptions = _get_subagent_descriptions()
    lines = []
    for i, (name, desc) in enumerate(descriptions.items(), 1):
        lines.append(f"{i}. **{name}**: {desc}")
    return "\n".join(lines)


def format_subagent_descriptions_for_agents(allowed_agents: List[str]) -> str:
    """Formatted sub-agent list scoped to an explicit allowlist of agent names."""
    descriptions = {
        name: agent.description if hasattr(agent, "description") and agent.description
        else f"{name} - description not available"
        for name, agent in SUBAGENTS.items() if name in allowed_agents
    }
    lines = []
    for i, (name, desc) in enumerate(descriptions.items(), 1):
        lines.append(f"{i}. **{name}**: {desc}")
    return "\n".join(lines)


__all__ = [
    "SUBAGENTS",
    "get_subagent_descriptions",
    "format_subagent_descriptions",
    "format_subagent_descriptions_for_agents",
]
