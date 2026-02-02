"""Utility agent for MatCreator - handles plotting, data extraction, and file operations."""

from __future__ import annotations

import os
import logging
from google.adk.agents import LlmAgent
from google.adk.tools.agent_tool import AgentTool
from google.adk.models.lite_llm import LiteLlm

from .plot_agent.agent import root_agent as plot_agent
from .constants import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL
from .callbacks import after_tool_callback

_model_name = os.environ.get("LLM_MODEL", LLM_MODEL)
_model_api_key = os.environ.get("LLM_API_KEY", LLM_API_KEY)
_model_base_url = os.environ.get("LLM_BASE_URL", LLM_BASE_URL)

logger = logging.getLogger(__name__)

_UTIL_INSTRUCTION = """
You are the utility agent for MatCreator workflows. Your primary role is to handle scientific visualization
tasks that support material science workflows.

**Your capability:**

**Scientific Plotting** (via plot_agent)
- Generate publication-quality matplotlib plots from computational data
- Energy convergence plots, force distributions, training loss curves
- Structure property visualizations (volume, density, lattice parameters vs. time/step)
- Comparison plots (predicted vs. labeled energies/forces)
- Multi-panel figures for comprehensive analysis

**How to operate:**

- For ALL plotting requests: **delegate to plot_agent** immediately
- Pass the user's data file path and plotting requirements to plot_agent
- Return the generated plot path to the user

**Important:**
- You do not generate plots yourself - always use plot_agent
- Ensure file paths are absolute and accessible
- Verify plot requirements are clear before delegating

**Examples:**

User: "Plot energy convergence from training log"
→ Use plot_agent with the log file path and "energy vs step" plot request

User: "Create a scatter plot comparing predicted and labeled forces"
→ Use plot_agent with data file and "scatter plot: predicted vs labeled forces"

User: "Visualize volume changes during MD simulation"
→ Use plot_agent with trajectory file and "volume vs time" plot request
"""

util_agent = LlmAgent(
    name="util_agent",
    model=LiteLlm(
        model=_model_name,
        base_url=_model_base_url,
        api_key=_model_api_key,
    ),
    description=(
        "Generates scientific plots and visualizations for computational materials science workflows. "
    ),
    instruction=_UTIL_INSTRUCTION,
    after_tool_callback=after_tool_callback,
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
    tools=[AgentTool(plot_agent)],
)

__all__ = ["util_agent"]
