"""Sub-agent package for MatCreator.

Exposes the domain-specific sub-agents (``plot_agent``, ``sql_agent``) that
are registered as tools inside the execution agent.
"""

from .plot_agent.agent import plot_agent
from .sql_agent.agent import sql_agent