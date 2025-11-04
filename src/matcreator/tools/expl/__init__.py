"""Exploration MCP server bootstrap.

Import the exploration tool modules (ase.py and atoms.py) so their
@mcp.tool-decorated functions register with the shared MCP instance,
then provide a simple entrypoint to run the server.
"""
from __future__ import annotations

from matcreator.tools.expl import ase as expl_tools_ase  # noqa: F401
from matcreator.tools.expl import atoms as expl_tools_atoms  # noqa: F401