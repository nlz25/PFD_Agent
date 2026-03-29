"""Utility tools for the MatCreator execution agent."""

from __future__ import annotations

import os
from pathlib import Path


def show_plot(plot_path: str) -> dict:
    """Register a generated plot so the UI can display it.

    Call this tool immediately after a plot file has been saved to disk.
    It validates the file exists and returns a structured response that the
    Streamlit UI recognises as a plot result.

    Args:
        plot_path: Absolute (or MATCLAW_WORKSPACE-relative) path to the PNG file.

    Returns:
        dict with ``plot_path`` (resolved absolute path) on success, or an
        ``error`` key if the file cannot be found.
    """
    # Resolve relative paths against MATCLAW_WORKSPACE
    p = Path(plot_path)
    if not p.is_absolute():
        ws_root = os.environ.get("MATCLAW_WORKSPACE") or str(
            Path(__file__).parent / ".workspace"
        )
        p = Path(ws_root) / p

    p = p.resolve()

    if not p.exists():
        return {"error": f"Plot file not found: {p}"}

    if not p.is_file():
        return {"error": f"Path is not a file: {p}"}

    return {"plot_path": str(p)}
