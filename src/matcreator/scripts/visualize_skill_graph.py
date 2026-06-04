#!/usr/bin/env python3
"""Visualize only the default (dev-maintained) skill graph as an interactive HTML page.

This is a focused wrapper around visualize_knowledge_graph.py that always
renders skill_graph.db only — no user memory nodes.

Usage:
    python script/visualize_skill_graph.py
    python script/visualize_skill_graph.py --output /tmp/skills.html
    python script/visualize_skill_graph.py --no-open
"""

from __future__ import annotations

import argparse
import sys
import webbrowser
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))

from visualize_knowledge_graph import (  # noqa: E402
    _SKILL_DB_PATH,
    _PROJECT_ROOT,
    load_graph,
    build_html,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize the MatCreator skill graph (default nodes only)."
    )
    parser.add_argument("--db-skills", default=str(_SKILL_DB_PATH),
                        help=f"Path to skill_graph.db (default: {_SKILL_DB_PATH})")
    parser.add_argument("--output", default=str(_PROJECT_ROOT / "knowledge_graph_skills.html"),
                        help="Output HTML file path")
    parser.add_argument("--no-open", action="store_true",
                        help="Write the file but do not open it in a browser")
    args = parser.parse_args()

    nodes, edges = load_graph(
        skill_db=Path(args.db_skills),
        memory_db=Path("/dev/null"),  # ignored — skills only
        graph="skills",
    )
    if not nodes:
        print("Skill graph is empty — run `matcreator knowledge seed` first.")
        return

    out_path = Path(args.output)
    html = build_html(nodes, edges)
    out_path.write_text(html, encoding="utf-8")
    print(f"Written: {out_path}  ({len(nodes)} nodes, {len(edges)} edges)")

    if not args.no_open:
        import platform, os
        is_wsl = "microsoft" in platform.uname().release.lower() or \
                 os.path.exists("/proc/sys/fs/binfmt_misc/WSLInterop")
        if is_wsl:
            print("WSL detected — open the file in VS Code or via Live Server:")
            print(f"  {out_path}")
        else:
            webbrowser.open(out_path.as_uri())


if __name__ == "__main__":
    main()
