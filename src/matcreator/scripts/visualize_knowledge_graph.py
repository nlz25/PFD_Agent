#!/usr/bin/env python3
"""Visualize the MatCreator knowledge graph as an interactive HTML page.

Usage:
    python script/visualize_knowledge_graph.py
    python script/visualize_knowledge_graph.py --output /tmp/kg.html
    python script/visualize_knowledge_graph.py --default-only   # skills/guides only
    python script/visualize_knowledge_graph.py --no-open        # write file, don't open browser

The script reads knowledge_graph.db directly from agents/MatCreator/.adk/
and generates a self-contained HTML file using vis.js (loaded from CDN).
No extra Python dependencies required beyond sqlalchemy.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import webbrowser
from pathlib import Path

# ---------------------------------------------------------------------------
# Locate the DB
# ---------------------------------------------------------------------------

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
_ADK_DIR = _PROJECT_ROOT / "agents" / "MatCreator" / ".adk"
_SKILL_DB_PATH   = _ADK_DIR / "skill_graph.db"
_MEMORY_DB_PATH  = _ADK_DIR / "memory_graph.db"


def _load_single_db(db_path: Path) -> tuple[list[dict], list[dict]]:
    """Load nodes and edges from a single SQLite DB file."""
    if not db_path.exists():
        return [], []

    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row

    existing_cols = {row[1] for row in con.execute("PRAGMA table_info(kg_nodes)")}
    immutable_expr = "immutable" if "immutable" in existing_cols else "0"

    nodes = [dict(r) for r in con.execute(
        f"SELECT id, COALESCE(category, type) as category, name, description, "
        f"reference_count, confidence, created_at, source_session, "
        f"{immutable_expr} as immutable "
        "FROM kg_nodes ORDER BY category, name"
    ).fetchall()]

    edges = [dict(r) for r in con.execute(
        "SELECT id, source_id, target_id, edge_type, weight FROM kg_edges"
    ).fetchall()]

    con.close()
    return nodes, edges


def load_graph(
    skill_db: Path,
    memory_db: Path,
    graph: str = "all",
    categories: list[str] | None = None,
    default_only: bool = False,
) -> tuple[list[dict], list[dict]]:
    """Return (nodes, edges) merged from the requested graph(s).

    *graph*: ``"skills"``, ``"memory"``, or ``"all"`` (default).
    *categories*: optional category filter applied after graph selection.
    *default_only*: if True, keep only immutable (dev-seeded) nodes.
    """
    if graph == "skills":
        nodes, edges = _load_single_db(skill_db)
    elif graph == "memory":
        nodes, edges = _load_single_db(memory_db)
    else:  # "all"
        s_nodes, s_edges = _load_single_db(skill_db)
        m_nodes, m_edges = _load_single_db(memory_db)
        nodes = s_nodes + m_nodes
        edges = s_edges + m_edges

    if not nodes:
        return [], []

    if default_only:
        nodes = [n for n in nodes if n.get("immutable")]
    elif categories:
        allowed = {cat.lower() for cat in categories}
        nodes = [n for n in nodes if (n.get("category") or "").lower() in allowed]

    node_ids = {n["id"] for n in nodes}
    edges = [e for e in edges if e["source_id"] in node_ids and e["target_id"] in node_ids]

    return nodes, edges


# ---------------------------------------------------------------------------
# Color palette per node type
# ---------------------------------------------------------------------------

_COLORS = {
    "skill":  {"background": "#A9DFBF", "border": "#1E8449"},
    "memory": {"background": "#D7BDE2", "border": "#7D3C98"},
    # Legacy type names (for migrated nodes not yet re-categorized)
    "Concept":  {"background": "#AED6F1", "border": "#2E86C1"},
    "Skill":    {"background": "#A9DFBF", "border": "#1E8449"},
    "Material": {"background": "#F9E79F", "border": "#D4AC0D"},
    "Result":   {"background": "#F5CBA7", "border": "#CA6F1E"},
    "Insight":  {"background": "#D7BDE2", "border": "#7D3C98"},
    "Workflow": {"background": "#F1948A", "border": "#C0392B"},
}
_DEFAULT_COLOR = {"background": "#D5DBDB", "border": "#717D7E"}


def _node_label(n: dict) -> str:
    """Short label: name truncated to 30 chars."""
    return n["name"][:30] + ("…" if len(n["name"]) > 30 else "")


def _node_title(n: dict) -> str:
    """Hover tooltip: full name + description + stats."""
    cat = n.get("category", "")
    lines = [
        f"<b>{n['name']}</b>",
        f"Category: {cat}",
    ]
    if n.get("immutable"):
        lines.append("<i>⬡ default (immutable)</i>")
    if n.get("description"):
        lines.append(f"<br>{n['description']}")
    lines.append(f"<br>References: {n['reference_count']}  |  Confidence: {n['confidence']:.2f}")
    if n.get("created_at"):
        lines.append(f"Created: {n['created_at'][:10]}")
    return "<br>".join(lines)


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>MatCreator Knowledge Graph</title>
<script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  html, body { height: 100%; font-family: sans-serif; background: #f5f5f5; }
  body { display: flex; flex-direction: column; }
  #header { padding: 10px 20px; background: #2c3e50; color: white;
            display: flex; align-items: center; gap: 20px; flex-shrink: 0; }
  #header h1 { font-size: 1.1rem; white-space: nowrap; }
  #stats { font-size: 0.85rem; opacity: 0.8; white-space: nowrap; }
  #legend { display: flex; gap: 10px; flex-wrap: wrap; font-size: 0.8rem; }
  .legend-item { display: flex; align-items: center; gap: 5px; }
  .legend-dot { width: 12px; height: 12px; border-radius: 50%;
                border: 2px solid #555; flex-shrink: 0; }
  #controls { padding: 8px 20px; background: #ecf0f1; border-bottom: 1px solid #ddd;
              display: flex; gap: 12px; align-items: center; flex-wrap: wrap;
              flex-shrink: 0; }
  #controls label { font-size: 0.85rem; }
  #controls select, #controls input { font-size: 0.85rem; padding: 3px 6px;
                                      border: 1px solid #bbb; border-radius: 4px; }
  #network { flex: 1; background: white; min-height: 0; }
  #info-panel { position: fixed; right: 0; top: 0; bottom: 0; width: 280px;
                background: white; border-left: 1px solid #ddd;
                padding: 14px; font-size: 0.82rem; display: none;
                overflow-y: auto; }
  #info-panel h3 { margin-bottom: 8px; font-size: 0.95rem; }
  #info-panel .field { margin-bottom: 6px; }
  #info-panel .label { color: #777; font-size: 0.75rem; text-transform: uppercase; }
</style>
</head>
<body>
<div id="header">
  <h1>MatCreator Knowledge Graph</h1>
  <div id="stats">STATS_PLACEHOLDER</div>
  <div id="legend">LEGEND_PLACEHOLDER</div>
</div>
<div id="controls">
  <label>Filter type:
    <select id="type-filter" onchange="applyFilter()">
      <option value="">All</option>
      TYPE_OPTIONS
    </select>
  </label>
  <label>Search name:
    <input id="search" type="text" placeholder="type to highlight…" oninput="applySearch()">
  </label>
  <label><input type="checkbox" id="show-labels" checked onchange="toggleLabels()"> Edge labels</label>
  <button onclick="network.fit()">Reset view</button>
</div>
<div id="network"></div>
<div id="info-panel">
  <h3 id="info-title">Node details</h3>
  <div id="info-body"></div>
</div>

<script>
const RAW_NODES = NODES_JSON;
const RAW_EDGES = EDGES_JSON;

// Build vis datasets
const nodeMap = {};
RAW_NODES.forEach(n => { nodeMap[n.id] = n; });

function makeVisNode(n, hidden) {
  const color = NODE_COLORS[n.category] || NODE_COLORS.__default__;
  return {
    id: n.id,
    label: n.label,
    title: n.title,
    color: color,
    font: { size: 13 },
    borderWidth: n.immutable ? 3 : 1,
    hidden: hidden || false,
  };
}

const visNodes = new vis.DataSet(RAW_NODES.map(n => makeVisNode(n, false)));
const visEdges = new vis.DataSet(RAW_EDGES.map(e => ({
  id: e.id,
  from: e.source_id,
  to: e.target_id,
  label: e.edge_type,
  title: `${e.edge_type}  (weight: ${e.weight.toFixed(1)})`,
  arrows: "to",
  font: { size: 10, color: "#888", align: "middle" },
  color: { color: "#aaa", highlight: "#555" },
  width: Math.min(1 + e.weight * 0.4, 4),
})));

const container = document.getElementById("network");
const network = new vis.Network(container, { nodes: visNodes, edges: visEdges }, {
  physics: {
    barnesHut: { gravitationalConstant: -8000, centralGravity: 0.3, springLength: 140 },
    stabilization: { iterations: 200 },
  },
  interaction: { hover: true, tooltipDelay: 150 },
  layout: { improvedLayout: true },
});

// Info panel on click
network.on("click", params => {
  const panel = document.getElementById("info-panel");
  const body  = document.getElementById("info-body");
  if (params.nodes.length === 0) { panel.style.display = "none"; return; }
  const n = nodeMap[params.nodes[0]];
  if (!n) return;
  panel.style.display = "block";
  document.getElementById("info-title").textContent = n.name;
  body.innerHTML = `
    <div class="field"><div class="label">Category</div>${n.category}</div>
    <div class="field"><div class="label">Default node</div>${n.immutable ? "Yes (immutable)" : "No (user-generated)"}</div>
    <div class="field"><div class="label">Description</div>${n.description || "—"}</div>
    <div class="field"><div class="label">References</div>${n.reference_count}</div>
    <div class="field"><div class="label">Confidence</div>${n.confidence.toFixed(2)}</div>
    <div class="field"><div class="label">Session</div>${n.source_session || "—"}</div>
    <div class="field"><div class="label">Created</div>${(n.created_at||"").slice(0,19)}</div>
  `;
});

function applyFilter() {
  const val = document.getElementById("type-filter").value;
  visNodes.forEach(vn => {
    const n = nodeMap[vn.id];
    visNodes.update({ id: vn.id, hidden: val && n.category !== val });
  });
}

function applySearch() {
  const q = document.getElementById("search").value.toLowerCase();
  visNodes.forEach(vn => {
    const n = nodeMap[vn.id];
    const hit = !q || n.name.toLowerCase().includes(q);
    const color = NODE_COLORS[n.category] || NODE_COLORS.__default__;
    visNodes.update({
      id: vn.id,
      color: hit ? color : { background: "#eee", border: "#ccc" },
      font: { size: hit ? 13 : 11, color: hit ? "#000" : "#bbb" },
    });
  });
}

function toggleLabels() {
  const show = document.getElementById("show-labels").checked;
  visEdges.forEach(e => {
    visEdges.update({ id: e.id, font: { size: show ? 10 : 0, color: "#888" } });
  });
}
</script>
</body>
</html>
"""


def build_html(nodes: list[dict], edges: list[dict]) -> str:
    # Stats line
    cat_counts: dict[str, int] = {}
    for n in nodes:
        cat = n.get("category", "memory")
        cat_counts[cat] = cat_counts.get(cat, 0) + 1
    stats = f"{len(nodes)} nodes · {len(edges)} edges  |  " + \
            "  ".join(f"{c}: {cnt}" for c, cnt in sorted(cat_counts.items()))

    # Legend HTML (only show the two main categories)
    _LEGEND_COLORS = {k: v for k, v in _COLORS.items() if k in ("skill", "memory")}
    legend_items = "".join(
        f'<div class="legend-item">'
        f'<div class="legend-dot" style="background:{col["background"]};border-color:{col["border"]}"></div>'
        f'{t}</div>'
        for t, col in _LEGEND_COLORS.items()
    )

    # Category filter options
    type_options = "\n".join(
        f'<option value="{cat}">{cat} ({cat_counts.get(cat, 0)})</option>'
        for cat in sorted(cat_counts.keys())
    )

    # Node JS objects (include raw fields for info panel)
    vis_nodes = []
    for n in nodes:
        cat = n.get("category", "memory")
        vis_nodes.append({
            "id": n["id"],
            "name": n["name"],
            "category": cat,
            "description": n.get("description") or "",
            "reference_count": n.get("reference_count", 0),
            "confidence": n.get("confidence", 1.0),
            "source_session": n.get("source_session") or "",
            "created_at": n.get("created_at") or "",
            "immutable": bool(n.get("immutable")),
            "label": _node_label(n),
            "title": _node_title(n),
        })

    # Edge JS objects
    vis_edges = []
    for e in edges:
        vis_edges.append({
            "id": e["id"],
            "source_id": e["source_id"],
            "target_id": e["target_id"],
            "edge_type": e["edge_type"],
            "weight": e.get("weight", 1.0),
        })

    # Color map for JS
    colors_js = {t: c for t, c in _COLORS.items()}
    colors_js["__default__"] = _DEFAULT_COLOR

    html = _HTML_TEMPLATE
    html = html.replace("STATS_PLACEHOLDER", stats)
    html = html.replace("LEGEND_PLACEHOLDER", legend_items)
    html = html.replace("TYPE_OPTIONS", type_options)
    html = html.replace("NODES_JSON", json.dumps(vis_nodes, ensure_ascii=False))
    html = html.replace("EDGES_JSON", json.dumps(vis_edges, ensure_ascii=False))
    html = html.replace("NODE_COLORS", json.dumps(colors_js))
    return html


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize the MatCreator knowledge graph.")
    parser.add_argument("--db-skills", default=str(_SKILL_DB_PATH),
                        help=f"Path to skill_graph.db (default: {_SKILL_DB_PATH})")
    parser.add_argument("--db-memory", default=str(_MEMORY_DB_PATH),
                        help=f"Path to memory_graph.db (default: {_MEMORY_DB_PATH})")
    parser.add_argument("--graph",
                        choices=["skills", "memory", "all"], default="all",
                        help="Which graph(s) to render: 'skills', 'memory', or 'all' (default)")
    parser.add_argument("--output", default=str(_PROJECT_ROOT / "knowledge_graph.html"),
                        help="Output HTML file path")
    parser.add_argument("--category", metavar="CAT", action="append", dest="categories",
                        help="Only include nodes of this category (repeatable, e.g. --category skill)")
    parser.add_argument("--default-only", action="store_true",
                        help="Show only default (immutable) nodes from SKILL.md and guides")
    parser.add_argument("--no-open", action="store_true",
                        help="Write the file but do not open it in a browser")
    args = parser.parse_args()

    nodes, edges = load_graph(
        skill_db=Path(args.db_skills),
        memory_db=Path(args.db_memory),
        graph=args.graph,
        categories=args.categories,
        default_only=args.default_only,
    )
    if not nodes:
        print("No nodes found — run `matcreator knowledge seed` first.")
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
