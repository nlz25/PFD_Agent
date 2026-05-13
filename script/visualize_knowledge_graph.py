#!/usr/bin/env python3
"""Visualize the MatCreator knowledge graph as an interactive HTML page.

Usage:
    python script/visualize_knowledge_graph.py
    python script/visualize_knowledge_graph.py --output /tmp/kg.html
    python script/visualize_knowledge_graph.py --no-open   # write file, don't open browser

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
_DB_PATH = _PROJECT_ROOT / "agents" / "MatCreator" / ".adk" / "knowledge_graph.db"


def load_graph(db_path: Path) -> tuple[list[dict], list[dict]]:
    """Return (nodes, edges) as plain dicts from the SQLite DB."""
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row

    nodes = [dict(r) for r in con.execute(
        "SELECT id, type, name, description, reference_count, confidence, created_at "
        "FROM kg_nodes ORDER BY type, name"
    ).fetchall()]

    edges = [dict(r) for r in con.execute(
        "SELECT id, source_id, target_id, edge_type, weight FROM kg_edges"
    ).fetchall()]

    con.close()
    return nodes, edges


# ---------------------------------------------------------------------------
# Color palette per node type
# ---------------------------------------------------------------------------

_COLORS = {
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
    lines = [
        f"<b>{n['name']}</b>",
        f"Type: {n['type']}",
    ]
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
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: sans-serif; background: #f5f5f5; }}
  #header {{ padding: 12px 20px; background: #2c3e50; color: white;
             display: flex; align-items: center; gap: 20px; }}
  #header h1 {{ font-size: 1.1rem; }}
  #stats {{ font-size: 0.85rem; opacity: 0.8; }}
  #legend {{ display: flex; gap: 10px; flex-wrap: wrap; font-size: 0.8rem; }}
  .legend-item {{ display: flex; align-items: center; gap: 5px; }}
  .legend-dot {{ width: 12px; height: 12px; border-radius: 50%;
                 border: 2px solid #555; }}
  #controls {{ padding: 8px 20px; background: #ecf0f1; border-bottom: 1px solid #ddd;
               display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }}
  #controls label {{ font-size: 0.85rem; }}
  #controls select, #controls input {{ font-size: 0.85rem; padding: 3px 6px;
                                       border: 1px solid #bbb; border-radius: 4px; }}
  #network {{ width: 100%; height: calc(100vh - 110px); background: white; }}
  #info-panel {{ position: fixed; right: 0; top: 110px; width: 280px;
                 background: white; border-left: 1px solid #ddd;
                 padding: 14px; font-size: 0.82rem; display: none;
                 overflow-y: auto; max-height: calc(100vh - 110px); }}
  #info-panel h3 {{ margin-bottom: 8px; font-size: 0.95rem; }}
  #info-panel .field {{ margin-bottom: 6px; }}
  #info-panel .label {{ color: #777; font-size: 0.75rem; text-transform: uppercase; }}
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
<div id="info-panel" id="info">
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
  const color = NODE_COLORS[n.type] || NODE_COLORS.__default__;
  return {
    id: n.id,
    label: n.label,
    title: n.title,
    color: color,
    font: { size: 13 },
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
    <div class="field"><div class="label">Type</div>${n.type}</div>
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
    visNodes.update({ id: vn.id, hidden: val && n.type !== val });
  });
}

function applySearch() {
  const q = document.getElementById("search").value.toLowerCase();
  visNodes.forEach(vn => {
    const n = nodeMap[vn.id];
    const hit = !q || n.name.toLowerCase().includes(q);
    const color = NODE_COLORS[n.type] || NODE_COLORS.__default__;
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
    type_counts: dict[str, int] = {}
    for n in nodes:
        type_counts[n["type"]] = type_counts.get(n["type"], 0) + 1
    stats = f"{len(nodes)} nodes · {len(edges)} edges  |  " + \
            "  ".join(f"{t}: {c}" for t, c in sorted(type_counts.items()))

    # Legend HTML
    legend_items = "".join(
        f'<div class="legend-item">'
        f'<div class="legend-dot" style="background:{col["background"]};border-color:{col["border"]}"></div>'
        f'{t}</div>'
        for t, col in _COLORS.items()
    )

    # Type filter options
    type_options = "\n".join(
        f'<option value="{t}">{t} ({type_counts.get(t, 0)})</option>'
        for t in sorted(_COLORS.keys())
    )

    # Node JS objects (include raw fields for info panel)
    vis_nodes = []
    for n in nodes:
        vis_nodes.append({
            "id": n["id"],
            "name": n["name"],
            "type": n["type"],
            "description": n.get("description") or "",
            "reference_count": n.get("reference_count", 0),
            "confidence": n.get("confidence", 1.0),
            "source_session": n.get("source_session") or "",
            "created_at": n.get("created_at") or "",
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
    parser.add_argument("--db", default=str(_DB_PATH),
                        help=f"Path to knowledge_graph.db (default: {_DB_PATH})")
    parser.add_argument("--output", default=str(_PROJECT_ROOT / "knowledge_graph.html"),
                        help="Output HTML file path")
    parser.add_argument("--no-open", action="store_true",
                        help="Write the file but do not open it in a browser")
    args = parser.parse_args()

    db_path = Path(args.db)
    out_path = Path(args.output)

    nodes, edges = load_graph(db_path)
    if not nodes:
        print("Knowledge graph is empty — run some sessions first.")
        return

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
