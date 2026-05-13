# Knowledge Graph Memory System

## Motivation

The original memory system used a single flat `MEMORY.md` file that grew unboundedly through append-only writes. This caused three compounding problems:

1. **Context window overflow** — the entire file was injected into every planning prompt regardless of relevance.
2. **Knowledge staleness and duplication** — no deduplication, no pruning, and no expiry of outdated entries.
3. **No skill-knowledge linkage** — lessons learned were disconnected from the skills and workflows they applied to.

The knowledge graph replaces this with a structured, self-evolving property graph backed by SQLite, with targeted BFS-based retrieval replacing full-file injection.

---

## Storage and Persistence

The graph is stored in a SQLite database at:

```
agents/MatCreator/.adk/knowledge_graph.db
```

This places it alongside the ADK session database (`session.db`) in the `.adk` directory, which is anchored to the agent package path (`_AGENT_PATH / ".adk"`), not the user-facing workspace. The `.adk` directory is outside the workspace root and therefore not reachable by the `run_bash` or `run_python` tools available to the agent during task execution.

The database is created automatically on first use. It persists across server restarts. NetworkX is used as an in-memory traversal layer only — it is loaded from SQLite at query time and discarded afterward; it is not the source of truth.

Cross-session pipeline state (execution completion counter, last synthesizer run timestamp) is stored alongside the database at:

```
agents/MatCreator/.adk/.kg_state.json
```

---

## Data Model

### Node types

| Type | Meaning | Primary source |
|------|---------|---------------|
| `Concept` | Abstract domain knowledge (e.g. "VASP k-point convergence") | Extractor |
| `Skill` | Pointer to a `SKILL.md` procedural workflow | Extractor |
| `Material` | Material entity (e.g. "BaTiO3", "Si diamond") | Extractor / trajectory |
| `Result` | Quantitative finding (energy, accuracy, convergence value) | Extractor / trajectory |
| `Insight` | Heuristic or lesson valid across future sessions | Extractor / `save_to_knowledge_graph` / migration |
| `Workflow` | Multi-step procedure abstracted from repeated Insight clusters | Synthesizer |

### Edge types

| Type | Meaning |
|------|---------|
| `requires` | A Workflow or Skill depends on a Concept |
| `produces` | A Workflow or Skill yields a Result |
| `tested_on` | A Result was obtained for a specific Material |
| `specializes` | A Concept is a sub-type of a parent Concept |
| `similar_to` | Two nodes are near-duplicates (used by the Synthesizer before merging) |
| `discovered_in` | An Insight was learned while using a Skill or Workflow |
| `supersedes` | A newer Insight replaces an older one |

### SQLite schema

```sql
kg_nodes (
  id              TEXT PRIMARY KEY,   -- UUID
  type            TEXT NOT NULL,      -- node type enum
  name            TEXT NOT NULL,      -- canonical display name
  description     TEXT,
  content         JSON,               -- type-specific payload
  source_session  TEXT,               -- ADK session_id where extracted
  created_at      TEXT NOT NULL,
  updated_at      TEXT NOT NULL,
  reference_count INTEGER DEFAULT 0,  -- incremented on each retrieval
  confidence      REAL    DEFAULT 1.0 -- LLM extraction confidence 0–1
)

kg_edges (
  id         TEXT PRIMARY KEY,
  source_id  TEXT REFERENCES kg_nodes(id) ON DELETE CASCADE,
  target_id  TEXT REFERENCES kg_nodes(id) ON DELETE CASCADE,
  edge_type  TEXT NOT NULL,
  weight     REAL DEFAULT 1.0,        -- incremented when same edge re-extracted
  properties JSON,
  created_at TEXT NOT NULL
)
```

---

## System Components

### 1. `graph_store.py` — KnowledgeGraph

Core CRUD layer. Key behaviours:

**Fuzzy deduplication on upsert.** Before inserting a new node, `upsert_node` checks all existing nodes of the same type for name similarity using `difflib.SequenceMatcher`. If the ratio exceeds `similarity_threshold` (default 0.85), the existing node is returned and its description is updated if the new one is longer. This prevents the same concept from accumulating under slightly different names.

**Edge weight accumulation.** `upsert_edge` increments the `weight` field if the same (source, target, edge_type) triple already exists, rather than creating a duplicate edge. Frequently re-observed relationships therefore carry higher weight.

**NetworkX loader.** `load_networkx()` materialises the entire graph as a `nx.DiGraph` for in-memory BFS traversal. Called only at query time.

---

### 2. `query.py` — Retrieval

**`query_knowledge_graph(query, types, depth, top_k)`**

Called by the thinking agent at the start of planning. Returns structured Markdown grouped by node type. Algorithm:

1. Fuzzy name search via SQLite `LIKE '%query%'` → seed nodes.
2. Load full graph into NetworkX.
3. BFS from all seed nodes (bidirectional — follows both outgoing and incoming edges) up to `depth` hops.
4. Optionally filter by `types`.
5. Rank collected nodes by: `(1 + reference_count) × confidence × recency_decay`, where `recency_decay = 1 / (1 + days_old)`. Newer, more-referenced, high-confidence nodes rank first.
6. Return top `top_k` nodes. Increment `reference_count` for each returned node (hot-path signal for the synthesizer).

**`save_to_knowledge_graph(content, context)`**

Lets the thinking agent immediately persist an Insight during a session without waiting for the post-session extractor. The first 80 characters of `content` become the node name; deduplication still applies.

---

### 3. `extractor.py` — KnowledgeExtractor

Triggered automatically by the orchestrator after every successful execution phase. Reads two sources for a given `session_id`:

**Source A — per-step trajectory** (`trajectories/{session_id}.jsonl`): each line contains `step_index`, `active_skill`, `key_results`, `concise_summary`. This captures *what was done* and quantitative results.

**Source B — session-level summary** (`trajectories/{session_id}_summary.json`): written by the thinking agent via `write_session_summary` after execution returns to the planner. Schema:

```json
{
  "goal":            "Original user goal in their words",
  "approach":        "Overall approach and why it was chosen",
  "outcome":         "One-sentence statement of what was accomplished",
  "key_decisions":   ["decision 1", "decision 2"],
  "lessons_learned": ["heuristic 1", "heuristic 2"],
  "failed_attempts": ["tried X, failed because Y"]
}
```

This captures the *why*: planning rationale, key decisions, and failures that step-level entries miss.

Both sources are combined into a single LLM prompt. The LLM returns a JSON array of `{type, name, description, relations}` objects. The extractor then:

1. **Pass 1 — nodes**: upsert each entity (deduplication via `graph_store`).
2. **Pass 2 — edges**: upsert each declared relation (both endpoints must be present in the current batch).

Gracefully handles missing trajectory (skips) and missing session summary (falls back to a placeholder string).

---

### 4. `synthesizer.py` — KnowledgeSynthesizer

Runs every 10 completed executions (counted persistently across sessions in `.kg_state.json`). Executes three passes in order:

**Pass 1 — Prune stale nodes.** Deletes nodes where `reference_count ≤ stale_min_refs` (default 0) AND age ≥ `stale_days` (default 30 days). Implements a "use-it-or-lose-it" decay: knowledge that has never been retrieved and is older than a month is removed.

**Pass 2 — Merge similar_to clusters.** Finds all connected components in the subgraph of `similar_to` edges using union-find. Within each cluster, the node with the highest `reference_count` is designated canonical. All incoming and outgoing edges of non-canonical nodes are redirected to the canonical node, and the non-canonical nodes are deleted. Reference counts are summed.

**Pass 3 — Abstract Workflow nodes.** Groups Insight nodes by the Skill or Workflow they point to via `discovered_in` edges. When a Skill has accumulated ≥ `min_insights_for_workflow` (default 3) Insights, a new `Workflow` abstraction node is synthesised above them (e.g. `"Workflow: VASP single-point"`). The new node links back to all contributing Insights. This is how higher-level procedural knowledge emerges from accumulated experience.

---

### 5. `migrate.py` — One-time migration

Reads the existing `MEMORY.md`, converts each non-empty bullet line into an `Insight` node, and upserts it into the graph. Deduplication applies, so re-running is safe. The original `MEMORY.md` is kept as a read-only archive; no new writes go to it.

---

### 6. `kg_state.py` — Persistent pipeline state

Reads and writes `agents/MatCreator/.adk/.kg_state.json`. Tracks:

- `exec_completion_count` — global counter incremented by the orchestrator after every successful execution phase. Persists across server restarts so the synthesizer fires every 10 real executions, not every 10 within a single session.
- `last_synthesizer_run` — ISO-8601 timestamp of the most recent synthesizer invocation.

---

## Integration with the Agent System

### Thinking agent (`thinking_agent/agent.py`)

Two new tools replace `read_memory` / `update_memory` as the primary memory interface:

| Tool | When to call |
|------|-------------|
| `query_knowledge_graph(query, ...)` | At the start of planning, before drafting a plan |
| `save_to_knowledge_graph(content, context)` | Immediately after a significant finding, within the same session |
| `write_session_summary(summary)` | Once per session, after execution completes and results are available |

`read_memory` and `update_memory` are retained as fallbacks for manual use and backward compatibility, but are no longer called by default.

### Orchestrator (`orchestrator/agent.py`)

After every successful execution phase, the orchestrator calls:

```python
run_knowledge_extractor(ctx.session.id)   # always
run_knowledge_synthesizer()               # every 10th execution (cross-session)
```

Both calls are wrapped in a `try/except` so a failure in the knowledge pipeline never interrupts the main agent loop.

---

## Data Flow Diagram

```
User request
     │
     ▼
 Thinking agent
  ├── query_knowledge_graph(goal)    → reads kg_nodes / NetworkX BFS
  ├── drafts plan
  └── confirm_plan_and_start_execution
     │
     ▼
 Execution agent
  └── runs steps → writes trajectories/{session_id}.jsonl (per step)
     │
     ▼
 Thinking agent (back in planner after execution)
  └── write_session_summary(...)    → writes trajectories/{session_id}_summary.json
     │
     ▼
 Orchestrator post-execution hook
  ├── KnowledgeExtractor
  │    ├── reads .jsonl  (what was done, results)
  │    ├── reads _summary.json  (why, decisions, failures)
  │    ├── calls LLM → JSON array of {type, name, description, relations}
  │    └── upserts nodes + edges into knowledge_graph.db
  │
  └── KnowledgeSynthesizer (every 10 executions, cross-session)
       ├── Pass 1: prune stale unreferenced nodes
       ├── Pass 2: merge similar_to clusters (union-find → canonical node)
       └── Pass 3: abstract Workflow nodes from Insight clusters
```

---

## File Reference

```
agents/MatCreator/knowledge/
  __init__.py          public API
  schema.py            SQLAlchemy ORM: KgNode, KgEdge
  graph_store.py       CRUD + fuzzy dedup + NetworkX loader
  query.py             query_knowledge_graph, save_to_knowledge_graph
  extractor.py         KnowledgeExtractor (post-session, LLM-based)
  synthesizer.py       KnowledgeSynthesizer (prune / merge / abstract)
  migrate.py           one-time MEMORY.md → graph migration
  kg_state.py          persistent cross-session pipeline counters

agents/MatCreator/agents/thinking_agent/
  memory.py            re-exports graph tools; keeps legacy read/update_memory
  session_summary.py   write_session_summary tool + SessionSummary schema

agents/MatCreator/.adk/
  session.db                       ADK session store (managed by ADK)
  knowledge_graph.db               SQLite graph database
  .kg_state.json                   cross-session pipeline counters

{WORKSPACE_ROOT}/trajectories/
  {session_id}.jsonl               per-step execution log (input to extractor)
  {session_id}_summary.json        session-level narrative (input to extractor)
```
