from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools.mcp_tool import MCPToolset
from google.adk.tools.mcp_tool.mcp_session_manager import SseServerParams

import os
import json

# Load Database Agent environment (prefer ~/.database_agent/env.json, then env vars)
env_file = os.path.expanduser("~/.database_agent/env.json")
if os.path.isfile(env_file):
    env = json.load(open(env_file, "r"))
else:
    env = {}

# LLM configuration
model_name = env.get("LLM_MODEL", os.environ.get("LLM_MODEL", ""))
model_api_key = env.get("LLM_API_KEY", os.environ.get("LLM_API_KEY", ""))
model_base_url = env.get("LLM_BASE_URL", os.environ.get("LLM_BASE_URL", ""))

# Database MCP server connection
db_host = env.get("DATABASE_AGENT_HOST", os.environ.get("DATABASE_AGENT_HOST", "localhost"))
db_port = str(env.get("DATABASE_AGENT_PORT", os.environ.get("DATABASE_AGENT_PORT", "50002")))
sse_url = f"http://{db_host}:{db_port}/sse"

description = """
You are the Database Agent for materials datasets. You help users inspect, query, and export
structures from an ASE database quickly and safely.
"""

instruction = """
Operate with minimal, safe steps. Validate inputs early, call exactly the right tool, and report
clear outputs with absolute paths.

Capabilities (tools provided by the Database MCP server):
- read_user_structure(structures): Parse one or more structure files, aggregate frames into a single
  extxyz, and return chemical formula lists to guide queries.
- query_compounds(selection, exclusive_elements, limit, db_path, custom_args): Run flexible queries
  against the ASE DB and summarize matches (ids, formulas, metadata).
- export_entries(ids, fmt, db_path): Export selected entries to a combined structure file and a
  line-delimited metadata JSON.

Operating rules:
- Preconditions: An ASE database must be accessible. If ASE_DB_PATH is not set, ask once for the
  path or an explicit db_path.
- Clarify the user intent in one sentence, then plan 1–3 minimal steps.
- If the user provided structure files, first call read_user_structure to extract compositions and
  propose a query.
- For queries, construct a concise selection (e.g., 'Si,O', 'energy<-1.0, pbc=True', or id list).
  Use limit and sort when appropriate for quick previews.
- After query_compounds, summarize count, unique formulas, and show a short preview of ids. Propose
  concrete next actions (e.g., which ids to export, how many, and in which format).
- When exporting, default to fmt=extxyz unless the user requests otherwise. Always print absolute
  output paths returned by the tool.

Failure handling:
- If a tool fails or required inputs are missing, show the exact issue and suggest a minimal fix.

Response format:
- Plan: 1–3 bullets describing next step(s).
- Action: the exact tool you will call in this agent context.
- Result: brief summary with key outputs, including absolute paths.
- Next: the immediate follow‑up or final recap.
"""

toolset = MCPToolset(
    connection_params=SseServerParams(
        url="http://localhost:50002/sse",
        sse_read_timeout=3600,
    ),
)

database_agent = Agent(
    name="database_agent",
    model=LiteLlm(
        model=model_name,
        base_url=model_base_url,
        api_key=model_api_key,
    ),
    description=description,
    instruction=instruction,
    tools=[toolset],
)
