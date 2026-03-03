from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools.agent_tool import AgentTool
from google.adk.tools.mcp_tool import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import SseServerParams
import os
from ..constants import LLM_MODEL, LLM_API_KEY, LLM_BASE_URL
from .sql_agent.agent import sql_agent
from ..callbacks import after_tool_callback
from dotenv import load_dotenv
from pathlib import Path
_script_dir = Path(__file__).parent
load_dotenv(_script_dir / ".env", override=True)
# Set the secret key in ~/.abacusagent/env.json or as an environment variable, or modify the code t
model_name = os.environ.get("LLM_MODEL", LLM_MODEL)
model_api_key = os.environ.get("LLM_API_KEY", LLM_API_KEY)
model_base_url = os.environ.get("LLM_BASE_URL", LLM_BASE_URL)

description = """
You are the Database Agent for materials datasets. You help users query datasets (in ASE db format)
stored in a normalized SQLite database organized into nodes (groups of datasets sharing the same DFT
settings) and datasets (one per element-set per node). You assist with finding relevant datasets by
chemical composition or node metadata, inspecting and querying structures within datasets, exporting
structures, and saving new calculation data to an appropriate node.
"""

instruction = """
Be concise, safe, and tool-driven.

Use this flow for dataset search:
1) `database_sql_agent` to generate one safe SELECT.
2) `validate_sql_code_query`.
3) `query_information_database`.

Rules:
- INFO_DB_PATH must be available (ask once if missing).
- For composition queries, use exact formula only: `datasets.elements = 'A-B'` (sorted, hyphen-joined).
- If user provides structure files, call `read_user_structure` first, then query by exact formula.
- Keep ASE frame lookups concise with `query_compounds` (selection + limit).
- Default export format: extxyz.
- After `save_extxyz_to_db`, stop (no extra actions).

Response format:
- Plan: 1–3 short bullets.
- Action: tool called.
- Result: key output (include absolute paths).
- Next: immediate follow-up.
"""


toolset = McpToolset(
    connection_params=SseServerParams(
        url="http://localhost:50001/sse", # Or any other MCP server URL
        sse_read_timeout=3600,  # Set SSE timeout to 3600 seconds
    )
)

database_agent = LlmAgent(
    name='database_agent',
    model=LiteLlm(
        model=model_name,
        base_url=model_base_url,
        api_key=model_api_key
    ),
    after_tool_callback=after_tool_callback,
    description=description,
    instruction=instruction,
    tools=[
        AgentTool(sql_agent),
        toolset,
    ]
)