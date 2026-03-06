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
instruction="""
You are an agent responsible for querying, export and update of materials dataset.

Use this flow for dataset search:
1) search for available dataset domains:
  -`database_sql_agent` to generate one safe SELECT.
  - `validate_sql_code_query`.
  - `query_information_database` to check available domain datasets (e.g., "domain_SemiCond").
2) Use `query_compounds` to find target dataframes in the selected domain dataset.

When preparing training/validation dataset for machine learning force fields,
always export to extxyz format.
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
    #disallow_transfer_to_parent=True,
    #disallow_transfer_to_peers=True,
    tools=[
        AgentTool(sql_agent),
        toolset,
    ]
)