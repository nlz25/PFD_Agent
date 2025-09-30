import os
from anyio import Path
from dotenv import load_dotenv
from google.adk.agents import LlmAgent
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import SseServerParams, StdioConnectionParams
from dp.agent.adapter.adk import CalculationMCPToolset
from .prompt import (
    FTAGENT_NAME,
    FTAEGNTInstruction, 
    FTAEGNTDescription
    )
from ..utils.llm_config import LlmConfig

# load_dotenv()

"""
LOCAL_EXECUTOR = {
    "type": "local"
}
HTTPS_STORAGE = {
  "type": "https",
  "plugin": {
        "type": "bohrium",
        "username": os.getenv("BOHRIUM_EMAIL"),
        "password": os.getenv("BOHRIUM_PASSWORD"),
        "project_id": int(os.getenv("BOHRIUM_PROJECT_ID"))
    }
}

mcp_tools_ft = CalculationMCPToolset(
    connection_params = SseServerParams(url="http://dbhj1364970.bohrium.tech/sse"),
    # storage = HTTPS_STORAGE,
    # executor = LOCAL_EXECUTOR,
)
"""


mcp_tools_ft = McpToolset(
    #connection_params = SseServerParams(
    #    url="http://dbhj1364970.bohrium.tech:50001/sse",
    #    ),
    connection_params = StdioConnectionParams(
        server_params={
            "command": "python",
                "args": [os.getenv("TRAIN_MCP_SERVER")],
                "cwd": str(Path(os.getenv("TRAIN_MCP_SERVER")).parent),
                "env": {"BASE_MODEL_PATH": os.getenv("BASE_MODEL_PATH")}
            },
        timeout=600
        ),
    
)

def init_ft_agent(config):
    ft_agent = LlmAgent(
        name = FTAGENT_NAME,
        model= config.ali,
        instruction = FTAEGNTInstruction,
        description = FTAEGNTDescription,
        tools=[mcp_tools_ft],
        # output_key="query_result",
        # before_model_callback=update_invoke_message,
        # after_tool_callback=save_query_results,
    )
    return ft_agent

# Example usage
root_agent = init_ft_agent(LlmConfig)