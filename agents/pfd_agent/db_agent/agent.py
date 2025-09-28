import os
from pathlib import Path
from google.adk.agents import LlmAgent
from google.adk.tools.mcp_tool import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from dotenv import load_dotenv
from .prompt import GlobalInstruction, AgentInstruction, AgentDescription
from ..utils.llm_config import LlmConfig

load_dotenv()

toolset = McpToolset(
        connection_params=StdioConnectionParams(
        server_params={
                "command": "python",
                "args": [os.getenv("DB_MCP_SERVER")],
                "cwd": str(Path(os.getenv("DB_MCP_SERVER")).parent),
                "env": {"ASE_DB_PATH": os.getenv("ASE_DB_PATH")}
                },
            timeout=90,
            )
        )

def init_db_agent(config):
    """
    Initialize a database agent with the given configuration.
    
    Args:
        llm_config: LLM configuration object 
        
    Returns:
        DatabaseAgent: Configured database agent instance
    """
    db_agent = LlmAgent(
        name='ase_db_agent',
        model=config.ali, 
        description=AgentDescription,
        instruction=AgentInstruction,
        tools=[toolset],
        global_instruction=GlobalInstruction)
    
    return db_agent

root_agent = init_db_agent(LlmConfig)






