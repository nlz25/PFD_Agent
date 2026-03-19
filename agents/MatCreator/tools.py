# Tools configuration 
from google.adk.tools.agent_tool import AgentTool
from google.adk.tools.mcp_tool import McpToolset
from .sub_agents.sql_agent.agent import sql_agent
from google.adk.tools.mcp_tool.mcp_session_manager import SseServerParams


## All tool sets
TOOLSETS = []

# Database toolset
TOOLSETS.append(AgentTool(sql_agent))

TOOLSETS.append(
    McpToolset(
        connection_params=SseServerParams(
        url="http://localhost:50001/sse", # Or any other MCP server URL
        sse_read_timeout=3600,  # Set SSE timeout to 3600 seconds
        )
    )
    )

# DPA toolset
TOOLSETS.append(
    McpToolset(
        connection_params=SseServerParams(
        url="http://localhost:50002/sse", # Or any other MCP server URL
        sse_read_timeout=3600,  # Set SSE timeout to 3600 seconds
        )
    )
)

# STRUCTURE toolset
TOOLSETS.append(
    McpToolset(
    connection_params=SseServerParams(
        url="http://localhost:50004/sse", # Or any other MCP server URL
        sse_read_timeout=3600,  # Set SSE timeout to 3600 seconds
    )
)
)

# ABACUS toolset
TOOLSETS.append(
    McpToolset(
    connection_params=SseServerParams(
        url="http://localhost:50003/sse", # Or any other MCP server URL
        sse_read_timeout=3600,  # Set SSE timeout to 3600 seconds
    )
)
)

# VASP toolset
TOOLSETS.append(
    McpToolset(
    connection_params=SseServerParams(
        url="http://localhost:50005/sse", # Or any other MCP server URL
        sse_read_timeout=7200,  # Set SSE timeout to 3600 seconds
    ),
    tool_filter=[
        "vasp_relaxation_tool",
        "vasp_scf_tool",
        "vasp_scf_results_tool",
        "vasp_nscf_kpath_tool",
        "vasp_nscf_uniform_tool",
    ]
)
)

## MatterGen toolset
TOOLSETS.append(
    McpToolset(
    connection_params=SseServerParams(
        url="http://localhost:50006/sse", # Or any other MCP server URL
        sse_read_timeout=7200,  # Set SSE timeout to 7200 seconds
    )
)
)