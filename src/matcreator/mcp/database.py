import os
port = os.environ.get("DATABASE_AGENT_PORT", "50001")
host = os.environ.get("DATABASE_AGENT_HOST", "0.0.0.0")
model = os.environ.get("DATABASE_AGENT_MODE", "fastmcp")


if model == "fastmcp":
    from mcp.server.fastmcp import FastMCP
    mcp = FastMCP("DATABASE_AGENT", port=port, host=host)
elif model == "test": # For unit test of models
    class MCP:
        def tool(self):
            def decorator(func):
                return func
            return decorator
    mcp = MCP()
else:
    print("Please set the environment variable DATABASE_AGENT_MODEL to dp, fastmcp or test.")
    raise ValueError("Invalid DATABASE_AGENT_MODEL. Please set it to dp, fastmcp or test.")