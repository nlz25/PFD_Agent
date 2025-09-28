import os
from pathlib import Path
from typing import override, AsyncGenerator, Optional
from google.adk.agents import LlmAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.genai import types
from google.adk.tools.mcp_tool import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from google.adk.sessions import InMemorySessionService
from google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService # Optional
from google.adk.runners import Runner
from ..callback import init_before_agent, before_model_callback, after_model_callback, \
    enforce_single_tool_call
from .prompt import GlobalInstruction, AgentInstruction, AgentDescription


class DatabaseAgent(LlmAgent):
    """Database agent that provides access to ASE database through MCP server."""
    server_path:  Optional[str] = None
    db_path: Optional[str] = None
    toolset: Optional[McpToolset] = None  # initialized later

    def __init__(
        self, 
        llm_config, 
        server_path: Optional[str] = None, 
        db_path: Optional[str] = None):

        # Set up MCP toolset for database server
        server_env = os.environ.copy()
        if db_path:
            server_env["ASE_DB_PATH"] = str(Path(db_path).expanduser().resolve())

        toolset = McpToolset(
            connection_params=StdioConnectionParams(
                server_params={
                    "command": "python",
                    "args": [server_path, "--transport", "stdio"],
                    "cwd": str(Path(server_path).parent),
                    "env": server_env
                },
                timeout=30,
            )
        )

        prepare_state_before_agent = init_before_agent(llm_config)

        super().__init__(
            name="database_agent",
            model=llm_config.custom_model,
            description=AgentDescription,
            tools=[toolset],
            global_instruction=GlobalInstruction,
            instruction=AgentInstruction,
            before_agent_callback=prepare_state_before_agent,
            before_model_callback=[before_model_callback],
            after_model_callback=[enforce_single_tool_call, after_model_callback],
        )

        # assign after calling super().__init__
        self.server_path = server_path
        self.db_path = db_path
        self.toolset = toolset


    @override
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        """Run the agent with context processing similar to PFDAgent."""
        prompt = ""
        if ctx.user_content and ctx.user_content.parts:
            for part in ctx.user_content.parts:
                if part.text:
                    prompt += part.text
                elif part.inline_data:
                    pass
                elif part.file_data:
                    prompt += f", file_url = {part.file_data.file_uri}"
                    yield Event(
                        invocation_id=ctx.invocation_id,
                        author=self.name,
                        branch=ctx.branch,
                        content=types.Content(parts=[types.Part(text=prompt)], role="system")
                    )

        async for event in super()._run_async_impl(ctx):
            yield event

    async def close(self):
        """Clean up the MCP toolset connection."""
        if hasattr(self.toolset, 'close'):
            await self.toolset.close()


async def get_agent_async(llm_config, server_path: Optional[str] = None, db_path: Optional[str] = None):
    """
    Initialize a database agent with the given configuration.
    
    Args:
        llm_config: LLM configuration object 
        server_path: Optional path to database MCP server
        db_path: Optional path to ASE database file
        
    Returns:
        DatabaseAgent: Configured database agent instance
    """
    database_agent = DatabaseAgent(llm_config, server_path=server_path, db_path=db_path)
    # Optional: add tracking if needed
    # track_adk_agent_recursive(database_agent, llm_config.opik_tracer)
    return database_agent, database_agent.toolset



