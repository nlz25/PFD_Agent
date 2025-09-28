import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client
from google.adk.tools.mcp_tool import McpTool, McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import (
    MCPSessionManager,
    SseConnectionParams,
    StdioConnectionParams,
    SseServerParams
    )

# Ensure environment variables (e.g., ASE_DB_PATH) are loaded
load_dotenv(override=True)


def _extract_payload(result: Any) -> Any:
    """Best-effort extraction of JSON payload from a CallToolResult."""
    content_items = getattr(result, "content", None)
    if not content_items:
        return getattr(result, "dict", lambda: result)()

    for content in content_items:
        text = getattr(content, "text", None)
        if not text:
            continue
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text
    return getattr(result, "dict", lambda: result)()





tool_calls=[
    {
        "functionCall": {
            "name": "query_compounds",
            "args": {
                "selector": {"formula": "Si32"},
                #"output_dir": "./exported_structures",
                #"fmt": "extxyz"
            }
        }
    },
    {
        "functionCall": {
            "name": "export_entries",
            "args": {
                "ids": [1, 2, 3],
                "output_dir": "./exported_structures",
                "fmt": "extxyz"
            }
        }
    }
    
    
] 


async def run_tests(
    db_path: Optional[Path],
) -> None:

    server_env = os.environ.copy()
    if db_path is not None:
        server_env["ASE_DB_PATH"] = str(db_path.expanduser().resolve())
        
    toolset = McpToolset(
        connection_params=StdioConnectionParams(
            server_params={
               "command": "python",               # interpreter or executable
                "args": ["server.py","--transport","stdio"],       # arguments (module or script path)
                "cwd": ".",                        # optional: working directory
                "env": None                        # optional: environment vars
            },
            timeout=10,  # Optional: timeout in seconds (default: 60)
        )
    )

    tools: List[McpTool] = await toolset.get_tools()
    name_to_tool = {tool.name: tool for tool in tools}

    for tool_call in tool_calls:
        function_call = tool_call.get("functionCall", {})
        tool_name = function_call.get("name")
        args = function_call.get("args", {})
        
        if tool_name:
            tool = name_to_tool.get(tool_name)
            if tool is None:
                print(f"Tool '{tool_name}' not found. Available tools: {list(name_to_tool.keys())}")
                continue
            print(f"Running tool '{tool_name}' with args: {args}")
            try:
                result = await tool.run_async(args=args, tool_context=None)
                
                payload = _extract_payload(result)
                print(f"Result from '{tool_name}':\n", json.dumps(payload, indent=2))
            except Exception as e:
                print(f"Error running tool '{tool_name}': {e}")
    # need to close the session
    await toolset.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test ASE database MCP server tools")
    parser.add_argument(
        "--db-path",
        type=Path,
        default=None,
        help="Optional path to ase.db (overrides ASE_DB_PATH)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        asyncio.run(
            run_tests(
                db_path=args.db_path,
            )
        )
    except KeyboardInterrupt:
        print("Interrupted by user")
    except Exception as exc:
        print(f"Error running tests: {exc}")


if __name__ == "__main__":
    main()
