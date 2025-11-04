import os
import json
import time
import importlib
import traceback
from typing import List
from importlib.metadata import version
import argparse

__version__ = version("matcreator")

AVAILABLE_MODULES = ["db"]

ENVS = {
    
    # -------------------- DatabaseAgent settings ---------------------
    "DATABASE_AGENT_WORK_PATH": "/tmp/database/",

    # connection settings
    "DATABASE_AGENT_TRANSPORT": "sse",  # sse, streamable-http
    "DATABASE_AGENT_HOST": "localhost",
    "DATABASE_AGENT_PORT": "50001", 
    "DATABASE_AGENT_MODEL": "fastmcp",  # fastmcp, abacus, dp
    
    # LLM settings
    "LLM_MODEL": "",
    "LLM_API_KEY": "",
    "LLM_BASE_URL": "",

    # ------------------- Database settings --------------------------
    "ASE_DB_PATH": "",  # The path to the ASE database file, e.g., /path/to/ase.db
    
    "_comments":{}
}

def set_envs(transport_input=None, model_input=None, port_input=None, host_input=None):
    """
    Set environment variables for AbacusAgent.
    
    Args:
        transport_input (str, optional): The transport protocol to use. Defaults to None.
        model_input (str, optional): The model to use. Defaults to None.
        port_input (int, optional): The port number to run the MCP server on. Defaults to None.
        host_input (str, optional): The host address to run the MCP server on. Defaults to None.
    
    Returns:
        dict: The environment variables that have been set.
    
    Notes:
        - The input parameters has higher priority than the default values in `ENVS`.
        - If the `~/.abacusagent/env.json` file does not exist, it will be created with default values.
    """
    # read setting in ~/.abacusagent/env.json
    envjson_file = os.path.expanduser("~/.database_agent/env.json")
    if os.path.isfile(envjson_file):
        envjson = json.load(open(envjson_file, "r"))
    else:
        envjson = {}
    update_envjson = False    
    for key, value in ENVS.items():
        if key not in envjson:
            envjson[key] = value
            update_envjson = True
    
    if transport_input is not None:
        envjson["DATABASE_AGENT_TRANSPORT"] = str(transport_input)
    if port_input is not None:
        envjson["DATABASE_AGENT_PORT"] = str(port_input)
    if host_input is not None:
        envjson["DATABASE_AGENT_HOST"] = str(host_input)
    if model_input is not None:
        envjson["DATABASE_AGENT_MODEL"] = str(model_input)
        
    for key, value in envjson.items():
        os.environ[key] = str(value)
    
    if update_envjson:
        # write envjson to ~/.abacusagent/env.json
        os.makedirs(os.path.dirname(envjson_file), exist_ok=True)
        del envjson["_comments"]  # remove comments before writing
        envjson["_comments"] = ENVS["_comments"]  # add comments from ENVS
        json.dump(
            envjson,
            open(envjson_file, "w"),
            indent=4
        )
    return envjson
    
def create_workpath(work_path=None):
    """
    Create the working directory for AbacusAgent, and change the current working directory to it.
    
    Args:
        work_path (str, optional): The path to the working directory. If None, a default path will be used.
    
    Returns:
        str: The path to the working directory.
    """
    if work_path is None:
        work_path = os.environ.get("DATABASE_AGENT_WORK_PATH", "/tmp/database") + f"/{time.strftime('%Y%m%d%H%M%S')}"
        
    os.makedirs(work_path, exist_ok=True)
    cwd = os.getcwd()
    os.chdir(work_path)
    print(f"Changed working directory to: {work_path}")
    # write the environment variables to a file
    json.dump({
        k: os.environ.get(k) for k in ENVS.keys()
    }.update({"DATABASE_AGENT_START_PATH": cwd}), 
        open("env.json", "w"), indent=4)
    
    return work_path    

def load_tools(screen_modules: List[str] = []):
    """
    Load all tools from the abacusagent package.
    """
    for module in AVAILABLE_MODULES:
        if module in ["utils", "comm", "tool_wrapper"] + screen_modules:
            continue  # skipt __init__.py and utils.py
        module_name = f"matcreator.tools.{module}"
        try:
            module = importlib.import_module(module_name)
            print(f"✅ Successfully loaded: {module_name}")
        except Exception as e:
            traceback.print_exc()
            print(f"⚠️ Failed to load {module_name}: {str(e)}")

def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="DatabaseAgent Command Line Interface")
    
    parser.add_argument(
        "--transport",
        type=str,
        default=None,
        choices=["sse", "streamable-http"],
        help="Transport protocol to use (default: sse), choices: sse, streamable-http"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=["fastmcp"],
        help="Model to use (default: dp), choices: fastmcp, dp"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port to run the MCP server on (default: 50001)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="Host to run the MCP server on (default: localhost)"
    )
    parser.add_argument(
        "--create",
        type=str,
        nargs='?',
        default=None,
        const=".",
        help="Create a template for Google ADK agent in the specified directory (default: current directory)"
    )
    parser.add_argument(
        "--screen-modules",
        type=str,
        nargs='*',
        default=[],
        help="List of modules to screen for loading. If not specified, all modules will be loaded."
    )
    
    args = parser.parse_args()
    
    return args

def print_address():
    """
    Print the address of the MCP server based on environment variables.
    """
    address = f"{os.environ['DATABASE_AGENT_HOST']}:{os.environ['DATABASE_AGENT_PORT']}"
    if os.environ["DATABASE_AGENT_TRANSPORT"] == "sse":
        print("Address:", address + "/sse")
    elif os.environ["DATABASE_AGENT_TRANSPORT"] == "streamable-http":
        print("Address:", address + "/mcp")
    else:
        raise ValueError("Invalid transport protocol specified. Use 'sse' or 'streamable-http'.")

def print_version():
    """
    Print the version of the AbacusAgent.
    """
    print(f"\nDatabaseAgentTools Version: {__version__}")
    print("For more information, visit: https://github.com/deepmodeling/ABACUS-agent-tools\n")

def main():
    """
    Main function to run the MCP tool.
    """
    print_version()
    args = parse_args()  
    set_envs(
        transport_input=args.transport,
        model_input=args.model,
        port_input=args.port, 
        host_input=args.host)
    create_workpath()
    from matcreator.mcp.database import mcp
    load_tools([]) 
    print_address()
    mcp.run(transport=os.environ["DATABASE_AGENT_TRANSPORT"])

if __name__ == "__main__":
    main()