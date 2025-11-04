from pathlib import Path
import importlib
import os
import argparse
from typing import List
from importlib.metadata import version
import traceback
__version__ = version("matcreator")


import os
import json
import time

AVAILABLE_MODULES = [
    "expl", "dft", "train","log"]

ENVS = {
    
    # -------------------- DatabaseAgent settings ---------------------
    "PFD_AGENT_WORK_PATH": "/tmp/pfdagent/",

    # connection settings
    "PFD_AGENT_TRANSPORT": "sse",  # sse, streamable-http
    "PFD_AGENT_HOST": "localhost",
    "PFD_AGENT_PORT": "50001", 
    "PFD_AGENT_MODEL": "fastmcp",  # fastmcp, abacus, dp
    
    # LLM settings
    "LLM_MODEL": "",
    "LLM_API_KEY": "",
    "LLM_BASE_URL": "",

    # bohrium settings
    "BOHRIUM_USERNAME": "",
    "BOHRIUM_PASSWORD": "",
    "BOHRIUM_PROJECT_ID": "",
    
    # -------------------- General settings--------------------------
    "DPA_MODEL_PATH": "/path/to/your/model",  # The path to the base model, e.g., /path/to/your/model
    
    # ------------------- Database settings --------------------------
    "ASE_DB_PATH": "",  # The path to the ASE database file, e.g., /path/to/ase.db
    
    # -------------------- ABACUS settings--------------------------
    "BOHRIUM_ABACUS_IMAGE": "registry.dp.tech/dptech/abacus-stable:LTSv3.10", # THE bohrium image for abacus calculations, 
    "BOHRIUM_ABACUS_MACHINE": "c32_m64_cpu",  # THE bohrium machine for abacus calculations, c32_m64_cpu
    "BOHRIUM_ABACUS_COMMAND": "OMP_NUM_THREADS=1 mpirun -np 16 abacus",
    "ABACUSAGENT_SUBMIT_TYPE": "bohrium",  # local, bohrium
    
    # abacus pp orb settings
    "ABACUS_COMMAND": "abacus",  # abacus executable command
    "ABACUS_PP_PATH": "",  # abacus pseudopotential library path
    "ABACUS_ORB_PATH": "",  # abacus orbital library path
    "ABACUS_SOC_PP_PATH": "",  # abacus SOC pseudopotential library path
    "ABACUS_SOC_ORB_PATH": "",  # abacus SOC orbital library path


    # PYATB settings
    "PYATB_COMMAND": "OMP_NUM_THREADS=1 pyatb",
    
    "_comments":{
        "ABACUS_WORK_PATH": "The working directory for AbacusAgent, where all temporary files will be stored.",
        "ABACUS_SUBMIT_TYPE": "The type of submission for ABACUS, can be local or bohrium.",
        "ABACUSAGENT_TRANSPORT": "The transport protocol for AbacusAgent, can be 'sse' or 'streamable-http'.",
        "ABACUSAGENT_HOST": "The host address for the AbacusAgent server.",
        "ABACUSAGENT_PORT": "The port number for the AbacusAgent server.",
        "ABACUSAGENT_MODEL": "The model to use for AbacusAgent, can be 'fastmcp', 'test', or 'dp'.",
        "LLM_MODEL": "The model name for the LLM to use. Like: openai/qwen-turbo, deepseek/deepseek-chat",
        "LLM_API_KEY": "The API key for the LLM service.",
        "LLM_BASE_URL": "The base URL for the LLM service, if applicable.",
        "BOHRIUM_USERNAME": "The username for Bohrium.",        
        "BOHRIUM_PASSWORD": "The password for Bohrium.",
        "BOHRIUM_PROJECT_ID": "The project ID for Bohrium.",
        "BOHRIUM_ABACUS_IMAGE": "The image for Abacus on Bohrium.",
        "BOHRIUM_ABACUS_MACHINE": "The machine type for Abacus on Bohrium.",
        "BOHRIUM_ABACUS_COMMAND": "The command to run Abacus on Bohrium",
        "ABACUS_COMMAND": "The command to execute Abacus on local machine.",
        "ABACUS_PP_PATH": "The path to the pseudopotential library for Abacus.",
        "ABACUS_ORB_PATH": "The path to the orbital library for ABACUS_PP_PATH",
        "ABACUS_SOC_PP_PATH": "The path to the SOC pseudopotential library for Abacus.",
        "ABACUS_SOC_ORB_PATH": "The path to the orbital library for ABACUS_SOC_PP_PATH.",
        "PYATB_COMMAND": "The command to execute PYATB on local machine.",
        "_comments": "This dictionary contains the default environment variables for AbacusAgent."
    }
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
    envjson_file = os.path.expanduser("~/.pfd_agent/env.json")
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
        envjson["PFD_AGENT_TRANSPORT"] = str(transport_input)
    if port_input is not None:
        envjson["PFD_AGENT_PORT"] = str(port_input)
    if host_input is not None:
        envjson["PFD_AGENT_HOST"] = str(host_input)
    if model_input is not None:
        envjson["PFD_AGENT_MODEL"] = str(model_input)
        
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
        work_path = os.environ.get("PFD_AGENT_WORK_PATH", "/tmp/pfd_agent") + f"/{time.strftime('%Y%m%d%H%M%S')}"
        
    os.makedirs(work_path, exist_ok=True)
    cwd = os.getcwd()
    os.chdir(work_path)
    print(f"Changed working directory to: {work_path}")
    # write the environment variables to a file
    json.dump({
        k: os.environ.get(k) for k in ENVS.keys()
    }.update({"PFD_AGENT_START_PATH": cwd}), 
        open("env.json", "w"), indent=4)
    
    return work_path    

def load_tools(screen_modules: List[str] = []):
    """
    Selectively load tool modules, skipping those in the screen_modules list.
    """
    print("check")
    for module in AVAILABLE_MODULES:
        if module in ["utils", "comm", "tool_wrapper"] + screen_modules:
            continue  # skipt __init__.py and utils.py

        module_name = f"matcreator.tools.{module}"
        #print(f"Loading module: {module_name} ... ", end="")
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
    parser = argparse.ArgumentParser(description="PFD_Agent Command Line Interface")
    
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
        choices=["fastmcp", "dp"],
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
    address = f"{os.environ['PFD_AGENT_HOST']}:{os.environ['PFD_AGENT_PORT']}"
    if os.environ["PFD_AGENT_TRANSPORT"] == "sse":
        print("Address:", address + "/sse")
    elif os.environ["PFD_AGENT_TRANSPORT"] == "streamable-http":
        print("Address:", address + "/mcp")
    else:
        raise ValueError("Invalid transport protocol specified. Use 'sse' or 'streamable-http'.")

def print_version():
    """
    Print the version of the AbacusAgent.
    """
    print(f"\nAbacusAgentTools Version: {__version__}")
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

    from matcreator.mcp.pfd import mcp
    print("Loading tools...")
    load_tools([]) 

    print_address()
    mcp.run(transport=os.environ["PFD_AGENT_TRANSPORT"])

if __name__ == "__main__":
    main()
