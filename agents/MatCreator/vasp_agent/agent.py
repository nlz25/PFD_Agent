from google.adk.agents import  LlmAgent,BaseAgent,Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools.mcp_tool import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import SseServerParams
from dp.agent.adapter.adk import CalculationMCPToolset
import os, json
from typing import List, Dict, Any
from ..constants import LLM_MODEL, LLM_API_KEY, LLM_BASE_URL, BOHRIUM_USERNAME, BOHRIUM_PASSWORD, BOHRIUM_PROJECT_ID
from dotenv import load_dotenv
from pathlib import Path
_script_dir = Path(__file__).parent
load_dotenv(_script_dir / ".env", override=True)

# Set the secret key in ~/.abacusagent/env.json or as an environment variable, or modify the code t
model_name = os.environ.get("LLM_MODEL", LLM_MODEL)
model_api_key = os.environ.get("LLM_API_KEY", LLM_API_KEY)
model_base_url = os.environ.get("LLM_BASE_URL", LLM_BASE_URL)
bohrium_username = os.environ.get("BOHRIUM_USERNAME", BOHRIUM_USERNAME)
bohrium_password = os.environ.get("BOHRIUM_PASSWORD", BOHRIUM_PASSWORD)
bohrium_project_id = int(os.environ.get("BOHRIUM_PROJECT_ID", BOHRIUM_PROJECT_ID))

description="""
You are the VASP Agent for VASP calculation workflows.
"""

instruction ="""
Operate VASP safely with minimal steps and strict validation.


Must‑follow sequence
- First, check whether the user has entered a structure. If not, create a structure according to the user's requirements.
- Then create an inputs directory (INCAR, POSCAR, POTCAR, KPOINTS). 
- Then run exactly ONE property tool per step.
- When you finish your task, ALWAYS end with: "Task complete. Transferring control back to root_agent." Then call transfer_to_agent('root_agent').

"""

executor = {
    "bohr": {
        "type": "dispatcher",
        "machine": {
            "batch_type": "Bohrium",
            "context_type": "Bohrium",
            "remote_profile": {
                "email": bohrium_username,
                "password": bohrium_password,
                "program_id": bohrium_project_id,
                "input_data": {
                    "image_name": "registry.dp.tech/dptech/prod-15454/vasp:5.4.4",
                    "job_type": "container",
                    "platform": "ali",
                    "scass_type": "c32_m64_cpu",
                },
            },
        }
    },
    "local": {"type": "local",}
}

EXECUTOR_MAP = {
    "vasp_relaxation_tool": executor["bohr"],
    "vasp_scf_tool": executor["bohr"],
    "vasp_nscf_kpath_tool": executor["bohr"],
    "vasp_nscf_uniform_tool": executor["bohr"],
    "check_calculation_status_tool":executor["bohr"],
}

STORAGE = {
    "type": "https",
    "plugin":{
        "type": "bohrium",
        "username": bohrium_username,
        "password": bohrium_password,
        "project_id": bohrium_project_id,
    }
}


# tools hosted by MCP server
toolset = McpToolset(
    connection_params=SseServerParams(
        url="http://localhost:50005/sse", # Or any other MCP server URL
        sse_read_timeout=3600,  # Set SSE timeout to 3600 seconds
    ),
    tool_filter=[
        "vasp_relaxation_tool",
        "vasp_scf_tool",
        "vasp_scf_results_tool",
        "vasp_nscf_kpath_tool",
        "vasp_nscf_uniform_tool",
    ],
    # executor=executor["local"],
    # executor_map = EXECUTOR_MAP,
    # storage=STORAGE,
)


vasp_agent = LlmAgent(
    name='vasp_agent',
    model=LiteLlm(
        model=model_name,
        base_url=model_base_url,
        api_key=model_api_key
    ),
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
    description=description,
    instruction=instruction,
    tools=[toolset],
)