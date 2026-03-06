from google.adk.agents import  LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools.mcp_tool import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import SseServerParams
from dp.agent.adapter.adk import CalculationMCPToolset
import os
from typing import List, Dict, Any
from ..constants import LLM_MODEL, LLM_API_KEY, LLM_BASE_URL, BOHRIUM_USERNAME, BOHRIUM_PASSWORD, BOHRIUM_PROJECT_ID
from ..callbacks import after_tool_callback
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
You are the Mattergen Agent. You train Mattergen conditional and/or unconditional models,
 and generate crystal structures with Mattergen models.
"""

instruction ="""
- Capabilities (tools)
    - Data conversion: mattergen_ase_convert_tool
    - Training: mattergen_train_tool,
    - Generation: mattergen_generate_tool

- Preconditions
    - Data conversion: have input structure in ase format (such as .xyz, .cif, .extxyz) or
           Mattergen's internal format (.npy for structures and .json for conditioning properties).
           specify output path and conversion type (mattergen_to_ase, ase_to_mattergen, or auto).
    - Training: have model_root and data_root, data_root must contain 'train' and 
        optionally 'val' and 'test' subdirs with .npy structures and .json conditioning properties.
        Optionally specify a list of conditioned_properties to train a conditional model.
    - Generation: 
        need to specify model_path storing input model and results_dir to store generated structures.
        conditional generation requires `conditioned_property_values` as a dict of condition names and desired values
        , e.g., `{"energy_above_hull": 0.03}`.
        Used conditions must match those previously specified at training time.
    - For all tasks: 
        `custom_cmd` can be used to specify a custom training command when needed, which will override
        the default training command constructed from the other parameters.
        `venv_root` specifies the root directory for the Python virtual environment to run the training
        or generation in. Recommended when running locally as mattergen is typically installed via UV.
        The venv will be activated before running the command and deactivated afterwards.

- Minimal flows
    - Training:
        1) convert training data to Mattergen format with mattergen_ase_convert_tool if needed;
        2) choose properties to condition on based on the available data and desired generation control;
        3) train with mattergen_train_tool using the converted data and selected conditions;
        3) report model and log absolute paths; include training status and messages if available.
    - Generation:
        1) prepare generation conditions if doing conditional generation; find model_path.
        2) generate with mattergen_generate_tool using the input model and prepared conditions;
        3) report generated structure paths and log path; include generation status and messages if available.

- Defaults and tips
    - In a specific chemical system, prefer conditioning on `chemical_system` and `energy_above_hull` properties.
    - Always return absolute artifact paths. 
    - If a tool fails, surface the exact error and propose a minimal fix.

- Response format
    - Plan: 1-3 bullets with the next step(s).
    - Action: exact tool name you will call.
    - Result: key outputs with absolute paths and essential metrics.
    - Next: immediate follow-up or stop.
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
                    "image_name": "registry.dp.tech/dptech/dp/native/prod-26745/matcreator:0.0.1",
                    "job_type": "container",
                    "platform": "ali",
                    "scass_type": "1 * NVIDIA V100_16g",
                },
            },
        }
    },
    "local": {"type": "local",}
}

EXECUTOR_MAP = {
    "run_molecular_dynamics": executor["bohr"],
    "optimize_structure": executor["bohr"],
    "training": executor["bohr"],
    "ase_calculation": executor["bohr"],
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


# Supported tools.
tool_filter=[
        "mattergen_ase_convert_tool",
        "mattergen_train_tool",
        "mattergen_generate_tool",
    ]

# tools hosted by MCP server
toolset = McpToolset(
    connection_params=SseServerParams(
        url="http://localhost:50002/sse", # Or any other MCP server URL
        sse_read_timeout=3600,  # Set SSE timeout to 3600 seconds
    ),
    #tool_filter=tool_filter,
    #executor_map = EXECUTOR_MAP,
    #executor=executor["local"],
    #storage=STORAGE, 
)

mattergen_agent = LlmAgent(
    name='mattergen_agent',
    model=LiteLlm(
        model=model_name,
        base_url=model_base_url,
        api_key=model_api_key
    ),
    disallow_transfer_to_parent=False,
    disallow_transfer_to_peers=False,
    description=description,
    instruction=instruction,
    tools=[
        toolset,
        ],
    after_tool_callback=after_tool_callback
)