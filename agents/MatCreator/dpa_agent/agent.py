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
You are the DPA Agent for Deep Potential workflows. You inspect/split datasets,
validate configs, train DPA models, and run ASE-based MD and structure optimization using DPA.
"""

instruction ="""
- Capabilities (tools)
    - Training: train_input_doc, check_train_data, check_input, training
    - Simulation: get_base_model_path, run_molecular_dynamics, optimize_structure

- Preconditions
    - Training: have train_data; optionally split with check_train_data; validate with check_input before training.
    - MD/Opt: require a model path; if missing, resolve via get_base_model_path. For multi‑head DPA, set `head`.

- Minimal flows
    - Training:
        1) read train_input_doc
        2) check_train_data 
        3) check_input 
        4)training
        5) Report model and log absolute paths; include test metrics if available.
    - MD simulation:
        1) get_base_model_path(model_path?)
        2) run_molecular_dynamics(initial_structure, stages, model_path, calc_args)
        3) Report trajectory paths and log path.
    - Optimization:
        1) get_base_model_path(model_path?)
        2) optimize_structure(input_structure, model_path, relax_cell?)
        3) Report optimized structure path and final energy.

- Defaults and tips
    - Prefer quick validations first (small splits, short MD stages, modest relax steps).
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

# native supported tools
def list_calculators() -> List[Dict[str, Any]]:
    """List input requirements for DPA calculator.

    Returns:
      A list of dicts, one per registered calculator, with:
        - name: calculator key to use as model_style
        - description: short summary (aligned to wrapper implementation)
        - requires_model_path: whether `model_path` must be provided
        - optional_calc_args: optional kwargs for calculator initialization
        - md_supported: whether suitable for MD in this toolkit
        - notes: extra hints
        - example: minimal example for run_molecular_dynamics
    """
    # Specs derived from src/pfd_agent_tool/modules/expl/calculator.py wrappers
    specs: Dict[str, Dict[str, Any]] = {
        "dpa": {
            "description": "DeepMD (deepmd-kit) ASE calculator wrapper.",
            "requires_model_path": True,
            "optional_calc_args": ["head"],
            "md_supported": True,
            "notes": "Always specify `head` for multi-head models to select the desired potential. Default to `MP_traj_v024_alldata_mixu` if not specified.",
            "example": "run_molecular_dynamics(initial_structure=Path('in.xyz'), stages=stages, model_style='dpa', model_path=Path('.tests/dpa/DPA2_medium_28_10M_rc0.pt'),head='MP_traj_v024_alldata_mixu')",
        }
    }

    result: List[Dict[Any]] = []
    for name in specs.keys():
        spec = specs.get(name, {})
        result.append({
            "name": name,
            "description": spec.get("description", ""),
            "requires_model_path": spec.get("requires_model_path", False),
            "required_init_params": spec.get("required_init_params", []),
            "optional_init_params": spec.get("optional_init_params", []),
            "md_supported": spec.get("md_supported", True),
            "notes": spec.get("notes", ""),
            "example": spec.get("example", ""),
        })
    return result




# tools hosted by MCP server
toolset = McpToolset(
    connection_params=SseServerParams(
        url="http://localhost:50002/sse", # Or any other MCP server URL
        sse_read_timeout=3600,  # Set SSE timeout to 3600 seconds
    ),
    tool_filter=[
        "check_train_data",
        "check_input",
        "training",
        "train_input_doc",
        "run_molecular_dynamics",
        "optimize_structure",
        "get_base_model_path",
        "ase_calculation",
    ],
    #executor_map = EXECUTOR_MAP,
    #executor=executor["local"],
    #storage=STORAGE, 
)

dpa_agent = LlmAgent(
    name='dpa_agent',
    model=LiteLlm(
        model=model_name,
        base_url=model_base_url,
        api_key=model_api_key
    ),
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
    description=description,
    instruction=instruction,
    tools=[
        toolset,
        list_calculators,
        ],
    after_tool_callback=after_tool_callback
)