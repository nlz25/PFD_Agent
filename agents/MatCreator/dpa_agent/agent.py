from google.adk.agents import  LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools.mcp_tool import MCPToolset
from google.adk.tools.mcp_tool.mcp_session_manager import SseServerParams
from google.adk.tools import agent_tool
from ..abacus_agent.agent import abacus_agent
import os, json
from typing import List, Dict, Any

# Set the secret key in ~/.abacusagent/env.json or as an environment variable, or modify the code to set it directly.
env_file = os.path.expanduser("~/.pfd_agent/env.json")
if os.path.isfile(env_file):
    env = json.load(open(env_file, "r"))
else:
    env = {}
model_name = env.get("LLM_MODEL", os.environ.get("LLM_MODEL", ""))
model_api_key = env.get("LLM_API_KEY", os.environ.get("LLM_API_KEY", ""))
model_base_url = env.get("LLM_BASE_URL", os.environ.get("LLM_BASE_URL", ""))
bohrium_username = env.get("BOHRIUM_USERNAME", os.environ.get("BOHRIUM_USERNAME", ""))
bohrium_password = env.get("BOHRIUM_PASSWORD", os.environ.get("BOHRIUM_PASSWORD", ""))
bohrium_project_id = env.get("BOHRIUM_PROJECT_ID", os.environ.get("BOHRIUM_PROJECT_ID", ""))

description="""
You are the DPA Agent for Deep Potential workflows. You inspect/split datasets,
validate configs, train DPA models, and run ASE-based MD and structure optimization using DPA.
"""

instruction ="""
- Capabilities (tools)
    - Data/Training: list_training_strategies, train_input_doc, check_train_data, check_input, training
    - Simulation: list_calculators, get_base_model_path, run_molecular_dynamics, optimize_structure

- Hard constraints (must follow)
    - Training strategy is strictly 'dpa' only. Do not propose or call any other strategy.
    - Simulation model_style is strictly 'dpa' only. Do not use other calculator names.
    - Always pass strategy='dpa' to check_input and training; always pass model_style='dpa' to
        get_base_model_path, run_molecular_dynamics, and optimize_structure.
    - If the user asks for another strategy/calculator, respond that only 'dpa' is supported and offer
        the closest 'dpa' alternative (e.g., set calc_args.head appropriately).

- Preconditions
    - Training: have train_data; optionally split with check_train_data; validate with check_input(strategy='dpa') before training.
    - MD/Opt: require a model path; if missing, resolve via get_base_model_path(model_style='dpa'). For multi‑head DPA, set calc_args.head.

- Minimal flows
    - Training (DPA only):
        1) list_training_strategies → train_input_doc('dpa') → check_train_data(train_data, ratios?)
        2) check_input(config, command, strategy='dpa') → training(..., strategy='dpa')
        3) Report model and log absolute paths; include test metrics if available.
    - MD (DPA only):
        1) list_calculators → get_base_model_path(model_style='dpa', model_path?)
        2) run_molecular_dynamics(initial_structure, stages, model_style='dpa', model_path, calc_args)
        3) Report trajectory paths and log path.
    - Optimization (DPA only):
        1) get_base_model_path(model_style='dpa', ...)
        2) optimize_structure(input_structure, model_style='dpa', model_path, relax_cell?)
        3) Report optimized structure path and final energy.

- Defaults and tips
    - Prefer quick validations first (small splits, short MD stages, modest relax steps).
    - Always return absolute artifact paths. If a tool fails, surface the exact error and propose a minimal fix.

- Response format
    - Plan: 1–3 bullets with the next step(s).
    - Action: exact tool name you will call.
    - Result: key outputs with absolute paths and essential metrics.
    - Next: immediate follow‑up or stop.
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
                    "image_name": "registry.dp.tech/dptech/dp/native/prod-22618/abacus-agent-tools:v0.2",
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
    "generate_bulk_structure": executor["local"],
    "generate_molecule_structure": executor["local"],
    "abacus_prepare": executor["local"],
    "abacus_modify_input": executor["local"],
    "abacus_modify_stru": executor["local"],
    "abacus_collect_data": executor["local"],
    "abacus_prepare_inputs_from_relax_results": executor["local"],
    "generate_bulk_structure_from_wyckoff_position": executor["local"],
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
toolset = MCPToolset(
    connection_params=SseServerParams(
        url="http://localhost:50003/sse", # Or any other MCP server URL
        sse_read_timeout=3600,  # Set SSE timeout to 3600 seconds
    ),
    tool_filter=[
        "check_train_data",
        "check_input",
        "training",
        "train_input_doc",
        #"list_training_strategies", # maybe we need a seperate implementation later
        #"list_calculators", # maybe a seperate implementation later
        "run_molecular_dynamics",
        "optimize_structure",
        "get_base_model_path"
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
)