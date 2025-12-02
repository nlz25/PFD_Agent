from google.adk.agents import  LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools.mcp_tool import MCPToolset
from google.adk.tools.mcp_tool.mcp_session_manager import SseServerParams
from matcreator.tools.log import (
    create_workflow_log,
    update_workflow_log_plan,
    read_workflow_log,
    resubmit_workflow_log,
    after_tool_log_callback
    )
from ..abacus_agent.agent import abacus_agent
from ..dpa_agent.agent import dpa_agent
import os
from ..constants import LLM_MODEL, LLM_API_KEY, LLM_BASE_URL, BOHRIUM_USERNAME, BOHRIUM_PASSWORD, BOHRIUM_PROJECT_ID

model_name = os.environ.get("LLM_MODEL", LLM_MODEL)
model_api_key = os.environ.get("LLM_API_KEY", LLM_API_KEY)
model_base_url = os.environ.get("LLM_BASE_URL", LLM_BASE_URL)
bohrium_username = os.environ.get("BOHRIUM_USERNAME", BOHRIUM_USERNAME)
bohrium_password = os.environ.get("BOHRIUM_PASSWORD", BOHRIUM_PASSWORD)
bohrium_project_id = int(os.environ.get("BOHRIUM_PROJECT_ID", BOHRIUM_PROJECT_ID))

description="""
The main coordinator agent for PFD (pretrain-finetuning-distillation) workflow. Handles PFD workflow and delegates DPA/ABACUS tasks to specialized sub-agents.
"""

instruction ="""
Mission
- Orchestrate the standard PFD workflow with minimal, safe steps and clear outputs:
    MD exploration → data curation (entropy selection) → labeling → model training.

Before any actually calculation, you must verify with user the following critical parameters:
- General: task type (fine-tuning or distillation), max PFD iteration numbers (default 1) and convergence criteria for model training (e.g., 0.002 eV/atom)
- MD: ensemble (NVT/NPT/NVE), temperature(s), total simulation time (ps), timestep/expected steps, save interval steps.
- Curation: max_sel (and chunk_size if applicable).
- For fine-tuning, verify following:
    - ABACUS labeling: kspacing (default 0.14).
- For distillation, verify following:
    - DPA labeling: head (for multi-head models, default "MP_traj_v024_alldata_mixu").
- Training: target epochs (or equivalent); training-testing data split ratio.
- Interaction mode: chat (check with user for each step) or non-interactive batch (default, proceed if no error occurs).


You have two specialized sub‑agents: 
1. 'dpa_agent_pfd': Handles MD simulation, LABELING and TRAINING with DPA model. Delegate to it for these.
2. 'abacus_agent_pfd': Handles DFT calculations using ABACUS software. Delegate to it for these.

Never invent tools!

Workflow rules
- Create a workflow log for NEW PFD runs; show the initial plan; refine via update_workflow_log_plan until agreed. 
- Read the workflow log via 'read_workflow_log' everytime before delegating to sub-agents.
- In each step, either delegate to one sub-agent or execute a tool in this agent; do not mix.
- After each step, summarize artifacts with absolute paths and key metrics; propose the next step.
- Repeat the PFD cycle until reaching max iterations or convergence criteria for model training.

Failure and resume
- If a tool fails or is unavailable, show the exact error and propose a concrete alternative. Check with the user before proceeding.
- To resume or resubmit, use resubmit_workflow_log then read_workflow_log to determine the next action.

Response format (strict)
- Plan: 1–3 bullets (why these steps).
- Action: the exact tool name you will call or the sub-agent you will delegate to.
- Result: concise outputs with absolute paths and critical metrics (e.g., frames selected, final energy).
- Next: the immediate next step or a final recap with follow‑ups.
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


# entropy filter toolset
selector_toolset = MCPToolset(
    connection_params=SseServerParams(
        url="http://localhost:50004/sse", # Or any other MCP server URL
        sse_read_timeout=3600,  # Set SSE timeout to 3600 seconds
    ),
    tool_filter=[
        "filter_by_entropy",
    ],
)

allowed = {"abacus_prepare", "abacus_calculation_scf", "collect_abacus_scf_results",
           "training","run_molecular_dynamics","filter_by_entropy"}
name_map={
    "abacus_prepare":"labeling_abacus_scf_preparation",
    "abacus_calculation_scf":"labeling_abacus_scf_calculation",  
    "collect_abacus_scf_results":"labeling_abacus_scf_collect_results",
    "run_molecular_dynamics":"exploration_md",
    "filter_by_entropy":"explore_filter_by_entropy"
}

def after_tool_callback(tool,args,tool_context,tool_response):
    if getattr(tool, 'name', None) in allowed:
        return after_tool_log_callback(
            tool, args, tool_context, tool_response,step_name_map=name_map
        )


abacus_agent= abacus_agent.clone(
    update={
        "name": "abacus_agent_pfd",
        "after_tool_callback": after_tool_callback
        },
)

dpa_agent= dpa_agent.clone(
    update={
        "name": "dpa_agent_pfd",
        "after_tool_callback": after_tool_callback
        },
)

pfd_agent = LlmAgent(
    name='pfd_agent',
    model=LiteLlm(
        model=model_name,
        base_url=model_base_url,
        api_key=model_api_key
    ),
    description=description,
    instruction=instruction,
    tools=[
        selector_toolset,
        create_workflow_log,
        update_workflow_log_plan,
        read_workflow_log,
        resubmit_workflow_log,
        ],
    after_tool_callback=after_tool_callback,
    sub_agents=[
        abacus_agent,
        dpa_agent,
    ]
)