from google.adk.agents import  LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools.mcp_tool.mcp_session_manager import SseServerParams
from google.adk.tools.mcp_tool import MCPToolset
from typing import Literal, Optional, Dict, Any
from matcreator.tools.log import (
    create_workflow_log as _create_workflow_log,
    update_workflow_log_plan,
    read_workflow_log,
    resubmit_workflow_log,
    after_tool_log_callback
    )
from ..abacus_agent.agent import abacus_agent
from ..dpa_agent.agent import dpa_agent
from ..structure_agent.agent import structure_agent
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
    building structure → MD exploration → data curation (entropy selection) → labeling → model training → check convergence.

Before any actually calculation, you must verify with user the following critical parameters:
- General: task type (fine-tuning or distillation), max PFD iteration numbers (default 1) and convergence criteria for model training (e.g., 0.002 eV/atom)
- Structure building: crystal structure(newly built or given structure file), supercell size(s), perturbation parameters (number, cell/atom displacement magnitudes).
- MD: perturbation number, ensemble (NVT/NPT/NVE), temperature(s), total simulation time (ps), timestep/expected steps, save interval steps.
- Curation: max_sel (and chunk_size if applicable).
- For fine-tuning, verify following:
    - ABACUS labeling: kspacing (default 0.2).
- For distillation, verify following:
    - DPA labeling: head (for multi-head models, default "MP_traj_v024_alldata_mixu").
- Training: target epochs (or equivalent); training-testing data split ratio.
- Interaction mode: chat (check with user for each step) or non-interactive batch (default, proceed if no error occurs).

You have three specialized sub‑agents: 
1. 'dpa_agent_pfd': Handles MD simulation, LABELING and TRAINING with DPA model. Delegate to it for these.
2. 'abacus_agent_pfd': Handles DFT calculations using ABACUS software. Delegate to it for these.
3. 'structure_agent_pfd': Handles structure building, perturbation, and entropy-based selection. Delegate to it for these.

Workflow rules
- Create a workflow log for NEW PFD runs; show the initial plan; refine via update_workflow_log_plan until agreed. 
- Always call 'read_workflow_log' before delegating task to a sub-agent.
- After each step, summarize artifacts with absolute paths and key metrics; propose the next step.
- Repeat the cycle until reaching max iterations or convergence criteria for model training. 

Failure and resume
- If a tool fails or is unavailable, show the exact error and propose a concrete alternative. Check with the user before proceeding.
- To resume or resubmit, use resubmit_workflow_log then read_workflow_log to determine the next action.

Response format (strict)
- Plan: 1–3 bullets (why these steps).
- Action: the exact tool name you will call or the sub-agent you will delegate to.
- Result: concise outputs with absolute paths and critical metrics (e.g., frames selected, final energy).
- Next: the immediate next step or a final recap with follow‑ups.
"""

PFD_FT_INSTRUCTIONS = """
The PFD fine-tuning workflow have four major steps: 
1) Structure building: build initial structures or supercells as needed.
2) Exploration: generate new frames using molecular dynamics (MD) simulations. 
3) Data curation: select informative frames from the MD trajectory using entropy-based selection. You should verify the selection parameters such as chunk size, number of selections, k-nearest neighbors, cutoff distance, and entropy bandwidth before running the selection.
4) Data labeling: perform energy and force calculations for the selected frames, using DFT calculation (e.g. ABACUS). You should verify the DFT parameters such as pseudopotentials, basis set, k-point sampling, energy cutoff, and convergence criteria before running DFT calculations.
5) Model training: fine-tuning a machine learning force fields using the labeled data. You should verify the training parameters such as number of epochs, and validation split before running the training.
   Always fine-tuning the base model with DFT data collected in ALL iterations.  

In theory, you can run multiple iterations of the above steps to gradually improve the model performance. However, in practice, a single iteration is often sufficient to achieve good results.
Notes: you need to verify the model style, base model path, and training strategy before training. Place them in the log header if needed.
"""

PFD_DIST_INSTRUCTIONS = """
The PFD distillation workflow have following major steps:
1) Structure building: build initial structures or supercells as needed.
2) Exploration: generate new frames using molecular dynamics (MD) simulations.
3) Data curation: select informative frames from the MD trajectory using entropy-based selection. You should verify the selection parameters such as chunk size, number of selections, k-nearest neighbors, cutoff distance, and entropy bandwidth before running the selection.
4) Data labeling: perform energy and force calculations for the selected frames using ASE calculators (e.g., DPA). You should verify the model path, and any additional calculator parameters before running the calculations.
5) Model training: train a machine learning force fields from scratch (Not fine-tuning!) using the labeled data. You should verify the training parameters such as number of epochs, and validation split before running the training.

In theory, you can run multiple iterations of the above steps to gradually improve the model performance. However, in practice, a single iteration is often sufficient to achieve good results.
Notes: you need to verify the model style, base model path, and training strategy before training. Place them in the log header if needed.
"""

TASK_INSTRUCTIONS = {
    "pfd_finetune":PFD_FT_INSTRUCTIONS,
    "pfd_distillation":PFD_DIST_INSTRUCTIONS,
}
def create_workflow_log(
    workflow_name: Literal["pfd_finetune","pfd_distillation"],
    additional_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Initialize a PFD workflow log and update the LOG_PATH environment variable.
    
    Args:
        workflow_name: Name of the workflow, either "pfd_finetune" or "pfd_distillation".
        additional_info: Optional additional information to include in the log. Must be dict if provided.

    - Uses a timestamped file name to avoid clobbering.
    - Stores absolute path in env LOG_PATH for subsequent updates.
    """
    return _create_workflow_log(
        workflow_name=workflow_name,
        task_instructions=TASK_INSTRUCTIONS,
        additional_info=additional_info,
    )

allowed = {"abacus_prepare", "abacus_calculation_scf", "collect_abacus_scf_results",
           "training","run_molecular_dynamics","filter_by_entropy","perturb_atoms"}
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


toolset = MCPToolset(
    connection_params=SseServerParams(
        url="http://localhost:50003/sse", # Or any other MCP server URL
        sse_read_timeout=3600,  # Set SSE timeout to 3600 seconds
    ),
    tool_filter=[
            "abacus_prepare_batch",
            "check_abacus_inputs_batch",
            "abacus_modify_input_batch",
            "abacus_modify_stru_batch",
            "abacus_calculation_scf_batch",
            "collect_abacus_scf_results_batch"
            ],
)


abacus_agent= abacus_agent.clone(
    update={
        "name": "abacus_agent_pfd",
        "after_tool_callback": after_tool_callback,
        "disallow_transfer_to_parent": False,
        "tools":[toolset]
        },
)

dpa_agent= dpa_agent.clone(
    update={
        "name": "dpa_agent_pfd",
        "after_tool_callback": after_tool_callback,
        "disallow_transfer_to_parent": False
        },
)

structure_agent = structure_agent.clone(
    update={
        "name": "structure_agent_pfd",
        "disallow_transfer_to_parent": False,
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
        create_workflow_log,
        update_workflow_log_plan,
        read_workflow_log,
        resubmit_workflow_log,
        ],
    after_tool_callback=after_tool_callback,
    disallow_transfer_to_peers=True,
    #disallow_transfer_to_parent=True,
    sub_agents=[
        abacus_agent,
        dpa_agent,
        structure_agent
    ]
)