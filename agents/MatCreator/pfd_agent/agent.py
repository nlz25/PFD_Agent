from google.adk.agents import  LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools.mcp_tool.mcp_session_manager import SseServerParams
from google.adk.tools.mcp_tool import McpToolset
from dp.agent.adapter.adk import CalculationMCPToolset

from ..abacus_agent.agent import abacus_agent
from ..dpa_agent.agent import dpa_agent
from ..vasp_agent.agent import vasp_agent
from ..structure_agent.agent import structure_agent
from ..callbacks import (
    before_agent_callback,
    after_tool_callback, #as _after_tool_callback,
    set_session_metadata,
    get_session_context
)
import os
from ..constants import LLM_MODEL, LLM_API_KEY, LLM_BASE_URL
from dotenv import load_dotenv
from pathlib import Path
_script_dir = Path(__file__).parent
load_dotenv(_script_dir / ".env", override=True)

model_name = os.environ.get("LLM_MODEL", LLM_MODEL)
model_api_key = os.environ.get("LLM_API_KEY", LLM_API_KEY)
model_base_url = os.environ.get("LLM_BASE_URL", LLM_BASE_URL)


description="""
The main coordinator agent for PFD (pretrain-finetuning-distillation) workflow. Handles PFD workflow and delegates DPA/ABACUS/VASP tasks to specialized sub-agents.
"""

instruction ="""
Mission
- Orchestrate the standard PFD workflow with minimal, safe steps and clear outputs:
    building structure → MD exploration → data curation (entropy selection) → labeling → model training → check convergence.

Session tracking
- At the start of a new PFD workflow, use 'set_session_metadata' to record workflow type, goals, and other session info.
- Use 'get_session_context' to retrieve complete session history including all tool executions grouped by workflow step.
- The system automatically logs ALL tool executions to a database for persistent session tracking.
- Review session context when resuming work to understand what has already been completed.
- Periodically use 'get_session_context' during workflow execution, especially after sub-agent completions of major steps, to maintain awareness of accomplished work and make informed decisions about next actions.

Before any calculation, verify with user the following critical parameters:
- General: task type (fine-tuning or distillation), max PFD iteration numbers (default 1) and convergence criteria (e.g., 0.002 eV/atom)
- Structure building: crystal structure (newly built or given file), supercell size(s), perturbation parameters (number, cell/atom displacement magnitudes)
- MD: perturbation number, ensemble (NVT/NPT/NVE), temperature(s), total simulation time (ps), timestep/expected steps, save interval steps
- Curation: max_sel (and chunk_size if applicable)
- For fine-tuning: ABACUS labeling parameters (kspacing default 0.2)
- For distillation: DPA labeling parameters (head for multi-head models, default "MP_traj_v024_alldata_mixu")
- Training: target epochs, training-testing data split ratio
- Interaction mode: chat (check with user for each step) or non-interactive batch (default, proceed if no error)

You have three specialized sub‑agents:
1. 'dpa_agent_pfd': MD simulation, labeling with DPA, and model training
2. 'abacus_agent_pfd': DFT calculations using ABACUS
3. 'structure_agent_pfd': Structure building, perturbation, and entropy-based selection

Workflow rules
- For NEW workflows: set session metadata with workflow type and goals
- After each step: summarize artifacts with absolute paths and key metrics; propose the next step
- Check session context before resuming to avoid redoing completed work
- Repeat cycles until reaching max iterations or convergence criteria

PFD Fine-tuning workflow:
1) Structure building → 2) MD exploration → 3) Data curation (entropy) → 4) DFT labeling (ABACUS) → 5) Fine-tune model with ALL collected data

PFD Distillation workflow:
1) Structure building → 2) MD exploration → 3) Data curation (entropy) → 4) Teacher model labeling (DPA) → 5) Train new model from scratch

Failure handling
- If a tool fails, show exact error and propose concrete alternative
- Check with user before proceeding after errors
- Use 'get_session_context' to review completed steps when resuming

Response format
- Plan: 1–3 bullets (why these steps)
- Action: exact tool name or sub-agent to delegate to
- Result: concise outputs with absolute paths and key metrics
- Next: immediate next step or final recap
"""


toolset = McpToolset(
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

vasp_agent= vasp_agent.clone(
    update={
        "name": "vasp_agent_pfd",
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
        set_session_metadata,
        get_session_context,
        ],
    before_agent_callback=before_agent_callback,
    after_tool_callback=after_tool_callback,
    disallow_transfer_to_peers=True,
    sub_agents=[
        abacus_agent,
        dpa_agent,
        vasp_agent,
        structure_agent
    ]
)