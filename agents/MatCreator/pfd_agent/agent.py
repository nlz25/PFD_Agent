from google.adk.agents import  LlmAgent,Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools.mcp_tool import MCPToolset
from google.adk.tools.mcp_tool.mcp_session_manager import SseServerParams
from google.adk.tools import agent_tool
from matcreator.tools.log import (
    create_workflow_log,
    update_workflow_log_plan,
    read_workflow_log,
    resubmit_workflow_log,
    after_tool_log_callback
    )
#from .abacus_agent.agent import abacus_agent
from ..abacus_agent.agent import abacus_agent
from ..dpa_agent.agent import dpa_agent
#from .dpa_agent.agent import dpa_agent
#from .structure_agent.agent import structure_agent
import os, json

# Set the secret key in ~/.abacusagent/env.json or as an environment variable, or modify the code to set it directly.
env_file = os.path.expanduser("~/.pfd_agent/env.json")
if os.path.isfile(env_file):
    env = json.load(open(env_file, "r"))
else:
    env = {}
model_name = env.get("LLM_MODEL", os.environ.get("LLM_MODEL", ""))
model_api_key = env.get("LLM_API_KEY", os.environ.get("LLM_API_KEY", ""))
model_base_url = env.get("LLM_BASE_URL", os.environ.get("LLM_BASE_URL", ""))

description="""
The main coordinator agent for PFD (pretrain-finetuning-distillation) workflow. Handles PFD workflow and delegates DPA/ABACUS tasks to specialized sub-agents.
"""

instruction ="""
Mission
- Orchestrate the standard PFD workflow with minimal, safe steps and clear outputs:
    MD exploration → data curation (entropy selection) → DFT labeling (ABACUS) → model training (DPA).

Before any actually calculation, you must verify with user the following critical parameters:
- MD: ensemble (NVT/NPT/NVE), temperature(s), total simulation time (ps), timestep/expected steps.
- Curation: max_sel (and chunk_size if applicable).
- Training: target epochs (or equivalent); propose a short validation run if long.

You have two specialized sub‑agents: 
1. 'dpa_agent_pfd': Handles MD simulation and TRAINING with DPA model. Delegate to it for these.
2. 'abacus_agent_pfd': Handles DFT calculations using ABACUS software. Delegate to it for these.

Never invent tools
- Only call tools from the allowlist above. Do not fabricate tool or agent names.

Workflow rules
- Create a workflow log for NEW PFD runs; show the initial plan; refine via update_workflow_log_plan until agreed. Read the log before delegating to sub-agents.
- In each step, either delegate to one sub-agent or execute a tool in this agent; do not mix.
- After each step, summarize artifacts with absolute paths and key metrics; propose the next step.

Failure and resume
- If a tool fails or is unavailable, show the exact error and propose a concrete alternative.
- To resume or resubmit, use resubmit_workflow_log then read_workflow_log to determine the next action.

Response format (strict)
- Plan: 1–3 bullets (why these steps).
- Action: the exact tool name you will call or the sub-agent you will delegate to.
- Result: concise outputs with absolute paths and critical metrics (e.g., frames selected, final energy).
- Next: the immediate next step or a final recap with follow‑ups.
"""


# entropy filter toolset
selector_toolset = MCPToolset(
    connection_params=SseServerParams(
        url="http://localhost:50003/sse", # Or any other MCP server URL
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
        #structure_agent
    ]
)