from google.adk.agents import  LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools.mcp_tool import MCPToolset
from google.adk.tools.mcp_tool.mcp_session_manager import SseServerParams
from google.genai import types
from dp.agent.adapter.adk import CalculationMCPToolset
import os
from matcreator.tools.log import after_tool_log_callback

from ..constants import LLM_MODEL, LLM_API_KEY, LLM_BASE_URL, BOHRIUM_USERNAME, BOHRIUM_PASSWORD, BOHRIUM_PROJECT_ID

# Set the secret key in ~/.abacusagent/env.json or as an environment variable, or modify the code t
model_name = os.environ.get("LLM_MODEL", LLM_MODEL)
model_api_key = os.environ.get("LLM_API_KEY", LLM_API_KEY)
model_base_url = os.environ.get("LLM_BASE_URL", LLM_BASE_URL)
bohrium_username = os.environ.get("BOHRIUM_USERNAME", BOHRIUM_USERNAME)
bohrium_password = os.environ.get("BOHRIUM_PASSWORD", BOHRIUM_PASSWORD)
bohrium_project_id = int(os.environ.get("BOHRIUM_PROJECT_ID", BOHRIUM_PROJECT_ID))

description="""
You are the ABACUS agent for executing ABACUS DFT materials calculations. You help users set up, run, and analyze ABACUS calculations.
"""

instruction ="""
Operate ABACUS safely with minimal steps and strict validation.

Must‑follow sequence
- abacus_prepare first to create an inputs directory (INPUT, STRU, pseudopotentials, orbitals). Prefer plane‑wave basis unless user requests otherwise.
- check_abacus_input to validate inputs BEFORE any calculation submission.
- Then run exactly ONE property tool per step (submission is asynchronous).
- collect_abacus_*_results AFTER the corresponding calculation completes.



Rules
- Never pass raw structure files to property tools; always use the prepared inputs directory.
- Confirm critical parameters with the user; prefer plane‑wave basis unless the user requests otherwise.
- If inputs are missing or invalid, stop and request the minimal fix.
- Never invent tools; only call from the allowlist.

Outputs
- Report absolute paths and essential metrics (e.g., final energy). Keep summaries tight and actionable.



Response format
- Plan: 1–3 bullets.
- Action: the exact ABACUS tool name you will call.
- Result: concise outputs with absolute paths.
- Next: immediate follow‑up or finish.
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

toolset = CalculationMCPToolset(
    connection_params=SseServerParams(
        url="http://localhost:50001/sse", # Or any other MCP server URL
        sse_read_timeout=3600,  # Set SSE timeout to 3600 seconds
    ),
    tool_filter=[
        "abacus_prepare",
        "check_abacus_inputs",
        "abacus_modify_input",
        "abacus_modify_stru",
        "abacus_calculation_scf",
        "collect_abacus_scf_results"
    ],
    #executor_map = EXECUTOR_MAP,
    executor=executor["local"],
    #storage=STORAGE,
)


def after_agent_cb2(callback_context):
  print('@after_agent_cb2')
  # ModelContent (or Content with role set to 'model') must be returned.
  # Otherwise, the event will be excluded from the context in the next turn.
  return types.ModelContent(
      parts=[
          types.Part(
              text='Handoff: return_to_parent',
          ),
      ],
  )


abacus_agent = LlmAgent(
    name='abacus_agent',
    model=LiteLlm(
        model=model_name,
        base_url=model_base_url,
        api_key=model_api_key
    ),
    description=description,
    instruction=instruction,
    tools=[toolset],
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
    #after_tool_callback=after_tool_callback,
    #after_agent_callback=after_agent_cb2,
    
)