from typing import AsyncGenerator
from google.adk.agents import Agent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.genai import types
# from opik.integrations.adk import track_adk_agent_recursive

from .callback import init_before_agent, before_model_callback, after_model_callback, \
    enforce_single_tool_call
from .prompt import (
    GlobalInstruction,
    PFDAgentDescription,
    PFDAgentInstruction,
)
from .utils.llm_config import LlmConfig
from .ft_agent.agent import init_ft_agent
from .db_agent.agent import init_db_agent

class PFDAgent(Agent):
    def __init__(self, llm_config):
        prepare_state_before_agent = init_before_agent(llm_config)
        ft_agent = init_ft_agent(llm_config)
        db_agent = init_db_agent(llm_config)
        # ...
        super().__init__(name="pfd_agent",
                         model=llm_config.ali,
                         sub_agents=[
                             ft_agent,
                             db_agent,
                             # ..._agent,
                         ],
                         # disallow_transfer_to_peers=True,
                         global_instruction = GlobalInstruction,
                         instruction = PFDAgentInstruction,
                         description = PFDAgentDescription,
                         # before_agent_callback=prepare_state_before_agent,
                         # before_model_callback=[before_model_callback],
                         # after_model_callback=[enforce_single_tool_call, after_model_callback],
                         )

def init_pfd_agent(llm_config):
    pfd_agent = PFDAgent(llm_config)
    # track_adk_agent_recursive(pfd_agent, llm_config.opik_tracer)
    return pfd_agent

# Example usage
root_agent = init_pfd_agent(LlmConfig)