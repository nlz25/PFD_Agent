from typing import override, AsyncGenerator
from google.adk.agents import Agent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.genai import types
# from opik.integrations.adk import track_adk_agent_recursive

from .callback import init_before_agent_callback, before_model_callback, after_model_callback, \
    enforce_single_tool_call
from agents.pfd_agent.prompt import *
from agents.pfd_agent.utils.llm_config import LlmConfig


class PFDAgent(Agent):
    def __init__(self, llm_config):
        prepare_state_before_agent = init_before_agent_callback(llm_config)
        # ft_agent = init_ml_agent(llm_config)
        # ...
        
        super().__init__(name="pfd_agent",
                         model=llm_config.deepseek_chat,
                         description="",
                         sub_agents=[
                             # ft_agent,
                             # ..._agent,
                         ],
                         # disallow_transfer_to_peers=True,
                         global_instruction=GlobalInstruction,
                         instruction=AgentInstruction,
                         description=AgentDescription,
                         before_agent_callback=prepare_state_before_agent,
                         before_model_callback=[before_model_callback],
                         after_model_callback=[enforce_single_tool_call, after_model_callback])

    @override
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        prompt = ""
        if ctx.user_content and ctx.user_content.parts:
            for part in ctx.user_content.parts:
                if part.text:
                    prompt += part.text
                elif part.inline_data:
                    pass
                elif part.file_data:
                    prompt += f", file_url = {part.file_data.file_uri}"
                    yield Event(
                        invocation_id=ctx.invocation_id,
                        author=self.name,
                        branch=ctx.branch,
                        content=types.Content(parts=[types.Part(text=prompt)], role="system"))

        async for event in super()._run_async_impl(ctx):
            yield event


def init_pfd_agent(llm_config):
    pfd_agent = PFDAgent(llm_config)
    # track_adk_agent_recursive(pfd_agent, llm_config.opik_tracer)
    return pfd_agent

# Example usage
root_agent = init_pfd_agent(LlmConfig)