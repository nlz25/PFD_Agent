import os
from anyio import Path
from dotenv import load_dotenv
from google.adk.agents import LlmAgent
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import SseServerParams, StdioConnectionParams
# from dp.agent.adapter.adk import CalculationMCPToolset
from .prompt import (
    FTAGENT_NAME,
    FTAEGNTInstruction, 
    FTAEGNTDescription
    )
from ..utils.llm_config import LlmConfig
from ..tools.train.train import (list_training_strategies,
                                 train_input_doc,check_input,
                                 check_train_data,training,
                                 )
from ..tools.artifacts.artifact_file_bridge import (list_files_and_artifacts_tool,
                                                    artifact_write_tool,
                                                    file_read_artifact_tool,
                                                    get_artifact_path_tool,
                                                    cleanup_temp_artifacts_tool,
                                                    )
# Import dp module to ensure DPTrain is registered
from ..tools.train import dp

def init_ft_agent(config):
    ft_agent = LlmAgent(
        name = FTAGENT_NAME,
        model= config.gpt_4o,
        instruction = FTAEGNTInstruction,
        description = FTAEGNTDescription,
        tools=[
            # train
            list_training_strategies,
            train_input_doc,
            check_input,
            check_train_data,
            training,
            # artifacts
            list_files_and_artifacts_tool,
            file_read_artifact_tool,
            get_artifact_path_tool,
            artifact_write_tool,
            # cleanup_temp_artifacts_tool,
            ],
    )
    return ft_agent

# Example usage
root_agent = init_ft_agent(LlmConfig)