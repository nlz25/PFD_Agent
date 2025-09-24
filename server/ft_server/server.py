import glob
import logging
import os
import sys
from pathlib import Path
from typing import (
    Optional, 
    TypedDict, 
    List, 
    Dict, 
    Union,
    Any
)
import sys
import argparse
from dp.agent.server import CalculationMCPServer
from pfd_agent.train import (
    Train,
    DPTrain)

### CONSTANTS
# Strategy registry (can later include metadata)
TRAINING_STRATEGIES: Dict[str, type[Train]] = {
    "dpa": DPTrain,
}

STRATEGY_META: Dict[str, Dict[str, Any]] = {
    "dpa": {
        "version": "1.0",
        "description": "Training routine for DPA model.",
        "required_command_keys": ["epochs"],
        "optional_command_keys": ["workdir"],
    }
}

def parse_args():
    """Parse command line arguments for MCP server."""
    parser = argparse.ArgumentParser(description="DPA Calculator MCP Server")
    parser.add_argument('--port', type=int, default=50001, help='Server port (default: 50001)')
    parser.add_argument('--host', default='0.0.0.0', help='Server host (default: 0.0.0.0)')
    parser.add_argument('--log-level', default='INFO', 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level (default: INFO)')
    try:
        args = parser.parse_args()
    except SystemExit:
        class Args:
            port = 50001
            host = '0.0.0.0'
            log_level = 'INFO'
        args = Args()
    return args


args = parse_args()
# a wrapper for FastMCP server
mcp = CalculationMCPServer("ModelTrainingServer", host=args.host, port=args.port)


class TrainingResult(TypedDict):
    """Result structure for model training"""
    model: Path
    log: Path
    message: str



@mcp.tool()
def list_training_strategies() -> Dict[str, Any]:
    """List available training strategies and their metadata."""
    return {
        "strategies": [
            {"name": name, **STRATEGY_META.get(name, {})}
            for name in TRAINING_STRATEGIES
        ]
    }

@mcp.tool()
def describe_training_strategy(strategy: str) -> Dict[str, Any]:
    """Describe a specific strategy."""
    if strategy not in TRAINING_STRATEGIES:
        raise ValueError(f"Unknown strategy '{strategy}'")
    return {"name": strategy, **STRATEGY_META.get(strategy, {})}

@mcp.tool()
def training(
    config: Dict[str, Any],
    train_data: Union[List[Path], Path],
    command: Optional[Dict[str, Any]] = None,
    model_path: Optional[Path] = None,
    valid_data: Optional[Union[List[Path], Path]] = None,
    test_data: Optional[Union[List[Path], Path]] = None,
    strategy: str = "dpa",
) -> TrainingResult:
    """Train a selected machine learning force field model via a chosen strategy.

    strategy: selects which RunTrain subclass to use (default: 'basic').
    """
    cls = TRAINING_STRATEGIES.get(strategy)
    if cls is None:
        raise ValueError(f"Unknown training strategy '{strategy}'. Available: {list(TRAINING_STRATEGIES)}")
    try:
        runner = cls(config=config, train_data=train_data, command=command, model_path=model_path,
                 valid_data=valid_data, test_data=test_data)
        runner.validate()
        model, log, message = runner.run()
        print("Training completed!")
        return TrainingResult(model=model, log=log, message=message)
    except Exception as e:
        logging.exception("Training failed")
        return TrainingResult(model=Path(""), log=Path(""), message=f"Training failed: {e}")
    

if __name__ == "__main__":
    logging.info("Starting Unified MCP Server with all tools...")
    # Get transport type from environment variable, default to SSE
    #transport_type = os.getenv('MCP_TRANSPORT', 'sse')
    transport_type = os.getenv('MCP_TRANSPORT', 'stdio')
    mcp.run(transport=transport_type)