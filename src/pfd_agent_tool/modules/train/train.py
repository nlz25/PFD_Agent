import glob
import logging
import os
import sys
from pathlib import Path
from typing import (
    Literal, 
    Optional, 
    Tuple, 
    TypedDict, 
    List, 
    Dict, 
    Union,
    Any
)
import sys
from abc import ABC, abstractmethod
from pfd_agent_tool.init_mcp import mcp
import json

from src.pfd_agent_tool.modules.train.prompt import TRAINING_STRATEGIES

class TrainingResult(TypedDict):
    """Result structure for model training"""
    model: Path
    log: Path
    message: str


class Train(ABC):
    """Abstract base class encapsulating a training pipeline.

    Subclasses must implement the lifecycle hooks to:
      1. prepare()  -> validate inputs, set up working dirs
      2. run()      -> execute the actual training loop / external command
      3. finalize() -> collect artifacts & metrics and return TrainingResult

    The template method execute() wires these steps together and handles
    logging + basic error capture.
    """

    __ModelTypes: Dict[str, Any] = {}
    
    def __init__(self, config: Dict[str, Any], train_data: Union[List[Path], Path], command: Optional[Dict[str, Any]] = None,
                 model_path: Optional[Path] = None, valid_data: Optional[Union[List[Path], Path]] = None,
                 test_data: Optional[Union[List[Path], Path]] = None,
                 optional_files: Optional[Union[List[Path], Path]] = None
                 ) -> None:
        self.config = config
        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = test_data
        self.model_path = model_path
        self.command = command or {}
        self.optional_files = optional_files
        self.workdir = Path(self.config.get("workdir", "./train_workspace")).absolute()
        self.artifacts: Dict[str, Path] = {}
        self.metrics: Dict[str, Any] = {}

    @classmethod
    @abstractmethod
    def training_meta(cls) -> Dict[str, Any]:
        """Return metadata about the training process."""
        raise NotImplementedError

    # ---- Required hooks -------------------------------------------------

    @abstractmethod
    def run(self) -> None:
        """Run the core training / optimization procedure.
        Should populate self.metrics and any intermediate artifacts.
        """
        raise NotImplementedError

    # ---- Optional overridable hooks ------------------------------------
    def validate(self) -> None:
        """Lightweight validation of high-level arguments."""
        if not self.train_data:
            raise ValueError("train_data is required")

    def setup_logging(self) -> None:
        self.workdir.mkdir(parents=True, exist_ok=True)
        logging.info(f"[RunTrain] Workdir: {self.workdir}")


    @staticmethod
    def register(key: str):
        """Register a model type. Used as decorators

        Args:
            key (str): key of the model
        """

        def decorator(object):
            Train.__ModelTypes[key] = object
            return object

        return decorator

    @staticmethod
    def get_driver(key: str):
        """Get a driver for ModelEval

        Args:
            key (str): _description_

        Raises:
            RuntimeError: _description_

        Returns:
            _type_: _description_
        """
        try:
            return Train.__ModelTypes[key]
        except KeyError as e:
            raise RuntimeError("unknown driver: " + key) from e

    @staticmethod
    def get_drivers() -> dict:
        """Get all drivers

        Returns:
            dict: all drivers
        """
        return Train.__ModelTypes

@mcp.tool()
def list_and_describe_training_strategies() -> Dict[str, Any]:
    """
    Returns:
        List available training strategies and their metadata.
    """
    return {
        "strategies": [
            {"name": name, **Train.get_driver(name).training_meta()}
            for name in Train.get_drivers()
        ]
    }


def load_json_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"文件不存在: {file_path}")
        return {}
    except json.JSONDecodeError as e:
        print(f"JSON 格式错误: {e}")
        return {}
    except Exception as e:
        print(f"读取文件失败: {e}")
        return {}

@mcp.tool()
def training(
    config: Dict[str, Any] = load_json_file(CONFIG_PATH),
    train_data: Union[List[Path], Path] = Path(TRAIN_DATA_PATH),
    command: Optional[Dict[str, Any]] = load_json_file(COMMAND_PATH),
    model_path: Optional[Path] = Path(MODEL_PATH),
    valid_data: Optional[Union[List[Path], Path]] = None,
    test_data: Optional[Union[List[Path], Path]] = None,
    strategy: str = "dpa",
) -> TrainingResult:
    """Train a selected machine learning force field model via a chosen strategy."""
    try:
        cls = Train.get_driver(strategy)
        runner = cls(config=config, train_data=train_data, command=command, model_path=model_path,
                 valid_data=valid_data, test_data=test_data)
        runner.validate()
        model, log, message = runner.run()
        print("Training completed!")
        return TrainingResult(model=model, log=log, message=message)
    except Exception as e:
        logging.exception("Training failed")
        return TrainingResult(model=Path(""), log=Path(""), message=f"Training failed: {e}")