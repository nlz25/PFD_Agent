import logging
from pathlib import Path
from typing import (
    Optional, 
    List, 
    Dict, 
    Union,
    Any
)
from abc import ABC, abstractmethod

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
        if model_path is not None:
            self.model_path = model_path.resolve()
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


    @abstractmethod
    def test(self) -> None:
        """Run evaluation on the test dataset if provided.
        Should populate self.metrics with test results.
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