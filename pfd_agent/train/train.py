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
import argparse
from abc import ABC, abstractmethod

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
