import random
import logging
from pathlib import Path
from typing import (
    Optional, 
    TypedDict, 
    List, 
    Dict, 
    Union,
    Any
)
from matcreator.tools.util.common import generate_work_path
from .dp import DPTrain
import os
from ase.io import read, write
from jsonschema import validate, ValidationError


class TrainInputDocResult(TypedDict):
    """Input format structure for training strategies"""
    name: str
    description: str
    config: str
    command: str

#@mcp.tool()
def train_input_doc() -> Dict[str, Any]:
    """
    Returns:
        List available training strategies and their metadata. 
        You can use these information to formulate template config and command dict.
    """
    try:
        training_meta = DPTrain.training_meta()
        return TrainInputDocResult(
            name="Deep Potential model",
            description=str(training_meta.get("description", "")),
            config=str(training_meta.get("config", {}).get("doc", "")),
            command=str(training_meta.get("command", {}).get("doc", "")),
        )
    except Exception as e:
        logging.exception("Failed to get training strategy doc")
        return TrainInputDocResult(
            name="",
            description="",
            config="",
            command="",
        )

class CheckTrainDataResult(TypedDict):
    """Result structure for get_training_data (with optional split)"""
    train_data: Path
    valid_data: Optional[Path]
    test_data: Optional[Path]
    num_frames: int
    ave_atoms: int
    num_train_frames: int
    num_valid_frames: int
    num_test_frames: int
    
#@mcp.tool()
def check_train_data(
    train_data: Path,
    valid_ratio: Optional[float] = 0.0,
    test_ratio: Optional[float] = 0.0,
    shuffle: bool = True,
    seed: Optional[int] = None,
    output_dir: Optional[Path] = None,
):
    """Inspect training data and optionally produce a train/valid/test split.

    Args:
        train_data: Path to a multi-frame structure file readable by ASE (e.g., extxyz).
        valid_ratio: Fraction in [0,1] for validation split size.
        test_ratio: Fraction in [0,1] for test split size.
        shuffle: Whether to shuffle before splitting.
        seed: Optional RNG seed for reproducible shuffling.
        output_dir: Where to write split files; defaults to train_data.parent.

    Returns:
        - train_data: Path to the (possibly new) training file when split is performed; otherwise the input.
        - valid_data: Path to the generated validation file when split is performed; otherwise None.
        - test_data: Path to the generated test file when split is performed; otherwise None.
        - num_frames: Total frames in the input dataset.
        - ave_atoms: Integer average atoms per frame.
        - num_train_frames: Frames in training split (or total when no split).
        - num_valid_frames: Frames in validation split (0 when no split).
        - num_test_frames: Frames in test split (0 when no split).
    """
    try:
        src_path = Path(train_data).resolve()
        frames = read(str(src_path), index=':')
        num_frames = len(frames)
        ave_atoms = sum(len(atoms) for atoms in frames) // num_frames if num_frames > 0 else 0

        out_train_path: Path = src_path
        out_valid_path: Optional[Path] = None
        out_test_path: Optional[Path] = None
        num_train_frames = num_frames
        num_valid_frames = 0
        num_test_frames = 0

        # Perform split if requested and possible
        wants_split = (valid_ratio or 0) > 0 or (test_ratio or 0) > 0
        if wants_split and num_frames > 2:
            rv = max(0.0, min(1.0, float(valid_ratio or 0.0)))
            rt = max(0.0, min(1.0, float(test_ratio or 0.0)))
            # initial counts
            n_valid = int(round(num_frames * rv))
            n_test = int(round(num_frames * rt))
            # ensure at least one train frame where possible
            if n_valid + n_test >= num_frames and num_frames > 2:
                # Reduce the larger split first
                if n_valid >= n_test and n_valid > 0:
                    n_valid = max(0, num_frames - 1 - n_test)
                elif n_test > 0:
                    n_test = max(0, num_frames - 1 - n_valid)
            # recompute if still too large
            if n_valid + n_test >= num_frames and num_frames > 2:
                # fallback: put everything not train into valid, no test
                n_test = 0
                n_valid = max(0, num_frames - 1)

            idx = list(range(num_frames))
            if shuffle:
                rng = random.Random(seed)
                rng.shuffle(idx)
            valid_idx = set(idx[:n_valid])
            test_idx = set(idx[n_valid:n_valid + n_test])
            train_idx = [i for i in idx if i not in valid_idx and i not in test_idx]

            valid_frames = [frames[i] for i in sorted(valid_idx)] if n_valid > 0 else []
            test_frames = [frames[i] for i in sorted(test_idx)] if n_test > 0 else []
            train_frames = [frames[i] for i in train_idx]

            out_dir = Path(output_dir).resolve() if output_dir else src_path.parent
            out_dir.mkdir(parents=True, exist_ok=True)
            stem = src_path.stem
            out_train_path = out_dir / f"{stem}_train.extxyz"
            if n_valid > 0:
                out_valid_path = out_dir / f"{stem}_valid.extxyz"
            if n_test > 0:
                out_test_path = out_dir / f"{stem}_test.extxyz"

            # Write splits as extxyz
            write(str(out_train_path), train_frames, format='extxyz')
            if n_valid > 0:
                write(str(out_valid_path), valid_frames, format='extxyz')
            if n_test > 0:
                write(str(out_test_path), test_frames, format='extxyz')

            num_train_frames = len(train_frames)
            num_valid_frames = len(valid_frames)
            num_test_frames = len(test_frames)

        return CheckTrainDataResult(
            train_data=out_train_path,
            valid_data=out_valid_path,
            test_data=out_test_path,
            num_frames=num_frames,
            ave_atoms=ave_atoms,
            num_train_frames=num_train_frames,
            num_valid_frames=num_valid_frames,
            num_test_frames=num_test_frames,
        )
    except Exception as e:
        logging.exception("Failed to get training data")
        return CheckTrainDataResult(
            train_data=Path(""),
            valid_data=None,
            test_data=None,
            num_frames=0,
            ave_atoms=0,
            num_train_frames=0,
            num_valid_frames=0,
            num_test_frames=0,
        )


class CheckInputResult(TypedDict):
    """Result structure for check_input"""
    valid: bool
    message: str
    command: Dict[str, Any]
    config: Dict[str, Any]
#@mcp.tool()
def check_input(
    config: Dict[str, Any], #= load_json_file(CONFIG_PATH),
    command: Optional[Dict[str, Any]] = {},#load_json_file(COMMAND_PATH),
) -> CheckInputResult:
    """You should validate the `config` and `command` input based on the selected strategy.
        You need to ensure that all required fields are present and correctly formatted.
        If any required field is missing or incorrectly formatted, return a message indicating the issue.
        Make sure to pass this validation step before proceeding to training.
    """
    try:
        training_meta = DPTrain.training_meta()
        validate(config, training_meta["config"]["schema"])
        config=DPTrain.normalize_config(config)
        validate(command, training_meta["command"]["schema"])
        command=DPTrain.normalize_command(command)
        return CheckInputResult(
            valid=True,
            message="Config is valid",
            command=command,
            config=config
        )
    except ValidationError as e:
        logging.exception("Config validation failed")
        return CheckInputResult(
            valid=False,
            message=f"Config validation failed: {e.message}",
            command=command,
            config=config
        )

class TrainingResult(TypedDict):
    """Result structure for model training"""
    model: Path
    log: Path
    message: str
    test_metrics: Optional[List[Dict[str, Any]]]

def training(
    config: Dict[str, Any], #= load_json_file(CONFIG_PATH),
    train_data: Path,# = Path(TRAIN_DATA_PATH),
    model_path: Optional[Path] = None,
    command: Optional[Dict[str, Any]] = {},#load_json_file(COMMAND_PATH),
    valid_data: Optional[Union[List[Path], Path]] = None,
    test_data: Optional[Union[List[Path], Path]] = None,
) -> TrainingResult:
    """Train a Deep Potential (DP) machine learning force field model. This tool should only be executed once all necessary inputs are gathered and validated.
       Always use 'train_input_doc' to get the template for 'config' and 'command', and use 'check_input' to validate them before calling this tool.
    
    Args:
        config: Configuration parameters for training (You can find an example for `config` from the 'train_input_doc' tool').
        command: Command parameters for training (You can find an example for `command` from the 'train_input_doc' tool').
        train_data: Path to the training dataset (required).
        model_path (Path, optional): Path to pre-trained base model. Required for model fine-tuning.
    
    """
    try:
        runner = DPTrain(config=config, train_data=train_data, command=command, model_path=model_path,
                 valid_data=valid_data, test_data=test_data)
        runner.validate()
        #work_path=Path(generate_work_path()).absolute()
        #work_path.mkdir(parents=True, exist_ok=True)
        #cwd = os.getcwd()
        # change to workdir
        #os.chdir(work_path)
        model, log, message = runner.run()
        logging.info("Training completed!")
        test_metrics = None
        if test_data:
            _, test_metrics = runner.test()
        #os.chdir(cwd)
        result ={
            "status":"success",
            "model": str(model.resolve()),
            "log": str(log.resolve()),
            "message": message,
            "test_metrics": test_metrics
                }
    
    except Exception as e:
        logging.exception("Training failed")
        
        result={
            "status":"error",
            "model": None,
            "log": None,
            "message": f"Training failed: {str(e)}",
        }
    return result