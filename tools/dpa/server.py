import argparse
import os
import time
import logging
import random
import traceback
import uuid
from typing import Optional, Union, Dict, Any, List, Tuple, TypedDict,Literal
from pathlib import Path
from jsonschema import validate, ValidationError
from dotenv import load_dotenv

# ASE / MD imports used by DPA tools
from ase.io import read, write
from ase.optimize import BFGS
from ase.constraints import ExpCellFilter
from deepmd.calculator import DP
from dpa_tool.train import (
    dpa_training_meta,
    normalize_dpa_command,
    normalize_dpa_config,
    _run_dp_training,
    _evaluate_trained_model,
    _ensure_path_list,
    model_inference as _model_inference,
)
from dpa_tool.ase_md import optimize_structure as _optimize_structure
from dpa_tool.utils import set_directory

_script_dir = Path(__file__).parent
load_dotenv(_script_dir / ".env", override=True)

DPA_MODEL_PATH = "./DPA2_medium_28_10M_rc0.pt"
DPA_SERVER_WORK_PATH = "/tmp/dpa_server"

default_dpa_model_path= os.environ.get("DPA_MODEL_PATH", DPA_MODEL_PATH)


def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="PFD_Agent Command Line Interface")
    
    parser.add_argument(
        "--transport",
        type=str,
        default="sse",
        choices=["sse", "streamable-http"],
        help="Transport protocol to use (default: sse), choices: sse, streamable-http"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="fastmcp",
        choices=["fastmcp", "dp"],
        help="Model to use (default: dp), choices: fastmcp, dp"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=50001,
        help="Port to run the MCP server on (default: 50001)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Host to run the MCP server on (default: localhost)"
    )
    args = parser.parse_args()
    return args


args = parse_args()  


if args.model == "dp":
    from dp.agent.server import CalculationMCPServer
    mcp = CalculationMCPServer(
            "AbacusServer",
            host=args.host,
            port=args.port
        )
elif args.model == "fastmcp":
    from mcp.server.fastmcp import FastMCP
    mcp = FastMCP(
            "AbacusServer",
            host=args.host,
            port=args.port
        )

## =========================
## Utility functions
## =========================


def generate_work_path(create: bool = True) -> str:
	"""Return a unique work dir path and create it by default."""
	calling_function = traceback.extract_stack(limit=2)[-2].name
	current_time = time.strftime("%Y%m%d%H%M%S")
	random_string = str(uuid.uuid4())[:8]
	work_path = f"{current_time}.{calling_function}.{random_string}"
	if create:
		os.makedirs(work_path, exist_ok=True)
	return work_path

## =========================
## DPA trainer tool implementations
## =========================

class TrainInputDocResult(TypedDict):
    """Input format structure for training strategies"""
    name: str
    description: str
    config: str
    command: str

@mcp.tool()
def train_input_doc() -> Dict[str, Any]:
    """
    Check the training input document for Deep Potential model training.
    Returns:
        List metadata for training a Deep Potential model. 
        You can use these information to formulate template 'config' and 'command' dict.
    """
    try:
        training_meta = dpa_training_meta()
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
    
@mcp.tool()
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

@mcp.tool()
def check_input(
    config: Dict[str, Any], #= load_json_file(CONFIG_PATH),
    command: Optional[Dict[str, Any]] = None,#load_json_file(COMMAND_PATH),
) -> CheckInputResult:
    """You should validate the `config` and `command` input based on the selected strategy.
        You need to ensure that all required fields are present and correctly formatted.
        If any required field is missing or incorrectly formatted, return a message indicating the issue.
        Make sure to pass this validation step before proceeding to training.
    """
    try:
        training_meta = dpa_training_meta()
        validate(config, training_meta["config"]["schema"])
        normalized_config = normalize_dpa_config(config)
        command_input = command or {}
        validate(command_input, training_meta["command"]["schema"])
        normalized_command = normalize_dpa_command(command_input)
        return CheckInputResult(
            valid=True,
            message="Config is valid",
            command=normalized_command,
            config=normalized_config
        )
    except ValidationError as e:
        logging.exception("Config validation failed")
        return CheckInputResult(
            valid=False,
            message=f"Config validation failed: {e.message}",
            command=command or {},
            config=config
        )

class TrainingResult(TypedDict):
    """Result structure for model training"""
    status: str
    model: str
    log: str
    message: str
    test_metrics: Optional[List[Dict[str, Any]]]

@mcp.tool()
def training(
    config: Dict[str, Any], #= load_json_file(CONFIG_PATH),
    train_data: Path,# = Path(TRAIN_DATA_PATH),
    model_path: Optional[Path] = None,
    command: Optional[Dict[str, Any]] = None,#load_json_file(COMMAND_PATH),
    valid_data: Optional[Union[List[Path], Path]] = None,
    #test_data: Optional[Union[List[Path], Path]] = None,
) -> Dict[str, Any]:
    """Train a Deep Potential (DP) machine learning force field model. This tool should only be executed once all necessary inputs are gathered and validated.
       Always use 'train_input_doc' to get the template for 'config' and 'command', and use 'check_input' to validate them before calling this tool.
    
    Args:
        config: Configuration parameters for training (You can find an example for `config` from the 'train_input_doc' tool').
        command: Command parameters for training (You can find an example for `command` from the 'train_input_doc' tool').
        train_data: Path to the training dataset (required).
        model_path (Path, optional): Path to pre-trained base model. Required for model fine-tuning.
        valid_data (Path | List[Path], optional): Path(s) to validation dataset(s).
        test_data (Path | List[Path], optional): Path(s) to test dataset(s) for post-training evaluation.
    
    """
    try:
        training_meta = dpa_training_meta()
        validate(config, training_meta["config"]["schema"])
        normalized_config = normalize_dpa_config(config)
        command_input = command or {}
        validate(command_input, training_meta["command"]["schema"])
        normalized_command = normalize_dpa_command(command_input)

        work_path = Path(generate_work_path()).absolute()
        work_path.mkdir(parents=True, exist_ok=True)

        train_paths = _ensure_path_list(train_data)
        valid_paths = _ensure_path_list(valid_data)
        

        train_result = _run_dp_training(
            workdir=work_path,
            config=normalized_config,
            command=normalized_command,
            train_data=train_paths,
            valid_data=valid_paths,
            model_path=model_path,
            workflow_name=f"dpa-training-{int(time.time())}",
        )
        result = train_result

    except Exception as e:
        logging.exception("Training failed")
        result = {
            "status": "error",
            "model": None,
            "log": None,
            "message": f"Training failed: {str(e)}",
        }
    return result

@mcp.tool()
def model_test(
    model_path: Path,
    test_data: Union[List[Path], Path],
    head: Optional[str] = None,
) -> Dict[str, Any]:
    """Evaluate and validate the accuracy of a Deep Potential model on a test dataset.
    
    This tool computes prediction accuracy metrics (MAE, RMSE) for energies and forces
    by comparing model predictions against reference labeled data.
    
    Args:
        model_path: Path to the trained DPA model file (.pt or .pb).
        test_data: Path(s) to test dataset file(s) in extxyz format with labeled energies and forces.
                   Can be a single Path or a list of Paths.
        head: Optional head name to use in multitask evaluation.
    
    Returns:
        Dictionary containing:
            - status: "success" or "error"
            - test_metrics: List of metric dictionaries, one per test dataset, each containing:
                * system_idx: Index identifier for the test dataset
                * mae_e: Mean absolute error for total energy (eV)
                * rmse_e: Root mean square error for total energy (eV)
                * mae_e_atom: Mean absolute error for energy per atom (eV/atom)
                * rmse_e_atom: Root mean square error for energy per atom (eV/atom)
                * mae_f: Mean absolute error for forces (eV/Å)
                * rmse_f: Root mean square error for forces (eV/Å)
                * n_frames: Number of structures tested
            - output_dir: Path to directory containing detailed comparison files
            - message: Status or error message
    
    Example:
        >>> result = model_test(
        ...     model_path=Path("model.ckpt.pt"),
        ...     test_data=Path("test_structures.extxyz"),
        ...     head="MP_traj_v024_alldata_mixu"
        ... )
        >>> print(result["test_metrics"][0]["rmse_e_atom"])
        0.0023  # eV/atom
    """
    try:
        work_path = Path(generate_work_path()).absolute()
        work_path.mkdir(parents=True, exist_ok=True)
        test_paths = _ensure_path_list(test_data)
        logging.info(f"Evaluating model {model_path} on {len(test_paths)} test dataset(s)")
        eval_result = _evaluate_trained_model(
            workdir=work_path, 
            model_file=model_path, 
            test_data=test_paths,
            head=head
        )
        return eval_result
    except Exception as e:
        logging.exception("Model testing failed")
        return {
            "status": "error",
            "output_dir": None,
            "test_metrics": None,
            "message": f"Model evaluation failed: {e}"
        }

## ===========================
## DPA calculator tool implementations
## ==========================
@mcp.tool()
def get_base_model_path(
    model_path: Optional[Path]=None
    ) -> Dict[str,Any]:
    """Resolve a usable base model path before using `run_molecular_dynamics` tool.
    
    Args:
        model_path: Path to a specific DPA model file. 
            IMPORTANT: If you want to use the default model, DO NOT provide this parameter at all.
            Do not pass None, "None", or empty string - simply omit the parameter entirely.
    
    Returns:
        Dict containing 'base_model_path' key with the resolved Path, or None if not found.
    """

    # Handle cases where LLM might pass string "None" or empty string
    if isinstance(model_path, str) and model_path.lower() in ("none", ""):
        source = default_dpa_model_path
    else:
        source = model_path if model_path not in (None, "") else default_dpa_model_path
    if not source:
        logging.error("No model path provided and no default_dpa_model_path configured.")
        return {"base_model_path": None}

    try:
        resolved = Path(source).expanduser().resolve()
    except Exception:
        logging.exception("Failed to resolve model path.")
        return {"base_model_path": None}

    if not resolved.exists():
        logging.error(f"Model path not found: {resolved}")
        return {"base_model_path": None}

    return {"base_model_path": resolved}

@mcp.tool()
def optimize_structure( 
    input_structure: Path,
    model_path: Path,
    head: Optional[str]= None,
    force_tolerance: float = 0.01, 
    max_iterations: int = 100, 
    relax_cell: bool = False,
) -> Dict[str, Any]:
    """Optimize crystal structure using a Deep Potential (DP) model.

    Args:
        input_structure (Path): Path to the input structure file (e.g., CIF, POSCAR).
        model_path (Path): Path to the model file
            If not provided, using the `get_base_model_path` tool to obtain the default model path.
        force_tolerance (float, optional): Convergence threshold for atomic forces in eV/Å.
            Default is 0.01 eV/Å.
        max_iterations (int, optional): Maximum number of geometry optimization steps.
            Default is 100 steps.
        relax_cell (bool, optional): Whether to relax the unit cell shape and volume in addition to atomic positions.
            Default is False.
        head (str, optional): For pretrained DPA multi-head models, an available head should be provided. 
            The head is defaulted to "MP_traj_v024_alldata_mixu" for multi-task model. 


    Returns:
        dict: A dictionary containing optimization results:
            - optimized_structure (Path): Path to the final optimized structure file.
            - optimization_traj (Optional[Path]): Path to the optimization trajectory file, if available.
            - final_energy (float): Final potential energy after optimization in eV.
            - message (str): Status or error message describing the outcome.
    """
    try:
        work_path=Path(generate_work_path())
        work_path = work_path.expanduser().resolve()
        work_path.mkdir(parents=True, exist_ok=True)
        with set_directory(work_path):
            result = _optimize_structure(
                input_structure=input_structure,
                model_path=model_path,
                head=head,
                force_tolerance=force_tolerance,
                max_iterations=max_iterations,
                relax_cell=relax_cell,
            )

        return result

    except Exception as e:
        logging.error(f"Optimization failed: {str(e)}", exc_info=True)
        return {
            "optimized_structure": Path(""),
            "optimization_traj": None, 
            "final_energy": -1.0,
            "message": f"Optimization failed: {str(e)}",
        }

@mcp.tool()
def model_inference(
    structure_path: Union[List[Path], Path],
    model_path: Optional[Path] = None,
    head: Optional[str] = None,
) -> Dict[str, Any]:
    """Calculate energy and force for given structures using a Deep Potential model.

    Parameters
    - structure_path: List[Path] | Path
        Path(s) to structure file(s) (extxyz/xyz/vasp/...). Can be a multi-frame file or a list of files.
    - model_style: str
        ASE calculator key (e.g., "dpa").
    - model_path: Path
        Model file(s) or URL(s) for ML calculators. 
    - head (str, optional): For pretrained DPA multi-head models, an available head should be provided. 
        The head is defaulted to "MP_traj_v024_alldata_mixu" for multi-task model if not specified. 

    Returns
    - Dict[str, Any]
        Dictionary containing paths to labeled data file and logs.
    """
    try:
        work_path=Path(generate_work_path())
        work_path = work_path.expanduser().resolve()
        work_path.mkdir(parents=True, exist_ok=True)
        with set_directory(work_path):
            results = _model_inference(
                structure_path=structure_path,
                model_path=model_path,
                head=head,
            )
        
    except Exception as e:
        logging.error(f"Error in ase_calculation: {str(e)}")
        results={
            "status": "error",
            "labeled_data": None,
            "message": f"ASE calculation failed: {e}"
            }
    return results   
    
@mcp.tool()
def run_molecular_dynamics(
    structure_paths: Union[Path, List[Path]],
    model_path: Path,
    stages: List[Dict[str, Any]],
    head: Optional[str]=None,
    save_interval_steps: int =100,
    )-> Dict[str, Any]:   
    """
    [Modified from AI4S-agent-tools/servers/DPACalculator] Run a multi-stage molecular dynamics simulation using Deep Potential. 

    This tool performs molecular dynamics simulations with different ensembles (NVT, NPT, NVE)
    in sequence, using the ASE framework with the Deep Potential calculator.

    Args:
        initial_structure (Path): Input atomic structure file (supports .xyz, .cif, etc.)
        model_path (Path): Path to the model file
            If not provided, using the `get_base_model_path` tool to obtain the default model path.
        stages (List[Dict]): List of simulation stages. Each dictionary can contain:
            - mode (str): Simulation ensemble type. One of:
                * "NVT" or "NVT-NH"- NVT ensemble (constant Particle Number, Volume, Temperature), with Nosé-Hoover (NH) chain thermostat
                * "NVT-Berendsen"- NVT ensemble with Berendsen thermostat. For quick thermalization
                * 'NVT-Andersen- NVT ensemble with Andersen thermostat. For quick thermalization (not rigorous NVT)
                * "NVT-Langevin" or "Langevin"- Langevin dynamics. For biomolecules or implicit solvent systems.
                * "NPT-aniso" - constant Number, Pressure (anisotropic), Temperature
                * "NPT-tri" - constant Number, Pressure (tri-axial), Temperature
                * "NVE" - constant Number, Volume, Energy (no thermostat/barostat, or microcanonical)
            - runtime_ps (float): Simulation duration in picoseconds. (default: 0.5 ps)
            - temperature_K (float, optional): Temperature in Kelvin (required for NVT/NPT).
            - pressure (float, optional): Pressure in GPa (required for NPT).
            - timestep (float, optional): Time step in picoseconds (default: 0.0005 ps = 0.5 fs).
            - tau_t_ps (float, optional): Temperature coupling time in picoseconds (default: 0.01 ps).
            - tau_p_ps (float, optional): Pressure coupling time in picoseconds (default: 0.1 ps).
        save_interval_steps (int): Interval (in MD steps) to save trajectory frames (default: 100).
        traj_prefix (str): Prefix for trajectory output files (default: 'traj').
        seed (int, optional): Random seed for initializing velocities (default: 42).
        head (str, optional): For pretrained DPA multi-head models, an available head should be provided. 
                The head is defaulted to "MP_traj_v024_alldata_mixu" for multi-task model. 

    Returns: A dictionary containing:
            - trajectory_list (List[Path]): The paths of output trajectory files generated.
            - log_file (Path): Path to the log file containing simulation output.

    Examples:
        >>> stages = [
        ...     {
        ...         "mode": "NVT",
        ...         "temperature_K": 300,
        ...         "runtime_ps": 5,
        ...         "timestep_ps": 0.0005,
        ...         "tau_t_ps": 0.01
        ...     },
        ...     {
        ...         "mode": "NPT-aniso",
        ...         "temperature_K": 300,
        ...         "pressure": 1.0,
        ...         "runtime_ps": 5,
        ...         "timestep_ps": 0.0005,
        ...         "tau_t_ps": 0.01,
        ...         "tau_p_ps": 0.1
        ...     },
        ...     {
        ...         "mode": "NVE",
        ...         "runtime_ps": 5,
        ...         "timestep_ps": 0.0005
        ...     }
        ... ]

        >>> result = run_molecular_dynamics(
        ...     initial_structure=Path("input.xyz"),
        ...     model_path=Path("model.pb"),
        ...     stages=stages,
        ...     save_interval_steps=50,
        ...     traj_prefix="cu_relax",
        ...     seed=42
        ... )
    """
    try:
        if isinstance(structure_paths, (str, Path)):
            structure_paths = [Path(structure_paths)]
        else:
            structure_paths = [Path(p) for p in structure_paths]
    
        # Validate inputs
        for path in structure_paths:
            if not path.exists():
                raise FileNotFoundError(f"Structure file not found: {path}")
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        from dpa_tool.ase_md import _run_md
        results=_run_md(
            model_path=model_path,
            structure_path=structure_paths,
            stages=stages,
            head=head,
            save_interval_steps=save_interval_steps,
            traj_prefix="traj",
            seed=42
        )    
        return results
    except Exception as e:
        logging.error(f"Error in run_molecular_dynamics: {str(e)}")
        return {
            "status": "error",
            "labeled_data": None,
            "message": f"MD simulation failed: {e}"
            }

if __name__ == "__main__":
    def create_workpath(work_path=None):
        """
        Create the working directory for AbacusAgent, and change the current working directory to it.
    
        Args:
            work_path (str, optional): The path to the working directory. If None, a default path will be used.
    
        Returns:
            str: The path to the working directory.
        """
        work_path = os.environ.get("DPA_SERVER_WORK_PATH", DPA_SERVER_WORK_PATH) + f"/{time.strftime('%Y%m%d%H%M%S')}"
        os.makedirs(work_path, exist_ok=True)
        os.chdir(work_path)
        print(f"Changed working directory to: {work_path}")
        return work_path    
    
    create_workpath()
    mcp.run(transport=args.transport)