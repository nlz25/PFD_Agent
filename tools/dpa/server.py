import argparse
import os
import time
import logging
import random
import traceback
import uuid
from typing import Optional, Union, Dict, Any, List, Tuple, TypedDict,Literal
from pathlib import Path
import numpy as np
import subprocess
import sys
import shlex
import selectors
from jsonschema import validate, ValidationError
from dotenv import load_dotenv

# ASE / MD imports used by DPA tools
from ase.io import read, write
from ase.atoms import Atoms
from ase.optimize import BFGS
from ase.constraints import ExpCellFilter
from deepmd.calculator import DP

from dflow.plugins.dispatcher import (
    DispatcherExecutor,
)

from dpa_tool.train import (
    dpa_training_meta,
    normalize_dpa_command,
    normalize_dpa_config,
    _run_dp_training,
    _evaluate_trained_model,
    _ensure_path_list
)

from dpa_tool.utils import (
    set_directory,
    bohrium_config_from_dict
    )

from dpa_tool.ase_md import _run_molecular_dynamics_batch

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

def run_command(
    cmd: Union[List[str], str],
    raise_error: bool = True,
    input: Optional[str] = None,
    try_bash: bool = False,
    login: bool = True,
    interactive: bool = True,
    shell: bool = False,
    print_oe: bool = False,
    stdout=None,
    stderr=None,
    **kwargs,
) -> Tuple[int, str, str]:
    """
    Run shell command in subprocess

    Parameters:
    ----------
    cmd: list of str, or str
        Command to execute
    raise_error: bool
        Wheter to raise an error if the command failed
    input: str, optional
        Input string for the command
    try_bash: bool
        Try to use bash if bash exists, otherwise use sh
    login: bool
        Login mode of bash when try_bash=True
    interactive: bool
        Alias of login
    shell: bool
        Use shell for subprocess.Popen
    print_oe: bool
        Print stdout and stderr at the same time
    **kwargs:
        Arguments in subprocess.Popen

    Raises:
    ------
    AssertionError:
        Raises if the error failed to execute and `raise_error` set to `True`

    Return:
    ------
    return_code: int
        The return code of the command
    out: str
        stdout content of the executed command
    err: str
        stderr content of the executed command
    """
    if print_oe:
        stdout = sys.stdout
        stderr = sys.stderr

    if isinstance(cmd, str):
        if shell:
            cmd = [cmd]
        else:
            cmd = cmd.split()
    elif isinstance(cmd, list):
        cmd = [str(x) for x in cmd]

    if try_bash:
        arg = "-lc" if (login and interactive) else "-c"
        script = "if command -v bash 2>&1 >/dev/null; then bash %s " % arg + \
            shlex.quote(" ".join(cmd)) + "; else " + " ".join(cmd) + "; fi"
        cmd = [script]
        shell = True

    with subprocess.Popen(
        args=cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=shell,
        **kwargs,
    ) as sub:
        if stdout is not None or stderr is not None:
            if input is not None:
                sub.stdin.write(bytes(input, encoding=sys.stdout.encoding))
                sub.stdin.close()
            out = ""
            err = ""
            sel = selectors.DefaultSelector()
            sel.register(sub.stdout, selectors.EVENT_READ)
            sel.register(sub.stderr, selectors.EVENT_READ)
            stdout_eof = False
            stderr_eof = False
            while not (stdout_eof and stderr_eof):
                for key, _ in sel.select():
                    line = key.fileobj.readline().decode(sys.stdout.encoding)
                    if not line:
                        if key.fileobj is sub.stdout:
                            stdout_eof = True
                        if key.fileobj is sub.stderr:
                            stderr_eof = True
                        continue
                    if key.fileobj is sub.stdout:
                        if stdout is not None:
                            stdout.write(line)
                            stdout.flush()
                        out += line
                    else:
                        if stderr is not None:
                            stderr.write(line)
                            stderr.flush()
                        err += line
            sub.wait()
        else:
            out, err = sub.communicate(bytes(
                input, encoding=sys.stdout.encoding) if input else None)
            out = out.decode(sys.stdout.encoding)
            err = err.decode(sys.stdout.encoding)
        return_code = sub.poll()
    if raise_error:
        assert return_code == 0, "Command %s failed: \n%s" % (cmd, err)
    return return_code, out, err


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
    model: Path
    log: Path
    message: str
    test_metrics: Optional[List[Dict[str, Any]]]

@mcp.tool()
def training(
    config: Dict[str, Any], #= load_json_file(CONFIG_PATH),
    train_data: Path,# = Path(TRAIN_DATA_PATH),
    model_path: Optional[Path] = None,
    command: Optional[Dict[str, Any]] = None,#load_json_file(COMMAND_PATH),
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
        model, log, message = _run_dp_training(
            workdir=work_path,
            config=normalized_config,
            command=normalized_command,
            train_data=train_paths,
            valid_data=valid_paths,
            model_path=model_path,
        )

        logging.info("Training completed!")
        test_metrics: Optional[List[Dict[str, Any]]] = None
        test_paths = _ensure_path_list(test_data)
        if test_paths:
            test_metrics = _evaluate_trained_model(work_path, model, test_paths)
        result = {
            "status": "success",
            "model": str(model.resolve()),
            "log": str(log.resolve()),
            "message": message,
            "test_metrics": test_metrics,
        }

    except Exception as e:
        logging.exception("Training failed")
        result = {
            "status": "error",
            "model": None,
            "log": None,
            "message": f"Training failed: {str(e)}",
        }
    return result

## ===========================
## DPA calculator tool implementations
## ==========================

@mcp.tool()
def get_base_model_path(
    model_path: Optional[Path]=None
    ) -> Dict[str,Any]:
    """Resolve a usable base model path before using `run_molecular_dynamics` tool."""

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
    model_path: Optional[Path]= None,
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
        base_name = input_structure.stem
        
        logging.info(f"Reading structure from: {input_structure}")
        atoms = read(str(input_structure))
       
        # Setup calculator
        calc=DP(model=model_path, head=head)  
        atoms.calc = calc

        traj_file = f"{base_name}_optimization_traj.extxyz"  
        if Path(traj_file).exists():
            logging.warning(f"Overwriting existing trajectory file: {traj_file}")
            Path(traj_file).unlink()

        logging.info("Starting structure optimization...")

        if relax_cell:
            logging.info("Using cell relaxation (ExpCellFilter)...")
            ecf = ExpCellFilter(atoms)
            optimizer = BFGS(ecf, trajectory=traj_file)
            optimizer.run(fmax=force_tolerance, steps=max_iterations)
        else:
            optimizer = BFGS(atoms, trajectory=traj_file)
            optimizer.run(fmax=force_tolerance, steps=max_iterations)
            
        work_path=Path(generate_work_path())
        work_path = work_path.expanduser().resolve()
        work_path.mkdir(parents=True, exist_ok=True)

        output_file = work_path / f"{base_name}_optimized.cif"
        write(output_file, atoms)
        final_energy = float(atoms.get_potential_energy())

        logging.info(
            f"Optimization completed in {optimizer.nsteps} steps. "
            f"Final energy: {final_energy:.4f} eV"
        )

        return {
            "optimized_structure": Path(output_file),
            "optimization_traj": Path(traj_file),
            "final_energy": final_energy,
            "message": f"Successfully completed in {optimizer.nsteps} steps",
        }

    except Exception as e:
        logging.error(f"Optimization failed: {str(e)}", exc_info=True)
        return {
            "optimized_structure": Path(""),
            "optimization_traj": None, 
            "final_energy": -1.0,
            "message": f"Optimization failed: {str(e)}",
        }

@mcp.tool()
def ase_calculation(
    structure_path: Union[List[Path], Path],
    model_path: Optional[Path] = None,
    head: Optional[str] = None,
) -> Dict[str, Any]:
    """Perform energy and force (and stress) calculation on given structures using a Deep Potential model.

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
        
        calc=DP(
            model=model_path, 
            head=head
            )    
        
        atoms_ls=[]
        if isinstance(structure_path, Path):
            structure_path = [structure_path]
        for path in structure_path:
            read_atoms = read(path, index=":")
            if isinstance(read_atoms, Atoms):
                atoms_ls.append(read_atoms)
            else:
                atoms_ls.extend(read_atoms)
        
        for atoms in atoms_ls:
            atoms.calc = calc
            energy= atoms.get_potential_energy()
            forces=atoms.get_forces()
            stress = atoms.get_stress()
            atoms.calc.results.clear()
            atoms.info['energy'] = energy
            atoms.set_array('forces', forces)
            atoms.info['stress'] = stress
        labeled_data = work_path / "ase_results.extxyz"
        write(labeled_data, atoms_ls, format="extxyz")
        
        result = {
            "status": "success",
            "labeled_data": str(labeled_data.resolve()),
            "message": f"ASE calculation completed for {len(atoms_ls)} structures."
        }
    
    except Exception as e:
        logging.error(f"Error in ase_calculation: {str(e)}")
        result={
            "status": "error",
            "message": f"ASE calculation failed: {e}"
            }
    return result   
    
@mcp.tool()
def run_molecular_dynamics(
    structure_paths: Union[Path, List[Path]],
    model_path: Path,
    config: Dict[str, Any],
    workflow_name: str = "molecular-dynamics-batch",
    #mode: Literal["debug", "bohrium"] = "debug",
    #prep_md_config: Optional[Dict[str, Any]] = None,
    #run_md_config: Optional[Dict[str, Any]] = None,
):
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
    import dpa_tool
    import dflow
    import ase 
    
    mode=os.environ.get("DPA_SUBMIT_TYPE", "bohrium")
    
    work_path=Path(generate_work_path())
    work_path = work_path.resolve()
    work_path.mkdir(parents=True, exist_ok=True)
    python_packages=[]
    python_packages.append(list(dpa_tool.__path__)[0])
    python_packages.append(list(ase.__path__)[0])
    python_packages.append(list(dflow.__path__)[0])
    
    print(python_packages)
        
    if mode == 'bohrium':
        debug=False
        try:
            bohrium_config_from_dict(
                {
                    "username": os.environ["BOHRIUM_USERNAME"],
                    "password": os.environ["BOHRIUM_PASSWORD"],
                    "project_id": os.environ["BOHRIUM_PROJECT_ID"]}
                )
            run_md_config={
                "template_config": {
                    "image":os.environ["BOHRIUM_DPA_IMAGE"],
                    "python_packages":python_packages
                    },
                "template_slice_config": {"group_size": 1, "pool_size": 1},
                "executor": DispatcherExecutor(
                    image_pull_policy="IfNotPresent",
                    machine_dict={
                        "batch_type": "Bohrium",
                        "context_type": "Bohrium",
                        "remote_profile": {
                            "input_data": {
                            "job_type": "container",
                            "platform": "ali",
                            "scass_type": os.environ["BOHRIUM_DPA_MACHINE"]}}})}
            
            prep_md_config={
                "template_config": {
                    "image":os.environ["BOHRIUM_DPA_IMAGE"],
                    "python_packages":python_packages
                    },
                "executor": DispatcherExecutor(
                    image_pull_policy="IfNotPresent",
                    machine_dict={
                        "batch_type": "Bohrium",
                        "context_type": "Bohrium",
                        "remote_profile": {
                            "input_data": {
                            "job_type": "container",
                            "platform": "ali",
                            "scass_type": "c2_m4_cpu"
                            }}})}
            
            
        except Exception as e:
            logging.error("Bohrium configuration failed. Check environment variables.", exc_info=True)
            return {
                "status": "error",
                "message": f"Bohrium configuration failed: {str(e)}"
            }
            
    else:
        prep_md_config=None
        run_md_config=None
        debug= True
        
    with set_directory(work_path):
        result=_run_molecular_dynamics_batch(
                structure_paths=structure_paths,
                model_path=model_path,
                config=config,
                workflow_name=workflow_name,
                debug=debug,
                prep_md_config=prep_md_config,
                run_md_config=run_md_config
            )
        
        # If successful, scan for result files and organize them
        if result.get("status") == "success" and "download_path" in result:
            download_path = Path(result["download_path"])
            
            # Scan for task directories and their files
            task_results = {}
            log_files = []
            status_files = []
            traj_files = []
            
            # Find all task directories
            task_dirs = list(download_path.glob("task.[0-9]*"))
            
            for task_dir in sorted(task_dirs):
                task_name = task_dir.name
                task_results[task_name] = {}
                
                # Find log files
                task_logs = list(task_dir.glob("*.log"))
                if task_logs:
                    task_results[task_name]["log"] = task_logs[0].resolve()
                    log_files.extend([str(f.resolve()) for f in task_logs])
                
                # Find status/json files  
                task_jsons = list(task_dir.glob("*.json"))
                if task_jsons:
                    task_results[task_name]["status"] = task_jsons[0].resolve()
                    status_files.extend([str(f.resolve()) for f in task_jsons])
                
                # Find trajectory files
                task_trajs = list(task_dir.glob("trajs_files/*.extxyz"))
                if task_trajs:
                    task_results[task_name]["trajectories"] = [f.resolve() for f in task_trajs]
                    traj_files.extend([str(f.resolve()) for f in task_trajs])
            
            # Update result with organized file information
            result.update({
                "task_results": task_results,
                "all_trajectory_files": traj_files,
                "all_log_files": log_files,
                "all_status_files": status_files,
                "num_tasks": len(task_dirs)
            })
            
            logging.info(f"Found {len(task_dirs)} tasks with {len(traj_files)} trajectory files")
        
        else:
            logging.error("Molecular dynamics batch run failed or no download path found.")
            result = {
                "status": "error",
                "message": "Molecular dynamics batch run failed or no download path found."
            }
        
    return result


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