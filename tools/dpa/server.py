import argparse
import os
import argparse
from typing import Optional, Union, Literal, Dict, Any, List, Tuple
from pathlib import Path
import time
from dotenv import load_dotenv
from matcreator.tools.dpa import (
    get_base_model_path as _get_base_model_path,
    optimize_structure as _optimize_structure,
    run_molecular_dynamics as _run_molecular_dynamics,
    ase_calculation as _ase_calculation,
    train_input_doc as _train_input_doc,
    check_train_data as _check_train_data,
    check_input as _check_input,
    training as _training,
    )

load_dotenv(os.path.expanduser(".env"), override=True)

DPA_MODEL_PATH = "/home/ruoyu/dev/PFD-Agent/.tests/dpa/DPA2_medium_28_10M_rc0.pt"
DPA_SERVER_WORK_PATH = "/tmp/dpa_server"


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
    #set_envs()
    #create_workpath()
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
    
@mcp.tool()
def get_base_model_path(
    model_path: Optional[Path]=DPA_MODEL_PATH
    ) -> Dict[str,Any]:
    """Resolve a usable base model path before using `run_molecular_dynamics` tool.
        Args:
            model_path (Path, optional): Explicit model path provided by the user, default to None. If 'None', the tool should determine
            the default model path from environment variables or other means.

        Returns: A dictionary contains:
            - base_model_path: normalized local Path or an HTTP(S) URI string (framework will serialize Paths),
            or None if nothing can be determined.
        """
    return _get_base_model_path(
            model_path=model_path
        )
    
@mcp.tool()
def optimize_structure( 
        input_structure: Path,
        model_path: Optional[Path]= None,
        head: Optional[str]= None,
        force_tolerance: float = 0.01, 
        max_iterations: int = 100, 
        relax_cell: bool = False,
        ) -> Dict[str, Any]:
    """
        Optimize crystal structure using a Deep Potential (DP) model.

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
    return _optimize_structure( 
            input_structure=input_structure,
            model_path=model_path,
            head=head,
            force_tolerance=force_tolerance,
            max_iterations=max_iterations,
            relax_cell=relax_cell
        )
        
    
@mcp.tool()
def run_molecular_dynamics(
        initial_structure: Path,
        stages: List[Dict],
        model_path: Optional[Path]= DPA_MODEL_PATH,
        save_interval_steps: int = 100,
        traj_prefix: str = 'traj',
        seed: Optional[int] = 42,
        head: Optional[str] = None,
        ) -> Dict:
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
    return _run_molecular_dynamics(
            initial_structure=initial_structure,
            stages=stages,
            model_path=model_path,
            save_interval_steps=save_interval_steps,
            traj_prefix=traj_prefix,
            seed=seed,
            head=head)  
        
@mcp.tool()
def ase_calculation(
        structure_path: Union[List[Path], Path],
        model_path: Optional[Path] = None,
        head: Optional[str] = None,
            ) -> Dict[str, Any]:
    """
        Labeling energy and force (and stress)on given structures using a Deep Potential model.

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
    return _ase_calculation(
            structure_path=structure_path,
            model_path=model_path,
            head=head
        )
    
@mcp.tool()
def train_input_doc() -> Dict[str, Any]:
    """
    Check the training input document for Deep Potential model training.
    Returns:
        List metadata for training a Deep Potential model. 
        You can use these information to formulate template 'config' and 'command' dict.
    """
    return _train_input_doc()
    
@mcp.tool()
def check_train_data(
        train_data: Path,
        valid_ratio: Optional[float] = 0.0,
        test_ratio: Optional[float] = 0.0,
        shuffle: bool = True,
        seed: Optional[int] = None,
        output_dir: Optional[Path] = None,
    ):
    """
        Inspect training data and optionally produce a train/valid/test split.

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
    return _check_train_data(
            train_data=train_data,
            valid_ratio=valid_ratio,
            test_ratio=test_ratio,
            shuffle=shuffle,
            seed=seed,
            output_dir=output_dir
        )
    
@mcp.tool()
def check_input(
        config: Dict[str, Any], #= load_json_file(CONFIG_PATH),
        command: Optional[Dict[str, Any]] = {},#load_json_file(COMMAND_PATH),
        #strategy: str = "dpa",
    ) -> Any:
    """
        Validate the `config` and `command` input.
        
        Pay special attention to the 'finetune_mode' key in `command` dict, it is 'True' for fine-tuning and `False` otherwise.
        
        You need to ensure that all required fields are present and correctly formatted.
        
        If any required field is missing or incorrectly formatted, return a message indicating the issue.
        
        Make sure to pass this validation step before proceeding to training.
        """
    return _check_input(
            config=config,
            command=command,
        )
    
    
@mcp.tool()
def training(
        config: Dict[str, Any], #= load_json_file(CONFIG_PATH),
        train_data: Path,# = Path(TRAIN_DATA_PATH),
        model_path: Optional[Path] = DPA_MODEL_PATH,
        command: Optional[Dict[str, Any]] = {},
        valid_data: Optional[Union[List[Path], Path]] = None,
        test_data: Optional[Union[List[Path], Path]] = None,
    ) -> Any:
    """Train a Deep Potential (DP) machine learning force field model. This tool should only be executed once all necessary inputs are gathered and validated.
            Always use 'train_input_doc' to get the template for 'config' and 'command', and use 'check_input' to validate them before calling this tool.
    
            Args:
                config: Configuration parameters for training (You can find an example for `config` from the 'train_input_doc' tool').
                command: Command parameters for training. 
                train_data: Path to the training dataset (required).
                model_path (Path, optional): Path to pre-trained base model. Required for model fine-tuning.
    
    """
    return _training(
            config=config,
            train_data=train_data,
            model_path=model_path,
            command=command,
            valid_data=valid_data,
            test_data=test_data
        )
    
    
    

if __name__ == "__main__":
    create_workpath()
    mcp.run(transport=args.transport)