from typing import List, Optional, Union, Dict, Any

from dotenv import load_dotenv
from pathlib import Path
import os
import argparse
import traceback
import time
import uuid

from mattergen_tool.common import (
    mattergen_to_ase, ase_to_mattergen, mattergen_train, mattergen_generate
)

_script_dir = Path(__file__).parent
load_dotenv(_script_dir / ".env", override=True)

MATTERGEN_MODEL_ROOT = "/opt/mattergen/checkpoints/"
MATTERGEN_SERVER_WORK_PATH = "/tmp/mattergen_server"

default_mattergen_model_root= os.environ.get("MATTERGEN_MODEL_ROOT", MATTERGEN_MODEL_ROOT)
default_mattergen_server_work_path = os.environ.get("MATTERGEN_SERVER_WORK_PATH", MATTERGEN_SERVER_WORK_PATH)
default_mattergen_venv_root = os.environ.get("MATTERGEN_VENV_ROOT", None)


def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Mattergen MCP server Command Line Interface")

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
        help="MCP implementation Model to use (default: dp), choices: fastmcp, dp"
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
## Mattergen tool implementations
## =========================
@mcp.tool()
def mattergen_ase_convert_tool(
        input_file: Union[str, Path],
        output_file: Union[str, Path],
        properties: Optional[List[str]] = None,
        mode: str = "auto"
) -> Dict[str, Any]:
    """Tool to interconvert data between ASE and Mattergen formats.

    Supported I/O formats include any format supported by ASE (e.g., .xyz, .cif)
    and Mattergen's internal .npy data format. The mode argument specifies the direction of conversion.

    Mattergen's internal .npy files:
    - cell.npy: (N, 3, 3) lattice vectors
    - pos.npy: (total_atoms, 3) concatenated positions
    - num_atoms.npy: (N,) atom counts per structure
    - atomic_numbers.npy: (total_atoms,) concatenated atomic numbers
    - structure_id.npy: (N,) structure IDs
    - {property}.json: optional property data

    Args:
        input_file (Union[str, Path]): path to input file for conversion.
        output_file (Union[str, Path]): path to output file for conversion.
        properties (Optional[List[str]]):
            Optional list of property names to read from .json files. Defaults to None.
            Allowed property names include:
                - "chemical_system": chemical system (e.g. "Fe-O") as a string
                - "energy_above_hull": energy above hull in eV/atom as a float.
                - "dft_band_gap": DFT band gap in eV as a float.
                - "dft_mag_density": DFT magnetization density in mu_B/Å^3 as a float.
                - "hhi_score": HHI score as a float.
                - "ml_bulk_modulus": machine learning predicted bulk modulus in GPa as a float
                - "space_group": space group number as an integer.
            The exact choice of properties should be determined from the training conditions of the model
            you wish to use or have used in this workflow.
        mode (str): direction of conversion, either "auto", "ase_to_mattergen", or "mattergen_to_ase".
            If "auto", the function will infer the mode from the file extensions of input_file and output_file.
    Returns:
        Dict[str, Any]:
         A dictionary containing the status of the conversion, the path to the converted file(s), and a message.
    """
    try:
        if mode == "auto":
            if output_file.endswith((".xyz", ".extxyz", ".cif")):
                mode = "mattergen_to_ase"
            elif input_file.endswith((".xyz", ".extxyz", ".cif")):
                mode = "ase_to_mattergen"
            else:
                raise ValueError(
                    "Could not infer conversion mode from file extensions."
                    " Please specify mode explicitly as either 'ase_to_mattergen' or 'mattergen_to_ase'."
                )

        time_init = time.time()
        if mode == "ase_to_mattergen":
            output_path = ase_to_mattergen(input_file, output_file, properties)
        elif mode == "mattergen_to_ase":
            output_path = mattergen_to_ase(input_file, output_file, properties)
        else:
            raise ValueError(
                f"Invalid mode: {mode}. Must be either 'auto', 'ase_to_mattergen' or 'mattergen_to_ase'."
            )

        return {
            "status": "success",
            "output_path": output_path,
            "message": f"Conversion completed successfully in {time.time() - time_init:.2f} seconds.",
        }

    except Exception as e:
        return {
            "status": "error",
            "output_path": None,
            "message": f"Conversion failed with error: {str(e)}",
        }


@mcp.tool()
def mattergen_train_tool(
        model_root: Union[str, Path],
        data_root: Union[str, Path],
        conditioned_properties: Optional[List[str]] = None,
        skip: bool = False,
        additional_args: Optional[Dict] = None,
        custom_cmd: Optional[str] = None,
        env_vars: Optional[Dict] = None,
        venv_root: Optional[str] = None,
        output_dir: Union[str, Path] = "./training_outputs",
) -> Dict[str, Any]:
    """Tool to fine-tune mattergen models from unconditional or conditional base models.

    Args:
        model_root (Union[str, Path]):
            Path to the model root directory. Contains each branch model under subdirectories
            named after the conditioning mode (e.g. "chemical_system", "dft_band_gap", etc.).
        data_root (Union[str, Path]):
            Path to the dataset root directory. Must contain train, may contain val and test subdirectories
            with mattergen structure data in .npy format, and conditioning properties in .json format if applicable.
        conditioned_properties (List[str], optional):
            Must correspond to a subdirectory under model_root containing the base model to finetune from,
            and determines which conditioning properties are used for training.
            Default is ["chemical_system", "energy_above_hull"].
            Available:
            - ["chemical_system"]: conditions on chemical system only, no additional properties.
            - ["chemical_system", "energy_above_hull"]: conditions on chemical system and energy above hull.
            - ["dft_band_gap"]: conditions on DFT band gap.
            - ["dft_mag_density"]: conditions on DFT magnetization density.
            - ["dft_mag_density", "hhi_score"]: conditions on DFT magnetization density and HHI score.
            - ["mattergen_base"]: no conditioning.
            - ["ml_bulk_modulus"]: conditions on machine learning predicted bulk modulus.
            - ["space_group"]: conditions on space group number.
        skip (bool, optional):
            If True, skip training and use the provided model directly.
            In this case, the returned training script and log will be placeholders indicating that
            training was skipped.
        additional_args(List[str], optional):
            Additional arguments to override the defaults.
            Should be in the same format as `common_train_args`,
            and will be merged with `common_train_args` with `additional_args` taking precedence.
        custom_cmd(str, optional):
            If provided, this command will be run directly instead of constructing a command
            from the other arguments, allowing for maximum flexibility in executing training
            with custom settings.
        env_vars(Dict, optional):
            Environment variables to set before running the training command.
        venv_root(str, optional):
            Root path to a virtual environment to activate before running the training command.
            Recommended to set as mattergen is typically installed in a venv.
        output_dir(str or path, optional):
            Directory to save training outputs (training script, log, checkpoints). Default is "./training_outputs".
    Returns:
        Dict[str, Any]:
         A dictionary containing:
            - the status of the training
            - path to the training script
            - path to the log file
            - path to model checkpoints
            - and a message.
    """
    try:
        time_init = time.time()
        script_path, log_path, model_path, extra_paths  = mattergen_train(
            model_root,
            data_root,
            conditioned_properties,
            skip,
            additional_args,
            custom_cmd,
            env_vars,
            venv_root,
            output_dir
        )
        return {
            "status": "success",
            "script_path": script_path,
            "log_path": log_path,
            "model_save_path": model_path,
            "extra_output_paths": extra_paths,
            "message": f"Mattergen training completed successfully in {time.time() - time_init:.2f} seconds.",
        }
    except Exception as e:
        return {
            "status": "error",
            "script_path": None,
            "log_path": None,
            "model_save_path": None,
            "extra_output_paths": None,
            "message": f"Mattergen training failed with error: {str(e)}",
        }


@mcp.tool()
def mattergen_generate_tool(
        model_path: Union[str, Path],
        results_dir: Union[str, Path] = "./generated_results",
        conditioned_property_values: Optional[Dict[str, Any]] = None,
        additional_args: Optional[Dict] = None,
        custom_cmd: Optional[str] = None,
        env_vars: Optional[Dict] = None,
        venv_root: Optional[str] = None,
) -> Dict[str, Any]:
    """Tool to generate new crystal structure candidates using an existing mattergen model with or without condition.

    Args:
        model_path (str or Path):
            Path to the model. Must contain a checkpoint file and a config.yaml.
        results_dir (str or Path, optional):
            Directory to save generation results. Default is "./generated_results".
        conditioned_property_values (Dict[str, Any], optional):
            Classifier-free Conditions used for generation, provided as a dictionary mapping property names
            to their desired values.
            Must correspond to a subdirectory under model_root containing the base model to finetune from,
            and determines which conditioning properties are used for training.
            Default is empty condition.
            Possible choices of keys:
            - ["chemical_system"]: conditions on chemical system only, no additional properties.
            - ["chemical_system", "energy_above_hull"]: conditions on chemical system and energy above hull.
            - ["dft_band_gap"]: conditions on DFT band gap.
            - ["dft_mag_density"]: conditions on DFT magnetization density.
            - ["dft_mag_density", "hhi_score"]: conditions on DFT magnetization density and HHI score.
            - ["mattergen_base"]: no conditioning.
            - ["ml_bulk_modulus"]: conditions on machine learning predicted bulk modulus.
            - ["space_group"]: conditions on space group number.
            But the exact choice should be determined from the trained conditioning of the model you use.
        additional_args (Dict, optional):
            Additional arguments for mattergen-generate to override the defaults for generation.
            Supported arguments include:
            - batch_size: number of structures to generate in each batch. Default is 128.
            - num_batches: number of batches to generate. Default is 26.
            - diffusion_guidance_factor:
                guidance scale for classifier-free guidance during sampling. Default is 2.0.
                Modification not recommended.
        custom_cmd (str, optional):
           If provided, this command will be run directly instead of constructing a command
            from the other arguments, allowing for maximum flexibility in executing generation
            with custom settings.
        env_vars (Dict, optional):
            Environment variables to set before running the generation command.
        venv_root (str, optional):
            Root path to a virtual environment to activate before running the generation command.
            Recommended to set as mattergen is typically installed in a venv.
    Returns:
            Dict[str, Any]:
            A dictionary containing:
                - the status of the generation
                - Path to the generated crystals in .extxyz format.
                - List of any additional output files generated during generation
                 (e.g. trajectories in .zip format).
                - and a message.
    """
    try:
        time_init = time.time()
        generated_crystals_path, extra_paths = mattergen_generate(
            model_path,
            results_dir,
            conditioned_property_values,
            additional_args,
            custom_cmd,
            env_vars,
            venv_root
        )
        return {
            "status": "success",
            "generated_crystals_path": generated_crystals_path,
            "extra_output_paths": extra_paths,
            "message": f"Mattergen generation completed successfully in {time.time() - time_init:.2f} seconds.",
        }
    except Exception as e:
        return {
            "status": "error",
            "generated_crystals_path": None,
            "extra_output_paths": None,
            "message": f"Mattergen generation failed with error: {str(e)}",
        }


# Run the MCP server.
if __name__ == "__main__":
    def create_workpath(work_path=None):
        """
        Create the working directory for AbacusAgent, and change the current working directory to it.

        Args:
            work_path (str, optional): The path to the working directory. If None, a default path will be used.

        Returns:
            str: The path to the working directory.
        """
        work_path = default_mattergen_server_work_path + f"/{time.strftime('%Y%m%d%H%M%S')}"
        os.makedirs(work_path, exist_ok=True)
        os.chdir(work_path)
        print(f"Changed working directory to: {work_path}")
        return work_path

    create_workpath()
    mcp.run(transport=args.transport)
