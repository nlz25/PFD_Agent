"""Tool functions for MatterGen training and generation."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from shutil import copy
from typing import Dict, List, Optional, Union, Any, Tuple
import logging
from ase.io import read, write
from ase import Atoms
import numpy as np
import json

from .submission import dflow_remote_execution
from .utils import prepare_train_command, dict_to_fire_args


### Functions that can be run fully locally.

def ase_to_mattergen(
        ase_extxyz_file: Union[str, Path],
        mattergen_data: Union[str, Path],
        properties: Optional[List[str]] = None,
) -> str:
    """Converts ase.extxyz file to mattergen data format.

    Saves structures as separate .npy files:
    - cell.npy: (N, 3, 3) lattice vectors
    - pos.npy: (total_atoms, 3) concatenated positions
    - num_atoms.npy: (N,) atom counts per structure
    - atomic_numbers.npy: (num_atoms,) concatenated atomic numbers
    - structure_id.npy: (num_atoms,) structure index for each atom

    Args:
        ase_extxyz_file(Union[str, Path]): Path to input extxyz file containing structures.
        mattergen_data(Union[str, Path]): Output directory to save MatterGen .npy files.
        properties (Optional[List[str]]): List of property names to extract from ASE info dict
        and save as {property}.json files. By default, no additional properties are saved.
        Allowed property names include:
            - "chemical_system": chemical system (e.g. "Fe-O") as a string.
            - "energy_above_hull": energy above hull in eV/atom as a float.
            - "dft_band_gap": DFT band gap in eV as a float.
            - "dft_mag_density": DFT magnetization density in mu_B/Å^3 as a float.
            - "hhi_score": HHI score as a float.
            - "ml_bulk_modulus": machine learning predicted bulk modulus in GPa as a float.
            - "space_group": space group number as an integer.
        The exact choice of properties should be determined from the training conditions of the model
        you wish to use.
    Returns:
        str: Path to the directory containing MatterGen data files.
    """
    if properties is None:
        properties = []

    atoms_ls = read(ase_extxyz_file, ":")
    if isinstance(mattergen_data, str):
        mattergen_data = Path(mattergen_data)
    mattergen_data.mkdir(parents=True, exist_ok=True)
    num_structures = len(atoms_ls)

    # Prepare arrays
    cells = []
    positions = []
    num_atoms = []
    atomic_numbers = []
    structure_ids = []
    properties_data = {
        prop: {'values': [], 'property_source_doc_id': prop, 'origins': None}
        for prop in properties
    }
    for idx, atoms in enumerate(atoms_ls):
        cells.append(atoms.get_cell().array)
        positions.append(atoms.get_scaled_positions())  # Use fractional coordinates
        num_atoms.append(len(atoms))
        atomic_numbers.append(atoms.get_atomic_numbers())
        if atoms.info.get("structure_id") is not None:
            structure_ids.append(atoms.info["structure_id"])
        else:
            structure_ids.append("%06d" % idx)
        for prop in properties:
            if prop in atoms.info:
                properties_data[prop]['values'].append(atoms.info[prop])
            else:
                properties_data[prop]['values'].append(None)
                logging.warning(f"Property '{prop}' not found in structure {idx}.")

    # Convert to numpy arrays
    cells_array = np.array(cells)  # (N, 3, 3)
    pos_array = np.vstack(positions)  # (total_atoms, 3)
    num_atoms_array = np.array(num_atoms, dtype=np.int32)  # (N,)
    atomic_numbers_array = np.concatenate(atomic_numbers)  # (total_atoms,)
    structure_id_array = np.array(structure_ids)  # (total_atoms,)

    # Save to disk
    np.save(mattergen_data / "cell.npy", cells_array)
    np.save(mattergen_data / "pos.npy", pos_array)
    np.save(mattergen_data / "num_atoms.npy", num_atoms_array)
    np.save(mattergen_data / "atomic_numbers.npy", atomic_numbers_array)
    np.save(mattergen_data / "structure_id.npy", structure_id_array)

    for prop in properties:
        with open(mattergen_data / f"{prop}.json", 'w') as f:
            json.dump(properties_data[prop], f, indent=4)

    logging.info(f"Saved {num_structures} structures to {mattergen_data}")
    logging.info(f"Total atoms: {len(pos_array)}")

    return str(mattergen_data)


def mattergen_to_ase(
        mattergen_data: Union[str, Path],
        ase_extxyz_file: Union[str, Path],
        properties: Optional[List[str]] = None,
) -> str:
    """Converts mattergen data format to ase.extxyz file.

    Reads structures from separate .npy files:
    - cell.npy: (N, 3, 3) lattice vectors
    - pos.npy: (total_atoms, 3) concatenated positions
    - num_atoms.npy: (N,) atom counts per structure
    - atomic_numbers.npy: (total_atoms,) concatenated atomic numbers
    - structure_id.npy: (N,) structure IDs
    - {property}.json: optional property data

    Outputs a single extxyz file with all structures, using fractional coordinates and including properties
     in the info dictionary.

    Args:
        mattergen_data: Path to directory containing MatterGen .npy files
        ase_extxyz_file: Output path for extxyz file
        properties: Optional list of property names to read from .json files.
        Allowed property names include:
            - "chemical_system": chemical system (e.g. "Fe-O") as a string
            - "energy_above_hull": energy above hull in eV/atom as a float.
            - "dft_band_gap": DFT band gap in eV as a float.
            - "dft_mag_density": DFT magnetization density in mu_B/Å^3 as a float.
            - "hhi_score": HHI score as a float.
            - "ml_bulk_modulus": machine learning predicted bulk modulus in GPa as a float
            - "space_group": space group number as an integer.
        The exact choice of properties should be determined from the training conditions of the model
        you wish to use.
    Returns:
        Path to the created extxyz file
    """
    if properties is None:
        properties = []

    if isinstance(mattergen_data, str):
        mattergen_data = Path(mattergen_data)

    if not mattergen_data.exists():
        raise ValueError(f"MatterGen data directory does not exist: {mattergen_data}")

    # Load numpy arrays
    cells = np.load(mattergen_data / "cell.npy")  # (N, 3, 3)
    positions = np.load(mattergen_data / "pos.npy")  # (total_atoms, 3)
    num_atoms = np.load(mattergen_data / "num_atoms.npy")  # (N,)
    atomic_numbers = np.load(mattergen_data / "atomic_numbers.npy")  # (total_atoms,)

    # Load structure IDs if available
    structure_id_file = mattergen_data / "structure_id.npy"
    if structure_id_file.exists():
        structure_ids = np.load(structure_id_file)
    else:
        structure_ids = [f"{i:06d}" for i in range(len(num_atoms))]

    # Load properties if specified
    properties_data = {}
    for prop in properties:
        prop_file = mattergen_data / f"{prop}.json"
        if prop_file.exists():
            with open(prop_file, 'r') as f:
                properties_data[prop] = json.load(f)
        else:
            logging.warning(f"Property file not found: {prop_file}")

    # Reconstruct individual Atoms objects
    atoms_list = []
    atom_offset = 0

    for i in range(len(num_atoms)):
        n_atoms = num_atoms[i]

        # Extract data for this structure
        cell = cells[i]
        pos_frac = positions[atom_offset:atom_offset + n_atoms]
        nums = atomic_numbers[atom_offset:atom_offset + n_atoms]

        # Create Atoms object with fractional coordinates
        atoms = Atoms(
            numbers=nums,
            scaled_positions=pos_frac,  # Use fractional coordinates
            cell=cell,
            pbc=True
        )

        # Add structure ID to info
        atoms.info["structure_id"] = str(structure_ids[i])

        # Add properties to info
        for prop, prop_data in properties_data.items():
            if 'values' in prop_data and i < len(prop_data['values']):
                value = prop_data['values'][i]
                if value is not None:
                    atoms.info[prop] = value

        atoms_list.append(atoms)
        atom_offset += n_atoms

    # Write to extxyz file
    write(ase_extxyz_file, atoms_list)

    logging.info(f"Converted {len(atoms_list)} structures from {mattergen_data} to {ase_extxyz_file}")
    logging.info(f"Total atoms: {len(positions)}")

    return str(ase_extxyz_file)


def _as_path(p: Optional[Union[str, Path]]) -> Optional[Path]:
    # Convert to Path if not None, else return None.
    if p is None:
        return None
    return p if isinstance(p, Path) else Path(p)


### Functions that may require remote execution.
@dflow_remote_execution(
    artifact_inputs={
        "model_root": Path,
        "data_root": Path,
    },
    artifact_outputs={
        "training_scirpt": Path,
        "training_log": Path,
        "trained_model": Path,
        "extra_training_outputs": List[Path],
    },
    parameter_inputs={
        "conditioned_properties": Optional[List[str]],
        "skip": bool,
        "additional_args": Optional[Dict],
        "custom_cmd": Optional[str],
        "env_vars": Optional[Dict],
        "venv_root": Optional[str],
        "output_dir": Union[str, Path],
    },
    op_name="MattergenTrainingOP"
)
def mattergen_train(
        *,
        model_root: Union[str, Path],
        data_root: Union[str, Path],
        conditioned_properties: Optional[List[str]] = None,
        skip: bool = False,
        additional_args: Optional[Dict] = None,
        custom_cmd: Optional[str] = None,
        env_vars: Optional[Dict] = None,
        venv_root: Optional[str] = None,
        output_dir: Union[str, Path] = "./training_outputs",
) -> tuple[str, str, str, List[str]]:
    """Run finetuning for mattergen models from mattergen base models.

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
        Tuple[str, str, str, List[str]]:
            - Path to training script used,
            - Path to training log,
            - Path to trained model, should contain a checkpoint subdirectory and a config.yaml.
            - List of any additional output files generated during training.
    """
    if env_vars is None:
        env_vars = {}

    if model_root is None:
        raise ValueError("Model path must always be provided.")

    if conditioned_properties is None:
        conditioned_properties = ["chemical_system", "energy_above_hull"]
    condition_mode = "_".join(conditioned_properties)

    output_dir = _as_path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    # Skip training and give placeholders.
    if skip:
        placeholder_script = output_dir / "skip_config.yaml"
        placeholder_log = output_dir / "skip_train.log"
        placeholder_script.write_text("skip: true\n")
        placeholder_log.write_text("Training skipped; using provided model.\n")
        model_path = _as_path(model_root) / condition_mode
        return (
            str(placeholder_script),
            str(placeholder_log),
            str(model_path),
            []
        )

    data_train = _as_path(data_root) / "train"
    if not data_train or not data_train.exists():
        raise ValueError("Training data path is required and must exist.")

    if custom_cmd:
        subprocess.run(custom_cmd.split(), env=env_vars, check=True)
    else:
        cmd = prepare_train_command(
            model_root=model_root,
            data_root=data_root,
            conditioned_properties=conditioned_properties,
            additional_args=additional_args,
        )
        # Isolate mattergen within a venv. Recommended way.
        logging.info("========Running mattergen training=======")
        if venv_root:
            # Activate and deactivate the venv within the same command
            # to avoid issues with subprocess and environment variable persistence.
            full_cmd = f"source {venv_root}/bin/activate && {cmd} && deactivate"
            subprocess.run(full_cmd, check=True, shell=True, env=env_vars, executable="/bin/bash")
        else:
            subprocess.run(cmd, check=True, shell=False, env=env_vars)

    outputs_path = Path("outputs")
    latest_dir = max(
        [
            p for p in outputs_path.rglob("*-*-*")
            if p.is_dir() and re.fullmatch(r"\d{2}-\d{2}-\d{2}", p.name)
        ],
        key=lambda d: d.stat().st_mtime,
        default=None,
    )
    train_script = output_dir / "config.yaml"
    log = output_dir / "train.log"
    (output_dir / "checkpoints").mkdir(exist_ok=True, parents=True)
    model_file = output_dir / "checkpoints" / "last.ckpt"

    if latest_dir and latest_dir.is_dir():
        train_script_tmp = latest_dir / "config.yaml"
        if train_script_tmp.exists():
            copy(train_script_tmp, train_script)
        else:
            train_script.write_text("No available training script!")

        if (log_tmp := next(latest_dir.rglob("metrics.csv"), None)):
            copy(log_tmp, log)
        else:
            log.write_text("Not available log file!")

        if (ckpt := next(latest_dir.rglob("last.ckpt"), None)):
            copy(ckpt, model_file)
            return str(train_script), str(log), str(output_dir), []
        else:
            raise RuntimeError(
                "Training failed or aborted! No checkpoint found in the latest directory."
            )

    raise RuntimeError("No output directory found after training.")


@dflow_remote_execution(
    artifact_inputs={
        "model_path": Path,
    },
    artifact_outputs={
        "generated_structures": Path,
        "extra_generation_outputs": List[Path],
    },
    parameter_inputs={
        "results_dir": Path,
        "conditioned_property_values": Optional[Dict],
        "additional_args": Optional[Dict],
        "custom_cmd": Optional[str],
        "env_vars": Optional[Dict],
        "venv_root": Optional[str],
    },
    op_name="MattergenGenerationOP"
)
def mattergen_generate(
        *,
        model_path: Union[str, Path],
        results_dir: Union[str, Path] = "./generated_results",
        conditioned_property_values: Optional[Dict[str, Any]] = None,
        additional_args: Optional[Dict] = None,
        custom_cmd: Optional[str] = None,
        env_vars: Optional[Dict] = None,
        venv_root: Optional[str] = None,
) -> Tuple[str, List[str]]:
    """Inference MatterGen to generate new crystal structure candidates using an existing model.

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
        Tuple[str, List[str]]:
            - Path to the generated crystals in .extxyz format.
            - List of any additional output files generated during generation (e.g. trajectories in .zip format).
    """
    env_vars = env_vars or {}

    if conditioned_property_values is None:
        conditioned_property_values = {}

    model_path = _as_path(model_path)
    if not model_path.exists():
        raise ValueError("Model path must exist for generation.")

    results_dir = _as_path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    if custom_cmd:
        subprocess.run(custom_cmd, check=True)
    else:
        args = {
            "model_path": str(model_path),
            "batch_size": 128,
            "num_batches": 26,
            "diffusion_guidance_factor": 2.0,
            "properties_to_condition_on": conditioned_property_values,
        }
        args.update(additional_args or {})
        cmd = f"mattergen-generate {str(results_dir)}" + " ".join(dict_to_fire_args(args))
        logging.info("========Running mattergen generation=======")
        if venv_root:
            # Activate and deactivate the venv within the same command
            # to avoid issues with subprocess and environment variable persistence.
            full_cmd = f"source {venv_root}/bin/activate && {cmd} && deactivate"
            subprocess.run(full_cmd, check=True, shell=True, env=env_vars, executable="/bin/bash")
        else:
            subprocess.run(cmd, check=True, shell=False, env=env_vars)

    generated_crystals = results_dir / "generated_crystals.extxyz"
    if not generated_crystals.exists():
        raise RuntimeError("No generated crystals found.")

    extra_outputs: List[str] = []
    generated_crystals_traj = results_dir / "generated_trajectories.zip"
    if generated_crystals_traj.exists():
        extra_outputs.append(str(generated_crystals_traj))

    return str(generated_crystals), extra_outputs
