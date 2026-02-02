import argparse
import os
import time
import logging
import random
import traceback
import uuid
from typing import Optional, Union, Dict, Any, List, Tuple, TypedDict
from pathlib import Path
import numpy as np
import subprocess
import sys
import shutil
import json
import glob

# ASE / MD imports used by DPA tools
from ase.io import read, write
from ase.atoms import Atoms
import dpdata
from deepmd.calculator import DP
from dargs import (
    Argument
)

from .utils import run_command, dflow_remote_execution, generate_work_path

DPA1_CONFIG_TEMPLATE = {
    "model": {
        "type_map": [
            "H","He","Li","Be","B","C","N","O","F","Ne","Na","Mg","Al","Si","P","S","Cl","Ar","K","Ca","Sc","Ti","V",
            "Cr","Mn","Fe","Co","Ni","Cu","Zn","Ga","Ge","As","Se","Br","Kr","Rb","Sr","Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd","In","Sn","Sb","Te",
            "I","Xe","Cs","Ba","La","Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu","Hf",
            "Ta","W","Re","Os","Ir","Pt","Au","Hg","Tl","Pb","Bi","Po","At","Rn","Fr","Ra","Ac","Th","Pa","U","Np","Pu","Am","Cm",
            "Bk","Cf","Es","Fm","Md","No","Lr","Rf","Db","Sg","Bh","Hs","Mt","Ds","Rg","Cn","Nh","Fl","Mc","Lv","Ts","Og"
        ],
        "descriptor": {
            "type": "se_atten_v2",
            "sel": "auto",
            "rcut_smth": 0.5,
            "rcut": 8.0,
            "neuron": [
                25,
                50,
                100
            ],
            "resnet_dt": False,
            "axis_neuron": 12,
            "attn": 128,
            "attn_layer": 0,
            "attn_dotr": True,
            "seed": 1111
        },
        "fitting_net": {
            "neuron": [
                240,
                240,
                240
            ],
            "resnet_dt": False,
            "seed": 1111
        }
    },
    "learning_rate": {
        "_comment": "The 'decay_steps' need to be dynamically updated based on the number of batches per epoch.",
        "type": "exp",
        "decay_steps": 10,
        "start_lr": 0.001,
        "stop_lr": 3.51e-08,
    },
    "loss": {
        "type": "ener",
        "start_pref_e": 0.02,
        "limit_pref_e": 1,
        "start_pref_f": 1000,
        "limit_pref_f": 1,
        "start_pref_v": 0,
        "limit_pref_v": 0,
        "_comment": " that's all"
    },
    "training": {
        "training_data": {
            "systems": [
            ],
            "batch_size": "auto",
            "auto_prob": "prob_sys_size"
        },
        "numb_steps": 100,
        "warmup_steps": 0,
        "gradient_max_norm": 5.0,
        "seed": 2912457061,
        "disp_file": "lcurve.out",
        "disp_freq": 100,
        "save_freq": 2000,
    }
}

DPA_CONFIG_TEMPLATE = {
    "_comment": "The template configuration file for training DPA model",
    "model": {
        "_comment": "The 'type map' lists all the elements that will be included in the model.",
        "type_map": [
            "H","He","Li","Be","B","C","N","O","F","Ne","Na","Mg","Al","Si","P","S","Cl","Ar","K","Ca","Sc","Ti","V",
            "Cr","Mn","Fe","Co","Ni","Cu","Zn","Ga","Ge","As","Se","Br","Kr","Rb","Sr","Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd","In","Sn","Sb","Te",
            "I","Xe","Cs","Ba","La","Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu","Hf",
            "Ta","W","Re","Os","Ir","Pt","Au","Hg","Tl","Pb","Bi","Po","At","Rn","Fr","Ra","Ac","Th","Pa","U","Np","Pu","Am","Cm",
            "Bk","Cf","Es","Fm","Md","No","Lr","Rf","Db","Sg","Bh","Hs","Mt","Ds","Rg","Cn","Nh","Fl","Mc","Lv","Ts","Og"
        ],
    },
    "learning_rate": {
        "_comment": "The 'decay_steps' need to be dynamically updated based on the number of batches per epoch.",
        "type": "exp",
        "decay_steps": 10,
        "start_lr": 0.001,
        "stop_lr": 3.51e-08,
    },
    "loss": {
        "type": "ener",
        "start_pref_e": 0.02,
        "limit_pref_e": 1,
        "start_pref_f": 1000,
        "limit_pref_f": 1,
        "start_pref_v": 0,
        "limit_pref_v": 0,
        "_comment": " that's all"
    },
    "training": {
        "training_data": {
            "systems": [
            ],
            "batch_size": "auto",
            "_comment": "There is no need to modify here, training tool would handle it automatically.",
            "auto_prob": "prob_sys_size"
        },
        "_comment": "You do need to update the 'numb_steps' based on your training data size. Usually, it should correspond to 50-100 epochs.",
        "numb_steps": 100,
        "warmup_steps": 0,
        "gradient_max_norm": 5.0,
        "seed": 2912457061,
        "disp_file": "lcurve.out",
        "disp_freq": 100,
        "save_freq": 2000,
    }
}

DPA_CONFIG_DOC: str = """
DPA_CONFIG specification (external config for DPTrain)

Type
- dict (JSON object)

Fields
- numb_steps: int > 0 (default: 100)
  Description: Total number of training steps. Should be compatible with the dataset size (usually corresponds to 50-100 epochs).
- decay_steps: int >= 0 (default: 100)
  Description: Learning-rate decay interval (in steps). Should be compatible with the dataset size (usually corresponds to one epoch).

Normalization & Validation
- Unknown keys are allowed and preserved but not used by DPTrain.
- Keys matching the pattern "_.*" are dropped during normalization (trim_pattern="_*").

Mapping to internal DeepMD (DPA2) input
- training.numb_steps = DPA_CONFIG.numb_steps
- training.decay_steps = DPA_CONFIG.decay_steps
- All other fields are taken from DPA2_CONFIG_TEMPLATE defaults.

Constraints & Notes
- Provide a consistent type_map that matches the dataset species.
- For mixed-type dataset handling, use the command.mixed_type flag (belongs to command, not config).

Examples
Minimal:
{
}

Explicit:
{
  "numb_steps": 400000,
  "decay_steps": 20000,
}
"""

DPA_CONFIG_JSON_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": True,
    "properties": {
        "numb_steps": {
            "type": "integer",
            "minimum": 1,
            "default": 100,
            "description": "Total number of training steps."
        },
        "decay_steps": {
            "type": "integer",
            "minimum": 0,
            "default": 100,
            "description": "Learning-rate decay interval."
        },
    },
    "required": []
}

DPA_COMMAND_DOC: str = """
DPA_COMMAND specification (external command options for DPTrain)

Type
- dict (JSON object)

Fields
- command: string (default: "dp")
    Description: Executable name or absolute path for DeepMD binary. Typically "dp".

- impl: string in {"tensorflow", "pytorch"} (default: "pytorch")
    Description: Training backend implementation. Always choose "pytorch" unless you have a specific reason to use TensorFlow.
    Alias: backend (deprecated; refer impl)

- finetune_args: string (default: "")
    Description: Extra arguments appended to the finetune command (raw CLI fragment).

- multitask: boolean (default: false)
    Description: Enable multitask training.

- head: string | null (default: null)
    Description: Head name to use in multitask training when multitask=true; otherwise unused.

- train_args: string (default: "")
    Description: Extra arguments appended to "dp train" (raw CLI fragment).

- finetune_mode: boolean (default: false)
    Description: Whether to run in finetune mode (set to true whenever a base_model_path is provided or fine-tuning is specified).

- mixed_type: boolean (default: false)
    Description: Whether to export/consume dataset in DeepMD mixed-type (deepmd/npy/mixed) format.

Normalization & Validation
- Unknown keys are allowed and preserved but not used by DPTrain.
- Keys matching the pattern "_.*" are dropped during normalization (trim_pattern="_*").
- Types are coerced when possible by the dargs normalizer; otherwise a validation error may be raised.

Notes
- "backend" is accepted as an alias for "impl" during normalization but may be removed in outputs.
- finetune_args/train_args are raw CLI strings; provide carefully.
"""

DPA_COMMAND_JSON_SCHEMA: Dict[str, Any] = {
        "type": "object",
        "additionalProperties": True,
        "properties": {
                "command": {
                        "type": "string",
                        "default": "dp",
                        "description": "Executable name or path for DeepMD (e.g., 'dp').",
                },
                "impl": {
                        "type": "string",
                        "enum": ["tensorflow", "pytorch"],
                        "default": "pytorch",
                        "description": "Training backend implementation.",
                },
                "finetune_args": {
                        "type": "string",
                        "default": "",
                        "description": "Extra arguments for finetuning (raw CLI fragment).",
                },
                "multitask": {
                        "type": "boolean",
                        "default": False,
                        "description": "Enable multitask training.",
                },
                "head": {
                        "type": ["string", "null"],
                        "default": None,
                        "description": "Head name for multitask mode; otherwise unused.",
                },
                "train_args": {
                        "type": "string",
                        "default": "",
                        "description": "Extra arguments for 'dp train' (raw CLI fragment).",
                },
                "finetune_mode": {
                        "type": "boolean",
                        "default": False,
                        "description": "Run in finetune mode (use init model when applicable).",
                },
                "mixed_type": {
                        "type": "boolean",
                        "default": False,
                        "description": "Use DeepMD mixed-type (deepmd/npy/mixed) dataset format.",
                },
        },
        "required": []
}


TRAIN_SCRIPT_NAME = "input.json"
TRAIN_LOG_FILE = "train.log"


def dpa_training_meta() -> Dict[str, Any]:
    return {
        "version": "v3.0",
        "description": "The default training parameters and command for dpa models",
        "config": {"schema": DPA_CONFIG_JSON_SCHEMA, "doc": DPA_CONFIG_DOC},
        "command": {"schema": DPA_COMMAND_JSON_SCHEMA, "doc": DPA_COMMAND_DOC},
    }


def dpa_command_args() -> List[Argument]:
    doc_command = "The command for DP, 'dp' for default"
    doc_impl = "The implementation/backend of DP. It can be 'tensorflow' or 'pytorch'. 'tensorflow' for default."
    doc_finetune_args = "Extra arguments for finetuning"
    doc_multitask = "Do multitask training"
    doc_head = "Head to use in the multitask training"
    doc_train_args = "Extra arguments for dp train"
    doc_finetune_mode = "Whether to run in finetune mode"
    doc_mixed_type = "Whether to use mixed type system for training"
    return [
        Argument("command", str, optional=True, default="dp", doc=doc_command),
        Argument("impl", str, optional=True, default="pytorch", doc=doc_impl, alias=["backend"]),
        Argument("finetune_args", str, optional=True, default="", doc=doc_finetune_args),
        Argument("multitask", bool, optional=True, default=False, doc=doc_multitask),
        Argument("head", str, optional=True, default=None, doc=doc_head),
        Argument("train_args", str, optional=True, default="", doc=doc_train_args),
        Argument("finetune_mode", bool, optional=True, default=False, doc=doc_finetune_mode),
        Argument("mixed_type", bool, optional=True, default=False, doc=doc_mixed_type),
    ]


def normalize_dpa_command(data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    ta = dpa_command_args()
    base = Argument("base", dict, ta)
    normalized = base.normalize_value(data or {}, trim_pattern="_*")
    base.check_value(normalized, strict=True)
    return normalized


def dpa_config_args() -> List[Argument]:
    return [
        Argument("numb_steps", int, optional=True, default=100, doc="Number of training steps"),
        Argument("decay_steps", int, optional=True, default=100, doc="Decay steps for learning rate decay"),
    ]


def normalize_dpa_config(data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    ta = dpa_config_args()
    base = Argument("base", dict, ta)
    normalized = base.normalize_value(data or {}, trim_pattern="_*")
    base.check_value(normalized, strict=False)
    return normalized


def _set_desc_seed(desc: Dict[str, Any]) -> None:
    if desc["type"] == "hybrid":
        for sub in desc["list"]:
            _set_desc_seed(sub)
    elif desc["type"] not in ["dpa1", "dpa2"]:
        desc["seed"] = random.randrange(sys.maxsize) % (2**32)


def _script_rand_seed(input_dict: Dict[str, Any]) -> Dict[str, Any]:
    jtmp = json.loads(json.dumps(input_dict))
    if "model_dict" in jtmp["model"]:
        for d in jtmp["model"]["model_dict"].values():
            if isinstance(d["descriptor"], str):
                _set_desc_seed(jtmp["model"]["shared_dict"][d["descriptor"]])
            d["fitting_net"]["seed"] = random.randrange(sys.maxsize) % (2**32)
    else:
        _set_desc_seed(jtmp["model"]["descriptor"])
        jtmp["model"]["fitting_net"]["seed"] = random.randrange(sys.maxsize) % (2**32)
    jtmp["training"]["seed"] = random.randrange(sys.maxsize) % (2**32)
    return jtmp


def build_training_template(config: Dict[str, Any], finetune_mode: bool) -> Dict[str, Any]:
    template = (DPA_CONFIG_TEMPLATE if finetune_mode else _script_rand_seed(DPA1_CONFIG_TEMPLATE))
    template = json.loads(json.dumps(template))
    template["training"]["numb_steps"] = config.get("numb_steps", 100)
    template["learning_rate"]["decay_steps"] = config.get("decay_steps", 100)
    return template


def write_data_to_input_script(
    idict: Dict[str, Any],
    train_data: List[Path],
    auto_prob_str: str = "prob_sys_size",
    valid_data: Optional[List[Path]] = None,
) -> Dict[str, Any]:
    odict = json.loads(json.dumps(idict))
    odict["training"]["training_data"]["systems"] = [str(p) for p in train_data]
    odict["training"]["training_data"].setdefault("batch_size", "auto")
    odict["training"]["training_data"]["auto_prob"] = auto_prob_str
    if valid_data is None:
        odict["training"].pop("validation_data", None)
    else:
        odict["training"]["validation_data"] = {
            "systems": [str(p) for p in valid_data],
            "batch_size": 1,
        }
    return odict


def ase2dpdata(atoms: Atoms, labeled: bool = False) -> dpdata.System:
    symbols = atoms.get_chemical_symbols()
    atom_names = list(dict.fromkeys(symbols))
    atom_numbs = [symbols.count(symbol) for symbol in atom_names]
    atom_types = np.array([atom_names.index(symbol) for symbol in symbols]).astype(int)
    cells = atoms.cell.array
    coords = atoms.get_positions()
    info_dict: Dict[str, Any] = {
        "atom_names": atom_names,
        "atom_numbs": atom_numbs,
        "atom_types": atom_types,
        "cells": np.array([cells]),
        "coords": np.array([coords]),
        "orig": np.zeros(3),
        "nopbc": not np.any(atoms.get_pbc()),
    }
    if labeled:
        info_dict["energies"] = np.array([atoms.get_potential_energy()])
        info_dict["forces"] = np.array([atoms.get_forces()])
        if "virial" in atoms.arrays:
            info_dict["virial"] = np.array([atoms.arrays["virial"]])
        return dpdata.LabeledSystem.from_dict({"data": info_dict})
    return dpdata.System.from_dict({"data": info_dict})


def ase2multisys(atoms_list: List[Atoms], labeled: bool = False) -> dpdata.MultiSystems:
    ms = dpdata.MultiSystems()
    for atoms in atoms_list:
        ms.append(ase2dpdata(atoms, labeled=labeled))
    return ms


def _make_train_command(
    dp_command: List[str],
    train_script_name: str,
    init_model: Optional[Path],
    finetune_mode: bool,
    finetune_args: str,
    train_args: str = "",
) -> List[str]:
    command: List[str]
    init_model_str = str(init_model) if init_model else ""
    if finetune_mode and init_model and os.path.isfile(init_model_str):
        command = (
            dp_command
            + ["train", train_script_name, "--finetune", init_model_str]
            + finetune_args.split()
            + ["--use-pretrain-script"]
        )
        logging.info(f"Finetune mode: using init model {init_model_str}")
    else:
        command = dp_command + ["train", train_script_name]
        logging.info("No available checkpoint found. Training from scratch.")
    command += train_args.split()
    return command


def _ensure_path_list(value: Optional[Union[List[Path], Path]]) -> List[Path]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [Path(p) for p in value]
    return [Path(value)]


def _load_atoms_from_paths(paths: List[Path]) -> List[Atoms]:
    frames: List[Atoms] = []
    for path in paths:
        frames.extend(read(str(path), index=":"))
    return frames


def _export_dpdata(atoms: List[Atoms], out_dir: Path, mixed_type: bool) -> List[Path]:
    if not atoms:
        raise ValueError("No structures found for dataset export.")
    out_dir.mkdir(parents=True, exist_ok=True)
    multisys = ase2multisys(atoms, labeled=True)
    fmt = "deepmd/npy/mixed" if mixed_type else "deepmd/npy"
    multisys.to(fmt, str(out_dir))
    return _get_system_path(str(out_dir))


def _prepare_training_datasets(
    workdir: Path,
    train_data: List[Path],
    valid_data: List[Path],
    mixed_type: bool,
) -> Tuple[List[Path], Optional[List[Path]]]:
    if not train_data:
        raise ValueError("train_data must contain at least one structure file.")
    train_atoms = _load_atoms_from_paths(train_data)
    train_paths = _export_dpdata(train_atoms, workdir / "train_data", mixed_type)
    valid_paths_refs: Optional[List[Path]] = None
    if valid_data:
        valid_atoms = _load_atoms_from_paths(valid_data)
        valid_paths_refs = _export_dpdata(valid_atoms, workdir / "valid_data", mixed_type)
    return train_paths, valid_paths_refs


@dflow_remote_execution(
    artifact_inputs={
        "train_data": List[Path],
        "valid_data": List[Path],
    },
    artifact_outputs={
        "model": Path,
        "log": Path,
    },
    parameter_inputs={
        "workdir": Path,
        "config": Dict[str, Any],
        "command": Dict[str, Any],
        "model_path": Path,  # Changed to parameter since it can be None
    },
    parameter_outputs={
        "message": str,
        "status": str,
    },
    op_name="DPTrainingOP"
)
def _run_dp_training(
    workdir: Path,
    config: Dict[str, Any],
    command: Dict[str, Any],
    train_data: List[Path],
    valid_data: List[Path],
    model_path: Optional[Path],
) -> Dict[str, Any]:
    """
    Run DeepPotential training (supports both local and remote execution via decorator).
    
    Returns:
        Dict with keys:
            - model: Path to trained model file
            - log: Path to training log file
            - message: Status/error message
    """
    try:
        workdir.mkdir(parents=True, exist_ok=True)
        dp_command = command.get("command", "dp").split()
        impl = command.get("impl", "tensorflow")
        if impl not in {"tensorflow", "pytorch"}:
            raise ValueError("command.impl must be either 'tensorflow' or 'pytorch'.")
        if impl == "pytorch":
            dp_command.append("--pt")
            
        #raise NotImplementedError("The current DP training tool only supports TensorFlow backend.")

        finetune_mode = bool(command.get("finetune_mode", False))
        finetune_args = command.get("finetune_args", "")
        train_args = command.get("train_args", "")
        mixed_type = bool(command.get("mixed_type", False))

        train_paths, valid_paths = _prepare_training_datasets(workdir, train_data, valid_data, mixed_type)
        auto_prob_str = "prob_sys_size"
        template = build_training_template(config, finetune_mode)
        config_payload = write_data_to_input_script(template, train_paths, auto_prob_str, valid_paths)
        config_payload["training"]["disp_file"] = "lcurve.out"

        train_script_path = workdir / TRAIN_SCRIPT_NAME
        with open(train_script_path, "w") as fp:
            json.dump(config_payload, fp, indent=4)

        init_model_path: Optional[Path] = None
        if model_path is not None:
            resolved_model = Path(model_path).expanduser().resolve()
            if finetune_mode:
                init_model_path = workdir / resolved_model.name
                if init_model_path.exists():
                    if init_model_path.is_symlink() or init_model_path.is_file():
                        init_model_path.unlink()
                init_model_path.symlink_to(resolved_model)
            else:
                init_model_path = resolved_model

        command_list = _make_train_command(dp_command, TRAIN_SCRIPT_NAME, init_model_path, finetune_mode, finetune_args, train_args)
        log_file_path = workdir / TRAIN_LOG_FILE
        with open(log_file_path, "w") as fplog:
            ret, out, err = run_command(
            command_list,
            raise_error=False,
            try_bash=False,
            interactive=False,
            cwd=str(workdir),
        )
            if ret != 0:
                logging.error(
                "".join(
                    (
                        "dp train failed\n",
                        "out msg: ",
                        out,
                        "\n",
                        "err msg: ",
                        err,
                        "\n",
                    )
                )
            )
                raise RuntimeError(f"dp train failed (see log)\nstdout:\n{out}\nstderr:\n{err}")
            fplog.write("#=================== train std out ===================\n")
            fplog.write(out)
            fplog.write("#=================== train std err ===================\n")
            fplog.write(err)

        compat_file = workdir / "input_v2_compat.json"
        if finetune_mode and compat_file.exists():
            shutil.copy2(compat_file, train_script_path)

        if impl == "pytorch":
            model_file = workdir / "model.ckpt.pt"
        else:
            ret, out, err = run_command(["dp", "freeze", "-o", "frozen_model.pb"], raise_error=False, cwd=str(workdir))
            if ret != 0:
                logging.error(
                    "".join(
                    (
                        "dp freeze failed\n",
                        "out msg: ",
                        out,
                        "\n",
                        "err msg: ",
                        err,
                        "\n",
                    )
                )
            )
                raise RuntimeError(f"dp freeze failed: {err}")
            model_file = workdir / "frozen_model.pb"

        return {
            "status": "success",
            "model": model_file.resolve(),
            "log": log_file_path.resolve(),
            "message": "Training completed successfully"
        }
    except Exception as e:
        logging.error(f"Error in training: {str(e)}")
        return {
            "status": "error",
            "model": workdir / "model.ckpt.pt",
            "log": workdir / TRAIN_LOG_FILE,
            "message": f"Training failed: {e}"
        }

@dflow_remote_execution(
    artifact_inputs={
        "model_file": Path,
        "test_data": List[Path],
    },
    artifact_outputs={
        "energy_files": List[Path],
        "energy_per_atom_files": List[Path],
        "force_files": List[Path],
    },
    parameter_inputs={
        "workdir": Path,
        "head": str,
    },
    parameter_outputs={
        "test_metrics": Dict[str, Any],
        "message": str,
        "status": str,
    },
    op_name="DPModelTestOP"
)
def _evaluate_trained_model(
    workdir: Path, 
    model_file: Path, 
    test_data: List[Path],
    head: Optional[str] = None
) -> Dict[str, Any]:
    """
    Evaluate a trained Deep Potential model on test datasets (supports remote execution).
    
    Returns:
        Dict with keys:
            - energy_files: List of paths to energy comparison files (label vs prediction)
            - energy_per_atom_files: List of paths to per-atom energy comparison files
            - force_files: List of paths to force comparison files
            - test_metrics: Dict mapping dataset index to metrics (MAE, RMSE for energy and forces)
            - message: Status/error message
    """
    try:
        from deepmd.calculator import DP as DeepmdCalculator  # type: ignore
        out_dir = workdir / "test_output"
        out_dir.mkdir(parents=True, exist_ok=True)
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")
        calc = DeepmdCalculator(model=str(model_file), head=head)
        results: Dict[str, Any] = {}
    
        for idx, data_path in enumerate(test_data):
            atoms_ls: List[Atoms] = read(str(data_path), index=":")
            pred_e: List[float] = []
            lab_e: List[float] = []
            pred_f: List[float] = []
            lab_f: List[float] = []
            atom_num: List[int] = []
            for atoms in atoms_ls:
                lab_e.append(atoms.get_potential_energy())
                lab_f.append(atoms.get_forces().flatten())
                atoms.calc = calc
                pred_e.append(atoms.get_potential_energy())
                pred_f.append(atoms.get_forces().flatten())
                atom_num.append(atoms.get_number_of_atoms())

            atom_num_arr = np.array(atom_num)
            pred_e_arr = np.array(pred_e)
            lab_e_arr = np.array(lab_e)
            pred_e_atom = pred_e_arr / atom_num_arr
            lab_e_atom = lab_e_arr / atom_num_arr
            pred_f_arr = np.hstack(pred_f)
            lab_f_arr = np.hstack(lab_f)

            np.savetxt(
                str(out_dir / (f"test_{idx:02d}_.energy.txt")),
                np.column_stack((lab_e_arr, pred_e_arr)),
                header='',
                comments='#',
                fmt="%.6f",
            )
            np.savetxt(
                str(out_dir / (f"test_{idx:02d}_.energy_per_atom.txt")),
                np.column_stack((lab_e_atom, pred_e_atom)),
                header='',
                comments='#',
                fmt="%.6f",
            )
            np.savetxt(
                str(out_dir / (f"test_{idx:02d}_.force.txt")),
                np.column_stack((lab_f_arr, pred_f_arr)),
                header='',
                comments='#',
                fmt="%.6f",
            )

            metrics = {
                "mae_e": _mae(pred_e_arr, lab_e_arr),
                "rmse_e": _rmse(pred_e_arr, lab_e_arr),
                "mae_e_atom": _mae(pred_e_atom, lab_e_atom),
                "rmse_e_atom": _rmse(pred_e_atom, lab_e_atom),
                "mae_f": _mae(pred_f_arr, lab_f_arr) if lab_f_arr.size else float('nan'),
                "rmse_f": _rmse(pred_f_arr, lab_f_arr) if lab_f_arr.size else float('nan'),
                "n_frames": float(len(atoms_ls)),
        }
            logging.info(f"Test completed on {len(atoms_ls)} frames. Metrics: {metrics}")
            results[f"{idx:02d}"] = metrics

        # Collect all file paths
        energy_files = sorted(out_dir.glob("*_.energy.txt"))
        energy_per_atom_files = sorted(out_dir.glob("*_.energy_per_atom.txt"))
        force_files = sorted(out_dir.glob("*_.force.txt"))
        
        return {
            "status": "success",
            "energy_files": [f.resolve() for f in energy_files],
            "energy_per_atom_files": [f.resolve() for f in energy_per_atom_files],
            "force_files": [f.resolve() for f in force_files],
            "test_metrics": results,
            "message": f"Model evaluation completed on {len(test_data)} dataset(s)"
            }
    except Exception as e:
        logging.error(f"Error in model evaluation: {str(e)}")
        return {
            "status": "error",
            "energy_files": [],
            "energy_per_atom_files": [],
            "force_files": [],
            "test_metrics": {},
            "message": f"Model evaluation failed: {e}"
            }

def _get_system_path(
    data_dir:Union[str,Path]
    ):
    return [Path(ii).parent for ii in glob.glob(str(data_dir) + "/**/type.raw",recursive=True)]


def _mae(a, b):
    mask = np.isfinite(a) & np.isfinite(b)
    return float(np.mean(np.abs(a[mask] - b[mask]))) if mask.any() else float('nan')

def _rmse(a, b):
    mask = np.isfinite(a) & np.isfinite(b)
    return float(np.sqrt(np.mean((a[mask] - b[mask]) ** 2))) if mask.any() else float('nan')


@dflow_remote_execution(
    artifact_inputs={
        "structure_path": List[Path],
        "model_path": Path,
    },
    artifact_outputs={
        "labeled_data": Path,
    },
    parameter_inputs={
        "head": str,
    },
    parameter_outputs={
        "message": str,
    },
    op_name="DPInferenceOP",
)
def model_inference(
    structure_path: Union[List[Path], Path],
    model_path: Optional[Path] = None,
    head: Optional[str] = None,
) -> Dict[str, Any]:
    """Calculate energy and force for given structures using a Deep Potential model.

    Parameters
    - structure_path: List[Path] | Path
        Path(s) to structure file(s) (extxyz/xyz/vasp/...). Can be a multi-frame file or a list of files.
    - model_path: Path
        Model file(s) or URL(s) for ML calculators. 
    - head (str, optional): For pretrained DPA multi-head models, an available head should be provided. 
        The head is defaulted to "MP_traj_v024_alldata_mixu" for multi-task model if not specified. 

    Returns
    - Dict[str, Any]
        Dictionary containing:
        - labeled_data: Path to extxyz file with structures and computed energy, forces, and stress
        - message: Status message
        
    Note:
        To extract energy and force values to separate files, use the `inspect_structure` tool
        with export_energy=True and export_forces=True flags.
    """
    try:
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
        
        labeled_data = "ase_results.extxyz"
        write(labeled_data, atoms_ls, format="extxyz")
        
        result = {
            "status": "success",
            "labeled_data": str(Path(labeled_data).resolve()),
            "message": f"ASE calculation completed for {len(atoms_ls)} structures. Use inspect_structure tool to extract properties."
        }
    except Exception as e:
        logging.error(f"Error in ase_calculation: {str(e)}")
        result={
            "status": "error",
            "labeled_data": None,
            "message": f"ASE calculation failed: {e}"
            }
    return result   