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
import shlex
import selectors
from jsonschema import validate, ValidationError
import shutil
import json
import glob
from dotenv import load_dotenv

# ASE / MD imports used by DPA tools
from ase.io import read, write
from ase.atoms import Atoms
from ase.md.andersen import Andersen
from ase.md.langevin import Langevin
from ase.md.nose_hoover_chain import NoseHooverChainNVT
from ase.md.npt import NPT
from ase.md.nvtberendsen import NVTBerendsen
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution,Stationary, ZeroRotation
from ase.md.verlet import VelocityVerlet
from ase.optimize import BFGS
from ase.constraints import ExpCellFilter
from ase import units
from deepmd.calculator import DP
import dpdata
from dargs import (
    Argument
)

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


def _run_dp_training(
    workdir: Path,
    config: Dict[str, Any],
    command: Dict[str, Any],
    train_data: List[Path],
    valid_data: List[Path],
    model_path: Optional[Path],
) -> Tuple[Path, Path, str]:
    workdir.mkdir(parents=True, exist_ok=True)
    dp_command = command.get("command", "dp").split()
    impl = command.get("impl", "tensorflow")
    if impl not in {"tensorflow", "pytorch"}:
        raise ValueError("command.impl must be either 'tensorflow' or 'pytorch'.")
    if impl == "pytorch":
        dp_command.append("--pt")

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
            raise RuntimeError("dp freeze failed")
        model_file = workdir / "frozen_model.pb"

    return model_file.resolve(), log_file_path.resolve(), err


def _evaluate_trained_model(workdir: Path, model_file: Path, test_data: List[Path]) -> List[Dict[str, Any]]:
    if not test_data:
        return []
    try:
        from deepmd.calculator import DP as DeepmdCalculator  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "Failed to import deepmd ASE calculator (deepmd.calculator.DP). "
            "Install deepmd-kit with ASE support or run evaluation externally."
        ) from exc

    out_dir = workdir / "test_output"
    out_dir.mkdir(parents=True, exist_ok=True)
    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_file}")
    calc = DeepmdCalculator(model=str(model_file))
    results: List[Dict[str, Any]] = []
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
            "system_idx": f"{idx:02d}",
            "mae_e": _mae(pred_e_arr, lab_e_arr),
            "rmse_e": _rmse(pred_e_arr, lab_e_arr),
            "mae_e_atom": _mae(pred_e_atom, lab_e_atom),
            "rmse_e_atom": _rmse(pred_e_atom, lab_e_atom),
            "mae_f": _mae(pred_f_arr, lab_f_arr) if lab_f_arr.size else float('nan'),
            "rmse_f": _rmse(pred_f_arr, lab_f_arr) if lab_f_arr.size else float('nan'),
            "n_frames": float(len(atoms_ls)),
        }
        logging.info(f"Test completed on {len(atoms_ls)} frames. Metrics: {metrics}")
        results.append(metrics)

    return results

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

def _log_progress(atoms, dyn):
    """Log simulation progress"""
    epot = atoms.get_potential_energy()
    ekin = atoms.get_kinetic_energy()
    temp = ekin / (1.5 * len(atoms) * units.kB)
    logging.info(f"Step: {dyn.nsteps:6d}, E_pot: {epot:.3f} eV, T: {temp:.2f} K")

def _run_md_stage(atoms, stage, save_interval_steps, traj_file, seed, stage_id):
    """Run a single MD simulation stage"""
    temperature_K = stage.get('temperature_K', None)
    pressure = stage.get('pressure', None)
    mode = stage['mode']
    runtime_ps = stage['runtime_ps']
    timestep_ps = stage.get('timestep', 0.0005)  # default: 0.5 fs
    tau_t_ps = stage.get('tau_t', 0.01)         # default: 10 fs
    tau_p_ps = stage.get('tau_p', 0.1)          # default: 100 fs

    timestep_fs = timestep_ps * 1000  # convert to fs
    total_steps = int(runtime_ps * 1000 / timestep_fs)

    # Initialize velocities if first stage with temperature
    if stage_id == 1 and temperature_K is not None:
        MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_K, 
                                rng=np.random.RandomState(seed))
        Stationary(atoms)
        ZeroRotation(atoms)

    # Choose ensemble
    if mode == 'NVT' or mode == 'NVT-NH':
        # Use NoseHooverChain for NVT by default
        dyn = NoseHooverChainNVT(
            atoms,
            timestep=timestep_fs * units.fs,
            temperature_K=temperature_K,
            tdamp=tau_t_ps * 1000 * units.fs
        )
    elif mode == 'NVT-Berendsen':
        dyn = NVTBerendsen(
            atoms,
            timestep=timestep_fs * units.fs,
            temperature_K=temperature_K,
            taut=tau_t_ps * 1000 * units.fs
        )
    elif mode == 'NVT-Andersen':
        dyn = Andersen(
            atoms,
            timestep=timestep_fs * units.fs,
            temperature_K=temperature_K,
            friction=1.0 / (tau_t_ps * 1000 * units.fs),
            rng=np.random.RandomState(seed)
        )
    elif mode == 'NVT-Langevin' or mode == 'Langevin':
        dyn = Langevin(
            atoms,
            timestep=timestep_fs * units.fs,
            temperature_K=temperature_K,
            friction=1.0 / (tau_t_ps * 1000 * units.fs),
            rng=np.random.RandomState(seed)
        )
    elif mode == 'NPT-aniso' or mode == 'NPT-tri':
        if mode == 'NPT-aniso':
            mask = np.eye(3, dtype=bool)
        elif mode == 'NPT-tri':
            mask = None
        else:
            raise ValueError(f"Unknown NPT mode: {mode}")

        if pressure is None:
            raise ValueError("Pressure must be specified for NPT simulations")

        dyn = NPT(
            atoms,
            timestep=timestep_fs * units.fs,
            temperature_K=temperature_K,
            externalstress=pressure * units.GPa,
            ttime=tau_t_ps * 1000 * units.fs,
            pfactor=tau_p_ps * 1000 * units.fs,
            mask=mask
        )
    elif mode == 'NVE':
        dyn = VelocityVerlet(
            atoms,
            timestep=timestep_fs * units.fs
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Prepare trajectory file
    os.makedirs(os.path.dirname(traj_file), exist_ok=True)
    if os.path.exists(traj_file):
        os.remove(traj_file)

    def _write_frame():
        """Write current frame to trajectory"""
        results = atoms.calc.results
        energy = results.get("energy", atoms.get_potential_energy())
        forces = results.get("forces", atoms.get_forces())
        stress = results.get("stress", atoms.get_stress(voigt=False))

        if np.isnan(energy).any() or np.isnan(forces).any() or np.isnan(stress).any():
            raise ValueError("NaN detected in simulation outputs. Aborting trajectory write.")

        new_atoms = atoms.copy()
        new_atoms.info["energy"] = energy
        new_atoms.arrays["force"] = forces
        if "occupancy" in atoms.info:
            del atoms.info["occupancy"]
        if "spacegroup" in atoms.info:
            del atoms.info["spacegroup"] 

        write(traj_file, new_atoms, format="extxyz", append=True)

    # Attach callbacks
    dyn.attach(_write_frame, interval=save_interval_steps)
    dyn.attach(lambda: _log_progress(atoms, dyn), interval=100)

    logging.info(f"[Stage {stage_id}] Starting {mode} simulation: T={temperature_K} K"
                + (f", P={pressure} GPa" if pressure is not None else "")
                + f", steps={total_steps}, dt={timestep_ps} ps")

    # Run simulation
    dyn.run(total_steps)
    logging.info(f"[Stage {stage_id}] Finished simulation. Trajectory saved to: {traj_file}\n")
    #traj_file=Path(traj_file).absolute()

    return atoms#, traj_file

def _run_md_pipeline(atoms, stages, save_interval_steps=100, traj_prefix='traj', traj_dir='trajs_files', seed=42):
    """Run multiple MD stages sequentially"""
    for i, stage in enumerate(stages):
        mode = stage['mode']
        T = stage.get('temperature_K', 'NA')
        P = stage.get('pressure', 'NA')

        tag = f"stage{i+1}_{mode}_{T}K"
        if P != 'NA':
            tag += f"_{P}GPa"
        traj_file = os.path.join(traj_dir, f"{traj_prefix}_{tag}.extxyz")

        atoms = _run_md_stage(
            atoms=atoms,
            stage=stage,
            save_interval_steps=save_interval_steps,
            traj_file=traj_file,
            seed=seed,
            stage_id=i + 1
        )

    return atoms

@mcp.tool()
def run_molecular_dynamics(
    initial_structure: Path,
    stages: List[Dict],
    model_path: Optional[Path]= None,
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
    # Create output directories
    try:
        work_path=Path(generate_work_path())
        work_path = work_path.expanduser().resolve()
        work_path.mkdir(parents=True, exist_ok=True)
    
        traj_dir = work_path / "trajs_files"
        traj_dir.mkdir(parents=True, exist_ok=True)
        log_file = work_path / "md_simulation.log"
    
        # Read initial structure
        atoms_ls = read(initial_structure,index=':')

        calc=DP(
            model=model_path, 
            head=head
            )  
    
        #final_structures_ls=[]
        for idx,atoms in enumerate(atoms_ls):
            atoms.calc = calc
        # Run MD pipeline
            _ = _run_md_pipeline(
                atoms=atoms,
                stages=stages,
                save_interval_steps=save_interval_steps,
                traj_prefix=traj_prefix+("_%03d"%idx),
                traj_dir=str(traj_dir),
                seed=seed
            )
    
        traj_list = sorted(
            p.resolve()  # use resolve(strict=True) if you want to fail on broken links
            for p in Path(traj_dir).glob("*.extxyz")
            )
        result = {
            "status": "success",
            "message": "Molecular dynamics simulation completed successfully.",
            "trajectory_list": traj_list,
            "log_file": log_file,
        }

    except Exception as e:
        logging.error(f"Molecular dynamics simulation failed: {str(e)}", exc_info=True)
        result = {
            "status": "error",
            "message": f"Molecular dynamics simulation failed: {str(e)}",
            "trajectory_list": [],
            "log_file": Path(""),
        }
    return result






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