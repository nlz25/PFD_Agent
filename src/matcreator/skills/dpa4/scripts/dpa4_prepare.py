#!/usr/bin/env python3
"""dpa4_prepare.py — Preparation-phase script for DPA4 (SeZM) finetuning jobs.

Converts raw structure data (xyz / extxyz / POSCAR / …) into deepmd/npy format
and writes an input.json ready for ``dp --pt train`` with version-specific DPA4
configuration.

DPA4 is currently in early stage. This script targets the **neo** version by default.
Future versions (air, plus, pro, …) will require their own matching model and parameters.
Each model version has a one-to-one correspondence with its input.json — do not mix
parameters across versions.

DPA4 jobs run exclusively on the Bohrium platform — there is no local execution.

Sub-commands
------------
  prepare-finetune   Single-task finetune of a pretrained DPA4 model
  convert-data       Convert ASE-readable files to deepmd/npy for ``dp test``

After running any sub-command the ``--workdir`` directory contains:

  input.json                  Training configuration consumed by ``dp --pt train``
  train_data/                 deepmd/npy training split
  test_data/                  deepmd/npy test split (for dp test evaluation)
  <model>                     Copy of the DPA4 pretrained model file

Execution (remote on Bohrium)
-----------------------------
  cd <workdir>

  # Finetune (--skip-neighbor-stat is required for train only):
  dp --pt train input.json --skip-neighbor-stat --finetune <model_dir> > train_log 2>&1

  # Freeze:
  dp --pt freeze -c model.ckpt.pt -o frozen

  # Test (frozen model):
  dp --pt test -m frozen.pt2 -s <test_data_dir> -d result-test -l log-test

  # Test (pretrained model directory — for zero-shot evaluation):
  dp --pt test -m <model_dir> -s <test_data_dir> -d result-test -l log-test

Data conversion for dp test
----------------------------
  python dpa4_prepare.py convert-data \\
      --data  test.extxyz            \\
      --outdir <out_dir>             \\
      [--mixed_type]                 \\
      [--nframes 100]

  # Then run dp test on each system dir printed in the JSON output:
  dp --pt test -m frozen.pt2 -s <system_dir> [-n <nframes>]
"""

import argparse
import glob
import json
import logging
import random
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from ase.atoms import Atoms
from ase.io import read
import dpdata

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Full periodic-table element list — used as universal type_map
# ---------------------------------------------------------------------------
ALL_TYPES: List[str] = [
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
    "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni",
    "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr",
    "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd",
    "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe",
    "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd",
    "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn",
    "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm",
    "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr",
    "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn",
    "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
]

# ---------------------------------------------------------------------------
# DPA4 version → input.json template mapping
#
# Each model version has its own matching parameters. Do NOT mix across versions.
# To add a new version (e.g. "air"), add a new entry here with the correct template.
# ---------------------------------------------------------------------------
_DPA4_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "neo": {
        "model": {
            "type": "SeZM",
            "descriptor": {
                "type": "SeZM",
                "sel": 416,
                "rcut": 6.0,
                "env_exp": [7, 5],
                "channels": 64,
                "n_radial": 16,
                "radial_mlp": [0],
                "use_env_seed": True,
                "random_gamma": True,
                "lmax": 3,
                "mmax": 1,
                "n_blocks": 2,
                "so2_layers": 3,
                "so2_norm": False,
                "so2_attn_res": "none",
                "radial_so2_mode": "degree_channel",
                "radial_so2_rank": 1,
                "n_focus": 1,
                "focus_dim": 0,
                "n_atten_head": 1,
                "atten_f_mix": False,
                "atten_v_proj": False,
                "atten_o_proj": False,
                "ffn_neurons": 0,
                "grid_mlp": False,
                "ffn_blocks": 1,
                "sandwich_norm": [False, True, True, False],
                "mlp_bias": False,
                "layer_scale": False,
                "full_attn_res": "none",
                "block_attn_res": "none",
                "s2_activation": [False, True],
                "lebedev_quadrature": True,
                "activation_function": "silu",
                "glu_activation": True,
                "use_amp": False,
                "precision": "float32",
                "seed": 42,
            },
            "fitting_net": {
                "neuron": [0],
                "activation_function": "silu",
                "precision": "float32",
                "seed": 42,
            },
            "use_compile": False,
            "enable_tf32": False,
        },
        "learning_rate": {
            "type": "wsd",
            "start_lr": 0.0007,
            "stop_lr": 1e-06,
            "warmup_steps": 780,
            "warmup_start_factor": 0.2,
            "decay_phase_ratio": 0.65,
            "decay_type": "cosine",
        },
        "loss": {
            "type": "ener",
            "loss_func": "mae",
            "f_use_norm": True,
            "start_pref_e": 20,
            "limit_pref_e": 20,
            "start_pref_f": 20,
            "limit_pref_f": 20,
            "start_pref_v": 5,
            "limit_pref_v": 5,
        },
        "optimizer": {
            "type": "HybridMuon",
            "muon_mode": "slice",
            "magma_muon": True,
            "lr_adjust": 0.0,
            "weight_decay": 0.001,
        },
        "training": {
            "training_data": {
                "systems": [],
                "batch_size": "auto:128",
            },
            "numb_steps": 10000,
            "gradient_max_norm": 1.0,
            "save_freq": 1000,
            "max_ckpt_keep": 1,
            "enable_ema": False,
            "disp_file": "lcurve.out",
            "disp_freq": 100,
            "disp_avg": True,
            "disp_training": True,
            "time_training": True,
            "seed": 42,
        },
    },
    # ── Future versions ──────────────────────────────────────────────────
    # "air": { ... },   # TODO: add air-specific template when available
    # "plus": { ... },  # TODO: add plus-specific template when available
    # "pro": { ... },   # TODO: add pro-specific template when available
}

AVAILABLE_VERSIONS = list(_DPA4_TEMPLATES.keys())


# ---------------------------------------------------------------------------
# Data utilities
# ---------------------------------------------------------------------------

def _load_atoms(paths: List[Path]) -> List[Atoms]:
    frames: List[Atoms] = []
    for p in paths:
        frames.extend(read(str(p), index=":"))
    logger.info("Loaded %d frames from %d file(s)", len(frames), len(paths))
    return frames


def _split(
    atoms: List[Atoms],
    ratio: float,
    shuffle: bool,
    seed: Optional[int],
) -> Tuple[List[Atoms], Optional[List[Atoms]]]:
    if ratio <= 0 or len(atoms) <= 1:
        return atoms, None
    ratio = max(0.0, min(1.0, ratio))
    n_valid = min(int(round(len(atoms) * ratio)), len(atoms) - 1)
    if n_valid == 0:
        return atoms, None
    indices = list(range(len(atoms)))
    if shuffle:
        random.Random(seed).shuffle(indices)
    valid_idx = set(indices[:n_valid])
    train = [atoms[i] for i in indices if i not in valid_idx]
    valid = [atoms[i] for i in sorted(valid_idx)]
    logger.info("Split → %d train / %d valid", len(train), len(valid))
    return train, valid


def _ase_to_labeled(atoms: Atoms) -> dpdata.LabeledSystem:
    symbols = atoms.get_chemical_symbols()
    names = list(dict.fromkeys(symbols))
    numbs = [symbols.count(n) for n in names]
    types = np.array([names.index(s) for s in symbols], dtype=int)
    if "virial" not in atoms.arrays:
        raise ValueError(
            "Virial data is required but not found in structure. "
            "All structures must have energy, forces, and virial labels from DFT."
        )
    data: Dict[str, Any] = {
        "atom_names": names,
        "atom_numbs": numbs,
        "atom_types": types,
        "cells": np.array([atoms.cell.array]),
        "coords": np.array([atoms.get_positions()]),
        "orig": np.zeros(3),
        "nopbc": not np.any(atoms.get_pbc()),
        "energies": np.array([atoms.get_potential_energy()]),
        "forces": np.array([atoms.get_forces()]),
        "virial": np.array([atoms.arrays["virial"]]),
    }
    return dpdata.LabeledSystem.from_dict({"data": data})


def _export(atoms: List[Atoms], out_dir: Path, mixed_type: bool) -> List[Path]:
    if not atoms:
        raise ValueError("No structures to export to deepmd/npy.")
    out_dir.mkdir(parents=True, exist_ok=True)
    ms = dpdata.MultiSystems()
    for a in atoms:
        ms.append(_ase_to_labeled(a))
    fmt = "deepmd/npy/mixed" if mixed_type else "deepmd/npy"
    ms.to(fmt, str(out_dir))
    paths = [
        Path(p).parent
        for p in glob.glob(str(out_dir) + "/**/type.raw", recursive=True)
    ]
    logger.info("Exported %d system(s) → %s", len(paths), out_dir)
    return paths


def _rand32() -> int:
    return random.randrange(2**32)


def _randomise_seeds(cfg: Dict[str, Any]) -> Dict[str, Any]:
    cfg = json.loads(json.dumps(cfg))
    desc = cfg.get("model", {}).get("descriptor", {})
    desc["seed"] = _rand32()
    fit = cfg.get("model", {}).get("fitting_net", {})
    fit["seed"] = _rand32()
    cfg["training"]["seed"] = _rand32()
    return cfg


def _place_model(base_model: Path, workdir: Path, copy: bool) -> Path:
    """Copy a DPA4 model file into workdir."""
    src = base_model.expanduser().resolve()
    if not src.exists():
        raise ValueError(f"DPA4 base model not found: {src}")
    if src.is_dir():
        raise ValueError(
            f"DPA4 base model must be a file (not a directory), got: {src}"
        )
    dest = workdir / src.name
    if dest.exists() or dest.is_symlink():
        dest.unlink()
    shutil.copy2(src, dest)
    logger.info("Copied base model file → %s", dest)
    return dest


def _apply_lr(cfg: Dict[str, Any], args) -> None:
    lr = cfg["learning_rate"]
    lr["type"] = args.lr_type
    lr["start_lr"] = args.start_lr
    lr["stop_lr"] = args.stop_lr
    if args.lr_type == "wsd":
        lr["warmup_steps"] = args.warmup_steps
        lr["warmup_start_factor"] = args.warmup_start_factor
        lr["decay_phase_ratio"] = args.decay_phase_ratio
        lr["decay_type"] = args.decay_type


def _apply_loss(cfg: Dict[str, Any], args) -> None:
    loss = cfg["loss"]
    loss["type"] = args.loss_type
    loss["loss_func"] = args.loss_func
    loss["start_pref_e"] = args.start_pref_e
    loss["limit_pref_e"] = args.limit_pref_e
    loss["start_pref_f"] = args.start_pref_f
    loss["limit_pref_f"] = args.limit_pref_f
    loss["start_pref_v"] = args.start_pref_v
    loss["limit_pref_v"] = args.limit_pref_v


def _set_data(
    cfg: Dict[str, Any],
    train_paths: List[Path],
    valid_paths: Optional[List[Path]],
    workdir: Path,
) -> None:
    cfg["training"]["training_data"]["systems"] = [
        str(p.relative_to(workdir)) for p in train_paths
    ]
    if valid_paths:
        cfg["training"]["validation_data"] = {
            "systems": [str(p.relative_to(workdir)) for p in valid_paths],
            "batch_size": "auto:128",
        }
    else:
        cfg["training"].pop("validation_data", None)


# ---------------------------------------------------------------------------
# Sub-command implementations
# ---------------------------------------------------------------------------

def cmd_prepare_finetune(args) -> None:
    version = args.version
    if version not in _DPA4_TEMPLATES:
        logger.error(
            "Unknown DPA4 version '%s'. Available: %s",
            version, ", ".join(AVAILABLE_VERSIONS),
        )
        sys.exit(1)

    workdir = Path(args.workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)

    atoms = _load_atoms([Path(p) for p in args.train_data])

    # Split into train / test based on max_train_frames
    max_train = args.max_train_frames
    if max_train > 0 and len(atoms) > max_train:
        indices = list(range(len(atoms)))
        if not args.no_shuffle:
            random.Random(args.seed).shuffle(indices)
        train_idx = sorted(indices[:max_train])
        test_idx = sorted(indices[max_train:])
        train_atoms = [atoms[i] for i in train_idx]
        test_atoms = [atoms[i] for i in test_idx]
        logger.info("Split → %d train / %d test (max_train_frames=%d)",
                     len(train_atoms), len(test_atoms), max_train)
    else:
        train_atoms = atoms
        test_atoms = None
        logger.info("No train/test split (all %d frames for training)", len(atoms))

    train_paths = _export(train_atoms, workdir / "train_data", args.mixed_type)
    test_paths = (
        _export(test_atoms, workdir / "test_data", args.mixed_type)
        if test_atoms else None
    )

    cfg = _randomise_seeds(_DPA4_TEMPLATES[version])
    cfg["training"]["numb_steps"] = args.numb_steps
    _apply_lr(cfg, args)
    _apply_loss(cfg, args)
    cfg["model"]["type_map"] = args.type_map if args.type_map else ALL_TYPES
    # Use train for training, test for validation during training
    _set_data(cfg, train_paths, test_paths, workdir)

    model_dest = _place_model(Path(args.base_model), workdir, args.copy_model)

    _write_input(workdir, cfg)

    exec_cmd = (
        f"dp --pt train input.json --skip-neighbor-stat "
        f"--finetune {model_dest.name} > train_log 2>&1"
    )
    exec_cmd += " && dp --pt freeze -c model.ckpt.pt -o frozen"
    has_test = test_paths is not None
    if has_test:
        exec_cmd += " && dp --pt test -m frozen.pt2 -s test_data -d result-test -l log-test"
    _print_result(
        workdir, "dpa4-finetune", exec_cmd, args.numb_steps,
        model_dest.name, version, has_test,
    )


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _write_input(workdir: Path, cfg: Dict[str, Any]) -> None:
    path = workdir / "input.json"
    with open(path, "w") as f:
        json.dump(cfg, f, indent=4)
    logger.info("Wrote %s", path)


def _print_result(
    workdir: Path, mode: str, exec_cmd: str, numb_steps: int,
    model_dir_name: str = "", version: str = "", has_test: bool = False,
) -> None:
    result = {
        "status": "prepared",
        "workdir": str(workdir),
        "mode": mode,
        "input_json": str(workdir / "input.json"),
        "numb_steps": numb_steps,
        "execution_command": exec_cmd,
        "has_test_data": has_test,
    }
    if version:
        result["version"] = version
    if model_dir_name:
        result["model_name"] = model_dir_name
    print(json.dumps(result, indent=2))


# ---------------------------------------------------------------------------
# convert-data sub-command
# ---------------------------------------------------------------------------

def cmd_convert_data(args) -> None:
    """Convert ASE-readable frames to deepmd/npy for use with ``dp test``."""
    outdir = Path(args.outdir).resolve()
    atoms = _load_atoms([Path(p) for p in args.data])
    if not atoms:
        logger.error("No frames loaded from provided files.")
        sys.exit(1)

    system_paths = _export(atoms, outdir, args.mixed_type)

    test_cmds = []
    for sp in system_paths:
        cmd = f"dp --pt test -m frozen.pt2 -s {sp}"
        if args.nframes:
            cmd += f" -n {args.nframes}"
        test_cmds.append(cmd)

    result = {
        "status": "converted",
        "outdir": str(outdir),
        "num_frames": len(atoms),
        "system_dirs": [str(p) for p in system_paths],
        "dp_test_commands": test_cmds,
    }
    print(json.dumps(result, indent=2))


# ---------------------------------------------------------------------------
# Argument parsers
# ---------------------------------------------------------------------------

def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--workdir", required=True, metavar="DIR",
        help="Output working directory (created if needed)",
    )
    p.add_argument(
        "--numb_steps", type=int, default=10000, metavar="N",
        help="Total training steps (default: 10000)",
    )
    # Learning rate — WSD schedule for DPA4 neo
    p.add_argument("--lr_type", default="wsd",
                   help="Learning rate scheduler type (default: wsd)")
    p.add_argument("--start_lr", type=float, default=0.0007,
                   help="Starting learning rate (default: 0.0007)")
    p.add_argument("--stop_lr", type=float, default=1e-6,
                   help="Stopping learning rate (default: 1e-6)")
    p.add_argument("--warmup_steps", type=int, default=780, metavar="N",
                   help="WSD warmup steps (default: 780)")
    p.add_argument("--warmup_start_factor", type=float, default=0.2,
                   help="WSD warmup start factor (default: 0.2)")
    p.add_argument("--decay_phase_ratio", type=float, default=0.65,
                   help="WSD decay phase ratio (default: 0.65)")
    p.add_argument("--decay_type", default="cosine",
                   help="WSD decay type (default: cosine)")
    # Loss — MAE for DPA4 neo
    p.add_argument("--loss_type", default="ener",
                   help="Loss function type (default: ener)")
    p.add_argument("--loss_func", default="mae",
                   help="Loss function variant (default: mae)")
    p.add_argument("--start_pref_e", type=float, default=20.0)
    p.add_argument("--limit_pref_e", type=float, default=20.0)
    p.add_argument("--start_pref_f", type=float, default=20.0)
    p.add_argument("--limit_pref_f", type=float, default=20.0)
    p.add_argument("--start_pref_v", type=float, default=5.0)
    p.add_argument("--limit_pref_v", type=float, default=5.0)
    # Data split
    p.add_argument(
        "--max_train_frames", type=int, default=0, metavar="N",
        help="Max frames for training (default: 0 = use all). "
             "If set and data has more frames, excess goes to test_data/.",
    )
    p.add_argument(
        "--no_shuffle", action="store_true",
        help="Disable shuffling before train/test split",
    )
    p.add_argument("--seed", type=int, default=None, metavar="N",
                   help="Random seed for reproducible data splitting")
    p.add_argument("--mixed_type", action="store_true",
                   help="Export data in deepmd/npy/mixed format")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dpa4_prepare.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="subcommand", required=True)

    # ── prepare-finetune ────────────────────────────────────────────
    pf = sub.add_parser(
        "prepare-finetune",
        help="Prepare workdir for DPA4 single-task finetuning",
    )
    pf.add_argument(
        "--train_data", nargs="+", required=True, metavar="FILE",
        help="Structure files readable by ASE (xyz, extxyz, POSCAR, …)",
    )
    pf.add_argument(
        "--base_model", required=True, metavar="PATH",
        help="Pretrained DPA4 model file",
    )
    pf.add_argument(
        "--version", default="neo", metavar="VER",
        choices=AVAILABLE_VERSIONS,
        help=(
            f"DPA4 model version — selects the matching input.json template "
            f"(available: {', '.join(AVAILABLE_VERSIONS)}; default: neo). "
            f"Model version and parameters must match exactly."
        ),
    )
    pf.add_argument(
        "--type_map", nargs="+", default=None, metavar="ELEMENT",
        help="Element type map (default: full periodic table)",
    )
    pf.add_argument(
        "--copy_model", action="store_true",
        help="Copy the base model directory into workdir (always done for DPA4)",
    )
    _add_common(pf)
    pf.set_defaults(func=cmd_prepare_finetune)

    # ── convert-data ────────────────────────────────────────────────
    cd = sub.add_parser(
        "convert-data",
        help="Convert ASE-readable structure files to deepmd/npy for dp test",
    )
    cd.add_argument(
        "--data", nargs="+", required=True, metavar="FILE",
        help="Structure files readable by ASE (xyz, extxyz, POSCAR, …)",
    )
    cd.add_argument(
        "--outdir", required=True, metavar="DIR",
        help="Output directory for deepmd/npy system directories",
    )
    cd.add_argument(
        "--mixed_type", action="store_true",
        help="Export in deepmd/npy/mixed format (allows variable composition)",
    )
    cd.add_argument(
        "--nframes", type=int, default=None, metavar="N",
        help="Number of frames to test; embedded in the printed dp test command (optional)",
    )
    cd.set_defaults(func=cmd_convert_data)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
