import argparse
import os
import time
import logging
import random
import traceback
import uuid
from typing import Literal, Optional, Union, Dict, Any, List, Tuple, TypedDict
from pathlib import Path
import numpy as np
import subprocess
import sys
import shutil
import json
import glob
import ast
from pydantic import BaseModel, Field

# ASE / MD imports used by DPA tools
from ase.io import read, write
from ase.atoms import Atoms
import dpdata
from deepmd.calculator import DP

from .utils import run_command, dflow_remote_execution

ALL_TYPES = [
            "H","He","Li","Be","B","C","N","O","F","Ne","Na","Mg","Al","Si","P","S","Cl","Ar","K","Ca","Sc","Ti","V",
            "Cr","Mn","Fe","Co","Ni","Cu","Zn","Ga","Ge","As","Se","Br","Kr","Rb","Sr","Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd","In","Sn","Sb","Te",
            "I","Xe","Cs","Ba","La","Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu","Hf",
            "Ta","W","Re","Os","Ir","Pt","Au","Hg","Tl","Pb","Bi","Po","At","Rn","Fr","Ra","Ac","Th","Pa","U","Np","Pu","Am","Cm",
            "Bk","Cf","Es","Fm","Md","No","Lr","Rf","Db","Sg","Bh","Hs","Mt","Ds","Rg","Cn","Nh","Fl","Mc","Lv","Ts","Og"
        ]

DP_CONFIG_TEMPLATE = {
    "model": {
        "descriptor": {
            "type": "se_atten_v2",
            "sel": "auto",
            "resnet_dt": False,
            "axis_neuron": 12,
            "attn": 128,
            "attn_layer": 0,
            "attn_dotr": True,
            "seed": 1111
        },
        "fitting_net": {
            "seed": 1111
        }
    },
    "learning_rate": {
    },
    "loss": {
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

MULTITASK_CONFIG_TEMPLATE = {
    "_comment": "Multi-task training configuration template",
    "model": {
        "shared_dict": {
            "dpa2_descriptor": {
                "type": "dpa2",
                "repinit": {
                    "tebd_dim": 8,
                    "rcut": 6.0,
                    "rcut_smth": 0.5,
                    "nsel": 120,
                    "neuron": [25, 50, 100],
                    "axis_neuron": 12,
                    "activation_function": "tanh",
                    "three_body_sel": 40,
                    "three_body_rcut": 4.0,
                    "three_body_rcut_smth": 3.5,
                    "use_three_body": True
                },
                "repformer": {
                    "rcut": 4.0,
                    "rcut_smth": 3.5,
                    "nsel": 40,
                    "nlayers": 6,
                    "g1_dim": 128,
                    "g2_dim": 32,
                    "attn2_hidden": 32,
                    "attn2_nhead": 4,
                    "attn1_hidden": 128,
                    "attn1_nhead": 4,
                    "axis_neuron": 4,
                    "update_h2": False,
                    "update_g1_has_conv": True,
                    "update_g1_has_grrg": True,
                    "update_g1_has_drrd": True,
                    "update_g1_has_attn": False,
                    "update_g2_has_g1g1": False,
                    "update_g2_has_attn": True,
                    "update_style": "res_residual",
                    "update_residual": 0.01,
                    "update_residual_init": "norm",
                    "attn2_has_gate": True,
                    "use_sqrt_nnei": True,
                    "g1_out_conv": True,
                    "g1_out_mlp": True
                },
                "add_tebd_to_repinit_out": False,
                "concat_output_tebd": True,
                "precision": "default",
                "smooth": True
            },
            "type_map_all": ALL_TYPES
        },
        "model_dict": {
            
            },
    },
    "learning_rate": {
        "type": "exp",
        "decay_steps": 10,
        "start_lr": 0.001,
        "stop_lr": 3.51e-08
    },
    "loss_dict": {},
    "training": {
        "model_prob": {},
        "data_dict": {},
        "numb_steps": 100,
        "warmup_steps": 0,
        "gradient_max_norm": 5.0,
        "seed": 2912457061,
        "disp_file": "lcurve.out",
        "disp_freq": 100,
        "save_freq": 2000
    }
}


TRAIN_SCRIPT_NAME = "input.json"
TRAIN_LOG_FILE = "train.log"


# ============================================================================
# Pydantic Config Models
# ============================================================================

class LearningRateConfig(BaseModel):
    """Learning rate configuration parameters."""
    lr_type: Literal["exp"] = Field(default="exp", description="Learning rate type")
    decay_steps: int = Field(default=100, ge=0, description="Learning rate decay interval")
    start_lr: float = Field(default=0.001, description="Starting learning rate")
    stop_lr: float = Field(default=3.51e-08, description="Stopping learning rate")


class LossConfig(BaseModel):
    """Loss function configuration parameters."""
    loss_type: Literal["ener"] = Field(default="ener", description="Loss function type")
    start_pref_e: float = Field(default=0.02, description="Starting energy prefactor")
    limit_pref_e: float = Field(default=1.0, description="Limit energy prefactor")
    start_pref_f: float = Field(default=1000.0, description="Starting force prefactor")
    limit_pref_f: float = Field(default=1.0, description="Limit force prefactor")
    start_pref_v: float = Field(default=0.0, description="Starting virial prefactor")
    limit_pref_v: float = Field(default=0.0, description="Limit virial prefactor")

class DescriptorConfig(BaseModel):
    """Descriptor configuration parameters."""
    rcut: float = Field(default=8.0, gt=0, description="Cutoff radius")
    rcut_smth: float = Field(default=0.5, ge=0, description="Smooth cutoff radius")
    descriptor_neuron: List[int] = Field(default_factory=lambda: [25, 50, 100], description="Neurons in descriptor layers")

class FittingNetConfig(BaseModel):
    """Fitting network configuration."""
    neuron: List[int] = Field(default_factory=lambda: [240, 240, 240], description="Neurons in fitting net layers")
    resnet_dt: bool = Field(default=False, description="Use ResNet with dt")

class DPTrainingConfig(LearningRateConfig, LossConfig, DescriptorConfig, FittingNetConfig):
    """Configuration for DPA1 model training from scratch."""
    numb_steps: int = Field(default=100, gt=0, description="Total training steps")
    type_map: Optional[List[str]] = Field(
        default=ALL_TYPES, 
        description="Element type map")
    impl: str = Field(default="pytorch", description="Backend: 'pytorch' or 'tensorflow'")
    #train_args: str = Field(default="", description="Extra CLI arguments for dp train")
    mixed_type: bool = Field(default=False, description="Use mixed-type dataset format")
    split_ratio: Optional[float] = Field(default=0.1, description="Split ratio for valid data. Default to 0.1 (no split)")
    shuffle: Optional[bool] = Field(default=True, description="Whether to shuffle data before splitting. Default: True")
    seed: Optional[int] = Field(default=None, description="seed: random.Random seed for reproducible shuffling. Default: None")


class DPAFinetuneConfig(LearningRateConfig, LossConfig):
    """Configuration for single-task DPA model finetuning."""
    numb_steps: int = Field(default=100, gt=0, description="Total training steps")
    type_map: Optional[List[str]] = Field(default=ALL_TYPES, description="Element type map")
    head: str = Field(default=None, description="Model head to be fine-tuned. Default to None (re-init)")
    mixed_type: bool = Field(default=False, description="Use mixed-type dataset format")
    split_ratio: Optional[float] = Field(default=0.1, description="Split ratio for valid data. Default to 0.1 (no split)")
    shuffle: Optional[bool] = Field(default=True, description="Whether to shuffle data before splitting. Default: True")
    seed: Optional[int] = Field(default=None, description="seed: random.Random seed for reproducible shuffling. Default: None")


class TaskConfig(LossConfig, FittingNetConfig):
    """Configuration for a single task in multi-task training."""
    model_prob: Optional[float] = Field(default=1, description="Task sampling probability. Default to 1")
    split_ratio: Optional[float] = Field(default=0.1, description="Split ratio for valid data. Default to 0.1 (no split)")
    shuffle: Optional[bool] = Field(default=True, description="Whether to shuffle data before splitting. Default: True")
    seed: Optional[int] = Field(default=None, description="seed: random.Random seed for reproducible shuffling. Default: None")
        

class DPAMultitaskFinetuneConfig(LearningRateConfig):
    """Configuration for multi-task DPA model finetuning."""
    numb_steps: int = Field(default=100, gt=0, description="Total training steps")
    decay_steps: int = Field(default=100, ge=0, description="Learning rate decay interval")
    type_map: Optional[List[str]] = Field(default=ALL_TYPES, description="Element type map")
    task_configs: Dict[str, TaskConfig] = Field(description="Per-task configurations")
    mixed_type: bool = Field(default=False, description="Use mixed-type dataset format")


# ============================================================================
# Main Training Functions
# ============================================================================

@dflow_remote_execution(
    artifact_inputs={
        "train_data": List[Path],
    },
    artifact_outputs={
        "model": Path,
        "log": Path,
    },
    parameter_inputs={
        "workdir": Path,
        "config": DPTrainingConfig
    },
    parameter_outputs={
        "message": str,
        "status": str,
    },
    op_name="DPTrainingOP"
)
def dp_training(
    workdir: Path,
    train_data: List[Path],
    config: DPTrainingConfig
) -> Dict[str, Any]:
    """Train DP model from scratch.
    
    Parameters:
        workdir: Working directory for training
        train_data: List of training data files (xyz/extxyz/vasp/...)
        config: DP training configuration
        
    Returns:
        Dict with model path, log path, status, and message
    """
    try:
        workdir.mkdir(parents=True, exist_ok=True)
        
        # Build command
        dp_command = ["dp"]
        if config.impl == "pytorch":
            dp_command.append("--pt")
        
        # Load training data
        train_atoms = _load_atoms_from_paths(train_data)
        
        # Handle validation split
        train_atoms, valid_atoms = _split_train_valid(train_atoms, config.split_ratio, config.shuffle, config.seed)
        
        # Export training and validation data
        train_paths = _export_dpdata(train_atoms, workdir / "train_data", config.mixed_type)
        valid_paths = None
        if valid_atoms:
            valid_paths = _export_dpdata(valid_atoms, workdir / "valid_data", config.mixed_type)
        
        # Build training configuration from template
        training_config = _script_rand_seed(DP_CONFIG_TEMPLATE)
        training_config = json.loads(json.dumps(training_config))
        
        # Update configuration parameters
        training_config["training"]["numb_steps"] = config.numb_steps
        training_config["learning_rate"]["type"] = config.lr_type
        training_config["learning_rate"]["decay_steps"] = config.decay_steps
        training_config["learning_rate"]["start_lr"] = config.start_lr
        training_config["learning_rate"]["stop_lr"] = config.stop_lr
        training_config["loss"]["type"] = config.loss_type
        training_config["loss"]["start_pref_e"] = config.start_pref_e
        training_config["loss"]["limit_pref_e"] = config.limit_pref_e
        training_config["loss"]["start_pref_f"] = config.start_pref_f
        training_config["loss"]["limit_pref_f"] = config.limit_pref_f
        training_config["loss"]["start_pref_v"] = config.start_pref_v
        training_config["loss"]["limit_pref_v"] = config.limit_pref_v
        training_config["model"]["descriptor"]["rcut"] = config.rcut
        training_config["model"]["descriptor"]["rcut_smth"] = config.rcut_smth
        training_config["model"]["descriptor"]["neuron"] = config.descriptor_neuron
        training_config["model"]["fitting_net"]["neuron"] = config.neuron
        training_config["model"]["fitting_net"]["resnet_dt"] = config.resnet_dt
        training_config["training"]["disp_file"] = "lcurve.out"
        
        if config.type_map:
            training_config["model"]["type_map"] = config.type_map
        
        # Populate training data paths
        training_config["training"]["training_data"]["systems"] = [str(p.relative_to(workdir)) for p in train_paths]
        training_config["training"]["training_data"]["batch_size"] = "auto"
        training_config["training"]["training_data"]["auto_prob"] = "prob_sys_size"
        
        # Populate validation data paths if provided
        if valid_paths:
            training_config["training"]["validation_data"] = {
                "systems": [str(p.relative_to(workdir)) for p in valid_paths],
                "batch_size": 1,
            }
        else:
            training_config["training"].pop("validation_data", None)
        
        # Write training script
        train_script_path = workdir / TRAIN_SCRIPT_NAME
        with open(train_script_path, "w") as fp:
            json.dump(training_config, fp, indent=4)
        
        # Run training
        command_list = dp_command + ["train", TRAIN_SCRIPT_NAME]
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
                raise RuntimeError(f"dp train failed\nstdout:\n{out}\nstderr:\n{err}")
            fplog.write("#=================== train std out ===================\n")
            fplog.write(out)
            fplog.write("#=================== train std err ===================\n")
            fplog.write(err)
        
        # Determine model file
        model_file = workdir / ("model.ckpt.pt" if config.impl == "pytorch" else "frozen_model.pb")
        if config.impl != "pytorch":
            ret, out, err = run_command(["dp", "freeze", "-o", "frozen_model.pb"], raise_error=False, cwd=str(workdir))
            if ret != 0:
                raise RuntimeError(f"dp freeze failed: {err}")
        
        return {
            "status": "success",
            "model": model_file.resolve(),
            "log": log_file_path.resolve(),
            "message": "DP training completed successfully"
        }
    except Exception as e:
        logging.error(f"Error in dpa_training: {str(e)}")
        return {
            "status": "error",
            "model": workdir / "model.ckpt.pt",
            "log": workdir / TRAIN_LOG_FILE,
            "message": f"DP training failed: {e}"
        }


@dflow_remote_execution(
    artifact_inputs={
        "train_data": List[Path],
        "base_model": Path,
    },
    artifact_outputs={
        "model": Path,
        "log": Path,
    },
    parameter_inputs={
        "workdir": Path,
        "config": DPAFinetuneConfig,
    },
    parameter_outputs={
        "message": str,
        "status": str,
    },
    op_name="DPAFinetuneOP"
)
def dpa_finetuning(
    workdir: Path,
    train_data: List[Path],
    base_model: Path,
    config: DPAFinetuneConfig,
) -> Dict[str, Any]:
    """Finetune a DPA model (single-task).
    
    Parameters:
        workdir: Working directory for finetuning
        train_data: List of training data files
        base_model: Path to pretrained model (.pt or .pb)
        config: Finetuning configuration
        
    Returns:
        Dict with model path, log path, status, and message
    """
    try:
        workdir.mkdir(parents=True, exist_ok=True)
        # Build command
        dp_command = ["dp","--pt"]
        
        # Load training data
        train_atoms = _load_atoms_from_paths(train_data)
        

        train_atoms, valid_atoms = _split_train_valid(train_atoms, config.split_ratio, config.shuffle, config.seed)
        
        # Export training and validation data
        train_paths = _export_dpdata(train_atoms, workdir / "train_data", config.mixed_type)
        valid_paths = None
        if valid_atoms:
            valid_paths = _export_dpdata(valid_atoms, workdir / "valid_data", config.mixed_type)
        
        # Build training configuration from finetune template
        training_config = json.loads(json.dumps(DPA_CONFIG_TEMPLATE))
        
        # Update configuration parameters
        training_config["training"]["numb_steps"] = config.numb_steps
        training_config["learning_rate"]["type"] = config.lr_type
        training_config["learning_rate"]["decay_steps"] = config.decay_steps
        training_config["learning_rate"]["start_lr"] = config.start_lr
        training_config["learning_rate"]["stop_lr"] = config.stop_lr
        training_config["loss"]["type"] = config.loss_type
        training_config["loss"]["start_pref_e"] = config.start_pref_e
        training_config["loss"]["limit_pref_e"] = config.limit_pref_e
        training_config["loss"]["start_pref_f"] = config.start_pref_f
        training_config["loss"]["limit_pref_f"] = config.limit_pref_f
        training_config["loss"]["start_pref_v"] = config.start_pref_v
        training_config["loss"]["limit_pref_v"] = config.limit_pref_v
        training_config["training"]["disp_file"] = "lcurve.out"
        
        #if config.type_map:
        training_config["model"]["type_map"] = config.type_map
        
        # Populate training data paths
        training_config["training"]["training_data"]["systems"] = [str(p.relative_to(workdir)) for p in train_paths]
        training_config["training"]["training_data"]["batch_size"] = "auto"
        training_config["training"]["training_data"]["auto_prob"] = "prob_sys_size"
        
        # Populate validation data paths if provided
        if valid_paths:
            training_config["training"]["validation_data"] = {
                "systems": [str(p.relative_to(workdir)) for p in valid_paths],
                "batch_size": "auto",
            }
        else:
            training_config["training"].pop("validation_data", None)
        
        # Write training script
        train_script_path = workdir / TRAIN_SCRIPT_NAME
        with open(train_script_path, "w") as fp:
            json.dump(training_config, fp, indent=4)
        
        # Setup init model
        resolved_model = base_model.expanduser().resolve()
        init_model_path = workdir / resolved_model.name
        if init_model_path.exists():
            if init_model_path.is_symlink() or init_model_path.is_file():
                init_model_path.unlink()
        init_model_path.symlink_to(resolved_model)
        
        # Run finetuning
        command_list = (
            dp_command
            + ["train", TRAIN_SCRIPT_NAME, "--finetune", str(init_model_path.relative_to(workdir)), "--use-pretrain-script"]
        )
        if config.head:
            command_list.append(config.head)
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
                raise RuntimeError(f"dp finetune failed\nstdout:\n{out}\nstderr:\n{err}")
            fplog.write("#=================== train std out ===================\n")
            fplog.write(out)
            fplog.write("#=================== train std err ===================\n")
            fplog.write(err)
        
        # Handle compat file
        compat_file = workdir / "input_v2_compat.json"
        if compat_file.exists():
            shutil.copy2(compat_file, train_script_path)
        
        # Determine model file
        model_file = workdir / "model.ckpt.pt"
        
        return {
            "status": "success",
            "model": model_file.resolve(),
            "log": log_file_path.resolve(),
            "message": "Finetuning completed successfully"
        }
    except Exception as e:
        logging.error(f"Error in dpa_finetuning: {str(e)}")
        return {
            "status": "error",
            "model": workdir / "model.ckpt.pt",
            "log": workdir / TRAIN_LOG_FILE,
            "message": f"Finetuning failed: {e}"
        }


@dflow_remote_execution(
    artifact_inputs={
        "train_data": Dict[str, List[Path]],
        "base_model": Path,
    },
    artifact_outputs={
        "model": Path,
        "log": Path,
    },
    parameter_inputs={
        "workdir": Path,
        "config": DPAMultitaskFinetuneConfig,
    },
    parameter_outputs={
        "message": str,
        "status": str,
    },
    op_name="DPAMultitaskFinetuneOP"
)
def dpa_finetuning_multitask(
    workdir: Path,
    train_data: Dict[str, List[Path]],
    base_model: Path,
    config: DPAMultitaskFinetuneConfig
) -> Dict[str, Any]:
    """Finetune a DPA model with multi-task learning.
    
    Parameters:
        workdir: Working directory for finetuning
        train_data: Dict mapping task names to training data files
        base_model: Path to pretrained model (.pt)
        config: Multi-task finetuning configuration
        
    Returns:
        Dict with model path, log path, status, and message
    """
    try:
        workdir.mkdir(parents=True, exist_ok=True)
        
        # Build command
        dp_command = ["dp"]
        dp_command.append("--pt")
    
        training_config = json.loads(json.dumps(MULTITASK_CONFIG_TEMPLATE))
        
        # Update shared learning rate
        training_config["training"]["numb_steps"] = config.numb_steps
        training_config["learning_rate"]["type"] = config.lr_type
        training_config["learning_rate"]["decay_steps"] = config.decay_steps
        training_config["learning_rate"]["start_lr"] = config.start_lr
        training_config["learning_rate"]["stop_lr"] = config.stop_lr
        
        for task_name, task_train_files in train_data.items():
            task_config = config.task_configs[task_name]
            training_config["model"]["model_dict"][task_name] = {
                "type_map":"type_map_all",
                "descriptor":"dpa2_descriptor",
                "fitting_net": {
                    "neuron": task_config.neuron,
                    "resnet_dt": task_config.resnet_dt,
                    "seed": 1
                }
            }
            training_config["loss_dict"][task_name] = {
                "type": task_config.loss_type,
                "start_pref_e": task_config.start_pref_e,
                "limit_pref_e": task_config.limit_pref_e,
                "start_pref_f": task_config.start_pref_f,
                "limit_pref_f": task_config.limit_pref_f,
                "start_pref_v": task_config.start_pref_v,
                "limit_pref_v": task_config.limit_pref_v,
            }
            training_config["training"]["model_prob"][task_name] = task_config.model_prob
            
            training_config["training"]["data_dict"][task_name] = {
                "training_data": {
                    "systems": [],
                    "batch_size": "auto",
                    "auto_prob": "prob_sys_size"
                }
            }
            
            # Load training data
            train_atoms = _load_atoms_from_paths(task_train_files)
            
            # Handle validation split
            train_atoms, task_valid_atoms = _split_train_valid(
                train_atoms, 
                task_config.split_ratio, 
                task_config.shuffle, 
                task_config.seed
                )
            if task_valid_atoms:
                logging.info(f"Task '{task_name}': Split complete")
            
            # Export training data
            train_paths = _export_dpdata(train_atoms, workdir / f"train_data_{task_name}", config.mixed_type)
            training_config["training"]["data_dict"][task_name]["training_data"]["systems"] = [str(p.relative_to(workdir)) for p in train_paths]
            
            # Export validation data if available
            if task_valid_atoms:
                valid_paths = _export_dpdata(task_valid_atoms, workdir / f"valid_data_{task_name}", config.mixed_type)
                training_config["training"]["data_dict"][task_name]["validation_data"] = {
                    "systems": [str(p.relative_to(workdir)) for p in valid_paths],
                    "batch_size": 1
                }
        
        # Write training script
        train_script_path = workdir / TRAIN_SCRIPT_NAME
        with open(train_script_path, "w") as fp:
            json.dump(training_config, fp, indent=4)
        
        # Setup init model
        resolved_model = base_model.expanduser().resolve()
        init_model_path = workdir / resolved_model.name
        if init_model_path.exists():
            if init_model_path.is_symlink() or init_model_path.is_file():
                init_model_path.unlink()
        init_model_path.symlink_to(resolved_model)
        
        # Run finetuning
        command_list = (
            dp_command
            + ["train", TRAIN_SCRIPT_NAME, "--finetune", str(init_model_path.relative_to(workdir)), "--use-pretrain-script"]
        )
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
                raise RuntimeError(f"dp multitask finetune failed\nstdout:\n{out}\nstderr:\n{err}")
            fplog.write("#=================== train std out ===================\n")
            fplog.write(out)
            fplog.write("#=================== train std err ===================\n")
            fplog.write(err)
        
        # Handle compat file
        compat_file = workdir / "input_v2_compat.json"
        if compat_file.exists():
            shutil.copy2(compat_file, train_script_path)
        
        # Multi-task always uses PyTorch
        model_file = workdir / "model.ckpt.pt"
        
        return {
            "status": "success",
            "model": model_file.resolve(),
            "log": log_file_path.resolve(),
            "message": f"Multi-task finetuning completed for {len(train_data)} tasks"
        }
    except Exception as e:
        logging.error(f"Error in dpa_finetuning_multitask: {str(e)}")
        return {
            "status": "error",
            "model": workdir / "model.ckpt.pt",
            "log": workdir / TRAIN_LOG_FILE,
            "message": f"Multi-task finetuning failed: {e}"
        }


# ============================================================================
# Utility Functions
# ============================================================================

def get_model_descriptor(model_path: Path) -> Dict[str, Any]:
    """Extract descriptor configuration from a DPA model.
    
    Parameters:
        model_path: Path to the DPA model file (.pt or .pb)
        
    Returns:
        Dict containing descriptor configuration. For multi-task models,
        returns the first branch's descriptor (they share the same descriptor).
        
    Raises:
        RuntimeError: If dp show command fails
        ValueError: If descriptor info cannot be parsed
    """
    ret, out, err = run_command(
        ["dp", "show", str(model_path), "descriptor"],
        raise_error=False,
        try_bash=False,
    )
    
    if ret != 0:
        raise RuntimeError(f"Failed to get descriptor info: {err}")
    
    # Parse the output to extract descriptor dict
    # Look for lines like: "The descriptor parameter of branch <name> is {...}"
    output = out + err  # Descriptor info may be in stderr
    
    for line in output.split('\n'):
        if 'The descriptor parameter of branch' in line:
            # Extract the dict part after " is "
            dict_str = line.split(' is ', 1)[1]
            try:
                descriptor_dict = ast.literal_eval(dict_str)
                return descriptor_dict
            except (ValueError, SyntaxError) as e:
                raise ValueError(f"Failed to parse descriptor dict: {e}")
    
    # If no multi-task format found, try to parse the whole output as JSON/dict
    # (for single-task models, output might be different)
    raise ValueError("Could not find descriptor parameters in model output")


def list_model_heads(model_path: Path) -> List[str]:
    """List available heads (branches) in a DPA model.
    
    Parameters:
        model_path: Path to the DPA model file (.pt or .pb)
        
    Returns:
        List of head names. Returns empty list for single-task models,
        or list of branch names for multi-task models.
        
    Raises:
        RuntimeError: If dp show command fails
    """
    ret, out, err = run_command(
        ["dp", "show", str(model_path), "model-branch"],
        raise_error=False,
        try_bash=False,
    )
    
    if ret != 0:
        # Try alternative: parse descriptor output for branch names
        ret2, out2, err2 = run_command(
            ["dp", "show", str(model_path), "descriptor"],
            raise_error=False,
            try_bash=False,
        )
        
        if ret2 != 0:
            raise RuntimeError(f"Failed to get model info: {err}")
        
        # Parse descriptor output for branch names
        output = out2 + err2
        heads = []
        
        # Check if it's a multitask model
        if "This is a multitask model" not in output:
            return []  # Single-task model
        
        # Extract branch names from lines like:
        # "The descriptor parameter of branch <name> is {...}"
        for line in output.split('\n'):
            if 'The descriptor parameter of branch' in line:
                # Extract branch name between "branch " and " is"
                parts = line.split('branch ', 1)
                if len(parts) > 1:
                    branch_name = parts[1].split(' is', 1)[0].strip()
                    heads.append(branch_name)
        
        return heads
    
    # Parse model-branch output if successful
    output = out + err
    heads = []
    
    for line in output.split('\n'):
        line = line.strip()
        if line and not line.startswith('[') and not line.startswith('DEEPMD'):
            heads.append(line)
    
    return heads


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

def _split_train_valid(
    train_atoms: List[Atoms],
    valid_ratio: float = 0.0,
    shuffle: bool = True,
    seed: Optional[int] = None,
) -> Tuple[List[Atoms], Optional[List[Atoms]]]:
    """Split training data into train and validation sets.
    
    Parameters:
        train_atoms: List of training structures
        valid_ratio: Fraction of data to use for validation (0.0-1.0)
        shuffle: Whether to shuffle before splitting
        seed: Random seed for reproducible shuffling
        
    Returns:
        Tuple of (train_atoms, valid_atoms), where valid_atoms is None if no split
    """
    if valid_ratio <= 0 or len(train_atoms) <= 1:
        return train_atoms, None
    
    rv = max(0.0, min(1.0, float(valid_ratio)))
    n_valid = int(round(len(train_atoms) * rv))
    
    # Ensure at least one training sample
    if n_valid >= len(train_atoms):
        n_valid = max(0, len(train_atoms) - 1)
    
    if n_valid == 0:
        return train_atoms, None
    
    # Shuffle and split
    indices = list(range(len(train_atoms)))
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(indices)
    
    valid_indices = set(indices[:n_valid])
    train_indices = [i for i in indices if i not in valid_indices]
    
    valid_atoms = [train_atoms[i] for i in sorted(valid_indices)]
    train_atoms_split = [train_atoms[i] for i in train_indices]
    
    logging.info(f"Split {len(valid_atoms)} validation samples from {len(train_atoms)} total structures")
    
    return train_atoms_split, valid_atoms


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