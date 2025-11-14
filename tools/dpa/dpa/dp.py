import random
from typing import Any, List, Dict, Union, Optional, Tuple
from pathlib import Path
import json
import sys
import os
import glob
import logging
import shutil
from ase import Atoms
from dargs import (
    Argument
)
from ase.io import read
import numpy as np
import dpdata  # type: ignore
from matcreator.modules.train import Train
from matcreator.modules.utils import run_command
import logging

DPA2_CONFIG_TEMPLATE = {
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
        "descriptor": {
            "type": "dpa2",
            "repinit": {
                "tebd_dim": 8,
                "rcut": 6.0,
                "rcut_smth": 0.5,
                "nsel": 120,
                "neuron": [
                    25,
                    50,
                    100
                ],
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
            "add_tebd_to_repinit_out": False
        },
        "fitting_net": {
            "neuron": [
                240,
                240,
                240
            ],
            "resnet_dt": True,
            "seed": 2509405570,
            "_comment": " that's all"
        },
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


# -----------------------------------------------------------------------------
# Command documentation & JSON Schema (mirrors command_args())
# -----------------------------------------------------------------------------
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




@Train.register("dpa")
class DPTrain(Train):
    """[Modified from DPGEN2 RunDPTrain]

    Args:
        Train (_type_): _description_
    """
    default_optional_parameter = {}
    train_script_name = "input.json"
    model_file = "model.pb"
    log_file = "train.log"
    lcurve_file = "lcurve.out"
    
    @classmethod
    def training_meta(cls) -> Dict[str, Any]:
        return {
            "version": "v3.0",
            "description": "The default training parameters and command for dpa models",
            "config": {"schema": DPA_CONFIG_JSON_SCHEMA,
                       "doc": DPA_CONFIG_DOC,},
            "command": {"schema": DPA_COMMAND_JSON_SCHEMA,
                        "doc": DPA_COMMAND_DOC,},
        }
    
    def _process_script(self) -> Any:
        input_template = DPA2_CONFIG_TEMPLATE.copy()
        input_template["training"]["numb_steps"] = self.config.get("numb_steps")
        input_template["learning_rate"]["decay_steps"] = self.config.get("decay_steps")
        return self._script_rand_seed(input_template)
    
    def validate(self):
        try:
            command = DPTrain.normalize_command(self.command)
            self.command = command
        except Exception as e:
            raise KeyError(f"Invalid training command: {e}")
        try:
            config = DPTrain.normalize_config(self.config)
            self.config = config
        except Exception as e:
            raise KeyError(f"Invalid training config: {e}")

    def run(self)->Tuple[Path, Path, str]:
        """Run the core training / optimization procedure."""
        
        config = self._process_script()
        ## prepare cli command for dp training
        dp_command = self.command.get("command", "dp").split()
        impl = self.command.get("impl", "tensorflow")
        assert impl in ["tensorflow", "pytorch"]
        if impl == "pytorch":
            dp_command.append("--pt")

        finetune_mode = self.command.pop("finetune_mode",False)
        finetune_args = self.command.get("finetune_args", "")
        train_args = self.command.get("train_args", "")
        mixed_type = self.command.pop("mixed_type",False)

        ## convert extxyz to dpdata systems...
        atoms_ls=[]
        if isinstance(self.train_data, Path):
            self.train_data=[self.train_data]
        for f in self.train_data:
            train_data= read(f,index=":")
            atoms_ls.extend(train_data)
            
            
        train_data=DPTrain.ase2multisys(atoms_ls,labeled=True)
        if mixed_type:
            train_data.to("deepmd/npy/mixed", "./train_data")
        else:
            train_data.to("deepmd/npy", "./train_data")

        
        if self.valid_data:
            valid_data = DPTrain.ase2multisys(read(self.valid_data, index=":"), labeled=True)
            if mixed_type:
                valid_data.to("deepmd/npy/mixed", "./valid_data")
            else:
                valid_data.to("deepmd/npy", "./valid_data")
            valid_data = _get_system_path("./valid_data")
        else:
            valid_data = None
            
        # auto prob style
        auto_prob_str = "prob_sys_size"
        
        config = DPTrain.write_data_to_input_script(
            config,
            _get_system_path("./train_data"),
            auto_prob_str,
            valid_data,
        )

        config["training"]["disp_file"] = "lcurve.out"


        # open log
        fplog = open(self.log_file, "w")
        
        def clean_before_quit():
            fplog.close()
        with open(self.train_script_name, "w") as fp:
            json.dump(config, fp, indent=4)

        if self.optional_files is not None:
            for f in self.optional_files:
                Path(f.name).symlink_to(f)
            
        command = DPTrain._make_train_command(
            dp_command,
            self.train_script_name,
            impl,
            self.model_path,
            finetune_mode,
            finetune_args,
            train_args,
            )

        ret, out, err = run_command(command,raise_error=False, try_bash=False, interactive=False)
        if ret != 0:
            clean_before_quit()
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
            raise RuntimeError("dp train failed")
        fplog.write("#=================== train std out ===================\n")
        fplog.write(out)
        fplog.write("#=================== train std err ===================\n")
        fplog.write(err)

        if finetune_mode == True and os.path.exists("input_v2_compat.json"):
            shutil.copy2("input_v2_compat.json", self.train_script_name)

        # freeze model
        if impl == "pytorch":
            self.model_file = "model.ckpt.pt"
        else:
            ret, out, err = run_command(["dp", "freeze", "-o", "frozen_model.pb"])
            if ret != 0:
                clean_before_quit()
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
            self.model_file = "frozen_model.pb"
        fplog.write("#=================== freeze std out ===================\n")
        fplog.write(out)
        fplog.write("#=================== freeze std err ===================\n")
        fplog.write(err)
        clean_before_quit()
        return Path(self.model_file).resolve(),Path(self.log_file).resolve(),err
        
    def test(
        self
    ) -> Tuple[Path, Dict[str, float]]:
        """Evaluate a trained DeepMD model on a list of labeled datasets using ASE.

        Returns:
            (out_dir, metrics) where metrics contains mae/rmse for energy and forces if labels are present.
        """
        try:
            from deepmd.calculator import DP as DeepmdCalculator  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "Failed to import deepmd ASE calculator (deepmd.calculator.DP). "
                "Install deepmd-kit with ASE support or run evaluation externally."
            ) from e
        
        out_dir = Path("./test_output")
        out_dir.mkdir(parents=True, exist_ok=True)

        model_fp = Path(self.model_file)
        if not Path(model_fp).exists():
            raise FileNotFoundError(f"Model file not found: {model_fp}")
        calc = DeepmdCalculator(model=str(model_fp))

        self.test_results=[]
        for idx, data in enumerate(self.test_data):
            atoms_ls: List[Atoms] = read(str(data), index=":")
            pred_e: List[float] = []
            lab_e: List[float] = []
            pred_f: List[float] = []
            lab_f: List[float] = []
            atom_num: List[int] = []
            for atoms in atoms_ls:
                lab_e.append(atoms.get_potential_energy())
                lab_f.append(atoms.get_forces().flatten())
                atoms.calc=calc
                pred_e.append(atoms.get_potential_energy())
                pred_f.append(atoms.get_forces().flatten())
                atom_num.append(atoms.get_number_of_atoms())

            atom_num = np.array(atom_num)
            # energy prediction
            pred_e = np.array(pred_e)
            lab_e = np.array(lab_e)
            pred_e_atom = pred_e / atom_num
            lab_e_atom = lab_e / atom_num

            # force prediction
            pred_f = np.hstack(pred_f)
            lab_f = np.hstack(lab_f)
        
            np.savetxt(
               str(out_dir / ("test_%02d_.energy.txt"%idx)),
                np.column_stack((lab_e, pred_e)),
                header='',
                comments='#',
                fmt="%.6f",
                )
            np.savetxt(
                str(out_dir / ("test_%02d_.energy_per_atom.txt"%idx)),
                np.column_stack((lab_e_atom, pred_e_atom)),
                header='',
                comments='#',
                fmt="%.6f",
                )

            np.savetxt(
                str(out_dir / ("test_%02d_.force.txt"%idx)),
                np.column_stack((lab_f, pred_f)),
                header='',
                comments='#',
                fmt="%.6f",
                )

            metrics: Dict[str, float] = {
                "system_idx": "%02d"%idx,
                "mae_e": _mae(pred_e, lab_e),
                "rmse_e": _rmse(pred_e, lab_e),
                "mae_e_atom": _mae(pred_e_atom, lab_e_atom),
                "rmse_e_atom": _rmse(pred_e_atom, lab_e_atom),
                "mae_f": _mae(pred_f, lab_f) if lab_f.size else float('nan'),
                "rmse_f": _rmse(pred_f, lab_f) if lab_f.size else float('nan'),
                "n_frames": float(len(atoms_ls)),
                }

            logging.info(f"Test completed on {len(atoms_ls)} frames. Metrics: {metrics}")
            self.test_results.append(metrics)
        return out_dir, self.test_results

    def _set_desc_seed(self, desc):
        """Set descriptor seed.

        Args:
            desc (_type_): _description_
        """
        if desc["type"] == "hybrid":
            for desc in desc["list"]:
                self._set_desc_seed(desc)
        elif desc["type"] not in ["dpa1", "dpa2"]:
            desc["seed"] = random.randrange(sys.maxsize) % (2**32)

    def _script_rand_seed(
            self,
            input_dict,
        ):
        jtmp = input_dict.copy()
        if "model_dict" in jtmp["model"]:
            for d in jtmp["model"]["model_dict"].values():
                if isinstance(d["descriptor"], str):
                    self._set_desc_seed(jtmp["model"]["shared_dict"][d["descriptor"]])
                d["fitting_net"]["seed"] = random.randrange(sys.maxsize) % (2**32)
        else:
            self._set_desc_seed(jtmp["model"]["descriptor"])
            jtmp["model"]["fitting_net"]["seed"] = random.randrange(sys.maxsize) % (
                2**32
            )
        jtmp["training"]["seed"] = random.randrange(sys.maxsize) % (2**32)
        return jtmp
    
    @staticmethod
    def write_data_to_input_script(
        idict: dict,
        train_data: List[Path],
        auto_prob_str: str = "prob_sys_size",
        valid_data: Optional[List[Path]] = None,
    ):
        odict = idict.copy()
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
    
    @staticmethod
    def ase2dpdata(
        atoms: Atoms,
        labeled: bool = False,
        ) -> dpdata.System:
        """[Modified from dpdata.plugins.ase] Convert ase.Atoms to dpdata System.
    
        Parameters
        ----------
        atoms : ase.Atoms
            The ase.Atoms object to convert.
        labeled : bool, optional
            Whether the atoms object has labels (forces, energies, virials), by default False
    
     """
        symbols = atoms.get_chemical_symbols()
        atom_names = list(dict.fromkeys(symbols))
        atom_numbs = [symbols.count(symbol) for symbol in atom_names]
        atom_types = np.array([atom_names.index(symbol) for symbol in symbols]).astype(
            int
            )
        cells = atoms.cell.array
        coords = atoms.get_positions()
        info_dict = {
            "atom_names": atom_names,
            "atom_numbs": atom_numbs,
            "atom_types": atom_types,
            "cells": np.array([cells]),
            "coords": np.array([coords]),
            "orig": np.zeros(3),
            "nopbc": not np.any(atoms.get_pbc()),
        }
        if labeled:
            energy = atoms.get_potential_energy()
            info_dict["energies"] = np.array([energy])
            forces = atoms.get_forces()
            info_dict["forces"] = np.array([forces])
            if "virial" in atoms.arrays:
                virials = atoms.arrays["virial"]
                info_dict["virial"] = np.array([virials])
            return dpdata.LabeledSystem.from_dict({'data':info_dict})
        return dpdata.System.from_dict({'data':info_dict})


    def ase2multisys(
        atoms_list: List[Atoms],
        labeled: bool = False,
        ) -> dpdata.MultiSystems:
        """Convert list of ase.Atoms to dpdata MultiSystem."""
        ms=dpdata.MultiSystems()
        for atoms in atoms_list:
            system = DPTrain.ase2dpdata(atoms, labeled=labeled)
            ms.append(system)
        return ms
    
    @staticmethod
    def _make_train_command(
        dp_command,
        train_script_name,
        impl,
        init_model,
        finetune_mode,
        finetune_args,
        train_args="",
        ):
        # find checkpoint
        if impl == "tensorflow" and os.path.isfile("checkpoint"):
            checkpoint = "model.ckpt"
        elif impl == "pytorch" and len(glob.glob("model.ckpt-[0-9]*.pt")) > 0:
            checkpoint = "model.ckpt-%s.pt" % max(
                [int(f[11:-3]) for f in glob.glob("model.ckpt-[0-9]*.pt")]
            )
        else:
            checkpoint = None
        # case of restart
        if checkpoint is not None:
            command = dp_command + ["train", "--restart", checkpoint, train_script_name]
            return command
        # case of init model and finetune
        assert checkpoint is None
        
        if finetune_mode is True and os.path.isfile(init_model):
            command = (
            dp_command
            + [
                "train",
                train_script_name,
                "--finetune",
                str(init_model),
            ]
            + finetune_args.split()
            )
            logging.info(f"Finetune mode: using init model {init_model}")
        else:
            command = dp_command + ["train", train_script_name]
            logging.info("No available checkpoint found. Training from scratch.")
        command += train_args.split()
        return command
    
    @staticmethod
    def command_args():
        doc_command = "The command for DP, 'dp' for default"
        doc_impl = "The implementation/backend of DP. It can be 'tensorflow' or 'pytorch'. 'tensorflow' for default."
        doc_finetune_args = "Extra arguments for finetuning"
        doc_multitask = "Do multitask training"
        doc_head = "Head to use in the multitask training"
        doc_train_args = "Extra arguments for dp train"
        doc_finetune_mode = "Whether to run in finetune mode"
        doc_mixed_type = "Whether to use mixed type system for training"
        return [
            Argument(
                "command",
                str,
                optional=True,
                default="dp",
                doc=doc_command,
            ),
            Argument(
                "impl",
                str,
                optional=True,
                default="pytorch",
                doc=doc_impl,
                alias=["backend"],
            ),
            Argument(
                "finetune_args",
                str,
                optional=True,
                default="",
                doc=doc_finetune_args,
            ),
            Argument(
                "multitask",
                bool,
                optional=True,
                default=False,
                doc=doc_multitask,
            ),
            Argument(
                "head",
                str,
                optional=True,
                default=None,
                doc=doc_head,
            ),
            Argument(
                "train_args",
                str,
                optional=True,
                default="",
                doc=doc_train_args,
            ),
            Argument(
                "finetune_mode",
                bool,
                optional=True,
                default=False,
                doc=doc_finetune_mode,
            ),
            Argument(
                "mixed_type",
                bool,
                optional=True,
                default=False,
                doc=doc_mixed_type,
            ),
        ]
    @staticmethod
    def normalize_command(data={}):
        ta = DPTrain.command_args()
        base = Argument("base", dict, ta)
        data = base.normalize_value(data, trim_pattern="_*")
        base.check_value(data, strict=True)
        return data
    
    @staticmethod
    def config_args():
        return [
            Argument("numb_steps", int, optional=True, default=100, doc="Number of training steps"),  
            Argument("type_map",List[str],optional=True,default=["Si"],doc="List of atomic types in the training system. Examples: ['H','O']"),          
            Argument("decay_steps", int, optional=True, default=100, doc="Decay steps for learning rate decay"),
        ]
        
    @staticmethod
    def normalize_config(data={}):
        ta = DPTrain.config_args()
        base = Argument("base", dict, ta)
        data = base.normalize_value(data, trim_pattern="_*")
        base.check_value(data, strict=False)
        return data

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