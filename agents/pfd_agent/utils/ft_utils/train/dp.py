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
from ase.io import read, write
from dflow.python import (
    FatalError,
    TransientError
)
import numpy as np
import dpdata  # type: ignore
from agents.pfd_agent.utils.ft_utils.train import Train
from dflow.utils import run_command


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
    
    def _process_script(self, input_dict) -> Any:
        return self._script_rand_seed(input_dict)
    
    def validate(self):
        try:
            command = DPTrain.normalize_config(self.command)
            self.command = command
        except Exception as e:
            raise KeyError(f"Invalid training config: {e}")

    def run(self)->Tuple[Path, Path, str]:
        """Run the core training / optimization procedure."""
        
        config = self._process_script(self.config)
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
        train_data= read(self.train_data,index=":")
        train_data=DPTrain.ase2multisys(train_data,labeled=True)
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
        do_init_model = False
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
            do_init_model,
            self.model_path,
            finetune_mode,
            finetune_args,
            False,# init_model_with_finetune,
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
        return Path(self.model_file),Path(self.log_file),err
        
        

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
        do_init_model,
        init_model,
        finetune_mode,
        finetune_args,
        init_model_with_finetune,
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
        case_init_model = do_init_model and (not init_model_with_finetune)
        case_finetune = finetune_mode == True or (
            do_init_model and init_model_with_finetune
            )
        if case_init_model:
            init_flag = "--init-frz-model" if impl == "tensorflow" else "--init-model"
            command = dp_command + [
                "train",
                init_flag,
                str(init_model),
                train_script_name,
            ]
        elif case_finetune:
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
        else:
            command = dp_command + ["train", train_script_name]
        command += train_args.split()
        return command
    @staticmethod
    def training_args():
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
                default="tensorflow",
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
    def normalize_config(data={}):
        ta = DPTrain.training_args()

        base = Argument("base", dict, ta)
        data = base.normalize_value(data, trim_pattern="_*")
        base.check_value(data, strict=True)

        return data




def _get_system_path(
    data_dir:Union[str,Path]
    ):
    return [Path(ii).parent for ii in glob.glob(str(data_dir) + "/**/type.raw",recursive=True)]


