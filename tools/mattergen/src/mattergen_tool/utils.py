from typing import Dict, Union, Optional, List

from pathlib import Path


def dict_to_fire_args(config: Dict) -> list:
    """Convert a nested dictionary into a list of CLI arguments in FIRE-readable form.

        Format: key=value.
    """
    return [f"--{key}={value}" for key, value in config.items()]


def dict_to_hydra_args(config: Dict) -> list:
    """Convert nested dict to flat hydra-style list.

        Format: key=value.
    """

    def flatten_dict(d, parent_key=""):
        items = []
        for k, v in d.items():
            # Dealing with duplicate names.
            if k.startswith("--"):
                new_key = f"{parent_key}.{k[2:]}" if parent_key else k[2:]
                if not isinstance(v, dict):
                    items.append((new_key, v))
                else:
                    raise ValueError("Keys starting with '--' cannot have nested dictionaries as values.")
            else:
                new_key = f"{parent_key}.{k}" if parent_key else k
                if isinstance(v, dict):
                    items.extend(flatten_dict(v, new_key))
                else:
                    items.append((new_key, v))
        return items

    return [f"{key}={value}" for key, value in flatten_dict(config)]


# Handling commandline arguments.
common_train_args = {
    "adapter":
        {
            # "pretrained_name": None,  # Name of the pretrained model to use, determined by conditioning mode.
            # "model_path": None,  # Path to the pretrained model to use, determined by conditioning mode.
        },
    "--data_module": "alex_mp_20",  # Only alex-mp-20 format supported.
    "data_module":
        {
            # "root_dir": None,  # Path to training dataset, must be set by caller.
            "batch_size": {
                "train": 6,
                "val": 6,
            },
            "num_workers": {
                "train": 6,
                "val": 6,
            },
            "max_epochs": 400,
            # "properties": None,  # List of properties to use for training, determined by conditioning mode.
        },
    "trainer": {
        "max_epochs": 400,
        "accumulate_grad_batches": 4,
    },
    "lightning_module": {
        "scheduler_partials": {
            "0": {
                "scheduler": {
                    "patience": 40,
                    "min_lr": 1e-7,
                },
            },
        },
    },
    "+lighting_module": {
        "scheduler_partials": {
            "0": {
                "scheduler": {
                    "threshold": 0.002,
                    "threshold_mode": "abs",
                },
            },
        },
    },
}


def prepare_train_command(
        model_root: Union[str, Path],
        data_root: Union[str, Path],
        conditioned_properties: List[str],
        additional_args: Optional[Dict] = None,
) -> str:
    """Get command for training a mattergen model with the specified conditioning mode.

    Args:
        model_root (Union[str, Path]): Path to the model root directory.
        data_root (Union[str, Path]): Path to the dataset root directory. Must contain train, may contain val and test.
        conditioned_properties (List[str], optional):
            Must correspond to a subdirectory under model_root containing the base model to finetune from,
            and determines which conditioning properties are used for training.
        additional_args (Optional[Dict]):
            Additional arguments to override the defaults.
            Should be in the same format as `common_train_args`,
            and will be merged with `common_train_args` with `additional_args` taking precedence.
    Returns:
        str: Commandline string to execute for training with the specified conditioning mode.
    """
    args_dict = common_train_args.copy()
    properties = conditioned_properties
    condition_mode = "_".join(properties)
    args_dict["adapter"]["pretrained_name"] = condition_mode
    args_dict["adapter"]["model_path"] = f"{model_root}/{condition_mode}"
    args_dict["data_module"]["root_dir"] = str(data_root)
    args_dict["data_module"]["properties"] = str(properties)  # Only accepts string for commandline.

    args = ["mattergen-finetune"] + dict_to_hydra_args(args_dict) + dict_to_hydra_args(additional_args or {})

    for prop in properties:
        args.append(
            f"+lightning_module/diffusion_module/model/property_embeddings"
            f"@adapter.adapter.property_embeddings_adapt.{prop}={prop}"
        )

    return " \\n    ".join(args)
