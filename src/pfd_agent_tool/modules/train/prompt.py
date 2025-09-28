

TRAINING_STRATEGIES: Dict[str, type[Train]] = {
    "dpa": DPTrain,
}

STRATEGY_META: Dict[str, Dict[str, Any]] = {
    "dpa": {
        "version": "1.0",
        "description": "Training routine for DPA model.",
        # "required_command_keys": ["epochs"],
        "optional_command_keys": ["workdir"],
    }
}