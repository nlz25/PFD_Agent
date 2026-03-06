from typing import Any, Callable, Dict, Optional
from functools import wraps
from pathlib import Path
import os
import logging
import importlib
import time

import dflow
from dflow.config import (
    config,
    s3_config,
)
from dflow.plugins import (
    bohrium,
)
from dflow.plugins.dispatcher import (
    DispatcherExecutor,
)
from dflow import (
    Step,
    Workflow,
    upload_artifact,
    download_artifact,
)
from dflow.python import (
    OP,
    OPIO,
    OPIOSign,
    PythonOPTemplate,
    Artifact,
    Parameter,
    BigParameter,
)

logger = logging.getLogger(__name__)

def _is_artifact_type(typ) -> bool:
    """Determine if a type should be treated as Artifact (True) or BigParameter (False).

    Args:
        typ: The type to check

    Returns:
        True if the type should be an Artifact (Path or List[Path]), False otherwise
    """
    # Handle typing.List, typing.Dict, etc.
    import typing

    # Direct Path type
    if typ is Path:
        return True

    # Check for List[Path]
    if hasattr(typ, '__origin__'):
        origin = typ.__origin__
        # For List[Path]
        if origin is list:
            args = getattr(typ, '__args__', ())
            if args and args[0] is Path:
                return True

    # Everything else (Dict, complex types, etc.) should be BigParameter
    return False


BOHRIUM_CONFIG={
      "host":"https://workflows.deepmodeling.com",
      "k8s_api_server":"https://workflows.deepmodeling.com",
      "repo_key": "oss-bohrium",
      "storage_client": "dflow.plugins.bohrium.TiefblueClient",
  }

def bohrium_config_from_dict(
    bohrium_config,
):

    config["host"] = bohrium_config.get("host",BOHRIUM_CONFIG["host"])
    config["k8s_api_server"] = bohrium_config.get("k8s_api_server",BOHRIUM_CONFIG["k8s_api_server"])
    bohrium.config["username"] = bohrium_config["username"]
    if bohrium_config.get("password"):
        bohrium.config["password"] = bohrium_config["password"]
    elif bohrium_config.get("ticket"):
        bohrium.config["ticket"] = bohrium_config["ticket"]
    bohrium.config["project_id"] = str(bohrium_config["project_id"])
    s3_config["repo_key"] = bohrium_config.get("repo_key", BOHRIUM_CONFIG["repo_key"])
    storage_client = bohrium_config.get("storage_client", BOHRIUM_CONFIG["storage_client"])
    module, cls = storage_client.rsplit(".", maxsplit=1)
    module = importlib.import_module(module)
    client = getattr(module, cls)
    s3_config["storage_client"] = client()


def dflow_remote_execution(
        artifact_inputs: Optional[Dict[str, type]] = None,
        artifact_outputs: Optional[Dict[str, type]] = None,
        parameter_inputs: Optional[Dict[str, type]] = None,
        parameter_outputs: Optional[Dict[str, type]] = None,
        op_name: Optional[str] = None,
):
    """
    Decorator to enable dflow remote execution for a function.

    This decorator wraps a function to support both local and remote execution via dflow.
    When mode="debug", the function runs locally. When mode="bohrium" or other remote modes,
    it submits the function as a dflow workflow.

    Args:
        artifact_inputs: Dict mapping parameter names to Path type (for file/directory inputs)
        artifact_outputs: Dict mapping return dict keys to Path type (for file/directory outputs)
        parameter_inputs: Dict mapping parameter names to their types (for simple inputs)
        parameter_outputs: Dict mapping return dict keys to their types (for simple outputs)
        op_name: Optional name for the OP class (defaults to function name)

    Example:
        @dflow_remote_execution(
            artifact_inputs={"train_data": List[Path], "model_path": Path},
            artifact_outputs={"model": Path, "log": Path},
            parameter_outputs={"message": str}
        )
        def my_training_function(workdir, config, train_data, model_path=None, ...):
            # Function implementation
            return {"model": model_path, "log": log_path, "message": "success"}

    Usage:
        # Local execution
        result = my_training_function(workdir, config, train_data, mode="debug")

        # Remote execution
        result = my_training_function(workdir, config, train_data, mode="bohrium",
                                     executor=my_executor, template_config={...})
    """
    artifact_inputs = artifact_inputs or {}
    artifact_outputs = artifact_outputs or {}
    parameter_inputs = parameter_inputs or {}
    parameter_outputs = parameter_outputs or {}

    def decorator(func: Callable) -> Callable:
        # Create the OP class dynamically
        class DynamicOP(OP):
            @classmethod
            def get_input_sign(cls):
                inputs = {}
                for key, typ in artifact_inputs.items():
                    if _is_artifact_type(typ):
                        inputs[key] = Artifact(typ)
                    else:
                        inputs[key] = BigParameter(typ)
                for key, typ in parameter_inputs.items():
                    inputs[key] = Parameter(typ)
                return OPIOSign(inputs)

            @classmethod
            def get_output_sign(cls):
                outputs = {}
                for key, typ in artifact_outputs.items():
                    if _is_artifact_type(typ):
                        outputs[key] = Artifact(typ)
                    else:
                        outputs[key] = BigParameter(typ)
                for key, typ in parameter_outputs.items():
                    outputs[key] = Parameter(typ)
                return OPIOSign(outputs)

            @OP.exec_sign_check
            def execute(self, ip: OPIO) -> OPIO:
                # Extract all inputs
                kwargs = {}
                for key in artifact_inputs.keys():
                    kwargs[key] = ip[key]
                for key in parameter_inputs.keys():
                    kwargs[key] = ip[key]

                # Call the original function
                result = func(**kwargs)

                # Prepare outputs
                op_output = {}
                for key in artifact_outputs.keys():
                    op_output[key] = result[key]
                for key in parameter_outputs.keys():
                    op_output[key] = result[key]

                return OPIO(op_output)

        # Set a better name for the OP class
        op_class_name = op_name or f"{func.__name__.title().replace('_', '')}OP"
        DynamicOP.__name__ = op_class_name
        DynamicOP.__module__ = __name__
        DynamicOP.__qualname__ = op_class_name

        # Register the OP class in the utils module namespace
        # This makes it available for DFlow's import: from dpa_tool.utils import DPTrainingOP
        globals()[op_class_name] = DynamicOP

        @wraps(func)
        def wrapper(*args, executor=None, template_config: Optional[Dict[str, Any]] = None,
                    workflow_name: Optional[str] = None, **kwargs):
            """
            Wrapped function that supports both local and remote execution.

            Args:
                mode: Execution mode - "debug" for local, "bohrium" or others for remote
                executor: DFlow executor for remote execution (e.g., DispatcherExecutor)
                template_config: Configuration dict for PythonOPTemplate
                workflow_name: Name for the workflow (auto-generated if not provided)
                **kwargs: All original function parameters
            """
            mode = os.environ.get("DPA_SUBMIT_TYPE", "local")
            # Local execution
            if mode == "local":
                return func(*args, **kwargs)

            # Remote execution via dflow
            logger.info(f"Submitting {func.__name__} to dflow in {mode} mode")

            # Set dflow mode
            if mode == "bohrium":
                dflow.config["mode"] = "default"
                bohrium_config = {
                    "username": os.environ.get("BOHRIUM_USERNAME"),
                    "password": os.environ.get("BOHRIUM_PASSWORD"),
                    "project_id": int(os.environ.get("BOHRIUM_PROJECT_ID", 0)),
                }
                bohrium_config_from_dict(bohrium_config)

                executor = DispatcherExecutor(
                    image_pull_policy="IfNotPresent",
                    machine_dict={
                        "batch_type": "Bohrium",
                        "context_type": "Bohrium",
                        "remote_profile": {
                            "input_data": {
                                "job_type": "container",
                                "platform": "ali",
                                "scass_type":
                                    os.environ["BOHRIUM_DPA_MACHINE"]}}})
                # python_packages =[]
                env_packages = os.environ.get("BOHRIUM_PYTHON_PACKAGES", "")
                python_packages = []

                for pkg_name in env_packages.split(","):
                    pkg_name = pkg_name.strip()
                    if pkg_name:
                        try:
                            mod = importlib.import_module(pkg_name)
                            python_packages.append(list(mod.__path__)[0])
                        except ImportError:
                            logging.warning(f"Failed to import package: {pkg_name}")

                template_config = {
                    "image": os.environ.get("BOHRIUM_DPA_IMAGE"),
                    "python_packages": python_packages,
                }

            elif mode == "debug":
                dflow.config["mode"] = mode
                template_config = {}
                executor = None

            else:
                raise ValueError(f"Unsupported mode: {mode}")
            # Prepare artifacts and parameters for upload
            artifacts = {}
            parameters = {}

            # Match positional args with function signature
            import inspect
            sig = inspect.signature(func)
            param_names = list(sig.parameters.keys())

            # Build kwargs from args and kwargs
            all_kwargs = dict(zip(param_names, args))
            all_kwargs.update(kwargs)

            # Separate artifacts and parameters
            for key, value in all_kwargs.items():
                if key in artifact_inputs:
                    typ = artifact_inputs[key]
                    if _is_artifact_type(typ):
                        # Upload as artifact
                        if isinstance(value, (list, tuple)):
                            artifacts[key] = upload_artifact(value)
                        else:
                            artifacts[key] = upload_artifact(value)
                    else:
                        # Use as big parameter
                        parameters[key] = value
                elif key in parameter_inputs:
                    parameters[key] = value

            # Create workflow
            wf_name = workflow_name or f"workflow"
            wf = Workflow(name=wf_name)

            # Create template config
            tmpl_config = template_config or {}

            step_name = f"{func.__name__.lstrip('_-').replace('_', '-')}-step"
            # Create step
            step = Step(
                name=step_name,
                template=PythonOPTemplate(
                    DynamicOP,
                    **tmpl_config
                ),
                parameters=parameters,
                artifacts=artifacts,
                executor=executor,
            )

            wf.add(step)

            # Submit workflow
            logger.info(f"Submitting workflow '{wf_name}'")
            wf.submit()

            # Wait for completion
            while wf.query_status() in ["Pending", "Running"]:
                time.sleep(5)

            status = wf.query_status()
            if status != "Succeeded":
                logger.error(f"Workflow '{wf_name}' failed with status: {status}")
                return {
                    "status": "error",
                    "workflow_id": wf.id,
                    "message": f"Workflow failed with status: {status}",
                }

            logger.info(f"Workflow '{wf_name}' completed successfully")

            # Download results
            step_info = wf.query()
            try:
                completed_step = step_info.get_step(name=step_name)[0]
            except (IndexError, KeyError):
                logger.warning(f"Could not find '{step_name}' for artifact download")
                return {
                    "status": "error",
                    "workflow_id": wf.id,
                    "message": f"Workflow completed but artifacts not found",
                }

            # Create download directory
            download_path = Path(f"./{step_name}_results")
            download_path.mkdir(exist_ok=True)
            download_path = download_path.resolve()
            logger.info(f"Downloading artifacts to: {download_path}")

            # Download all artifacts
            result = {}
            for key, typ in artifact_outputs.items():
                try:
                    if _is_artifact_type(typ):
                        artifact_path = download_artifact(
                            artifact=completed_step.outputs.artifacts[key],
                            path=download_path,
                        )
                        result[key] = artifact_path
                    else:
                        # Get from big parameters
                        result[key] = completed_step.outputs.parameters[key].value.recover() if isinstance(
                            completed_step.outputs.parameters[key].value,
                            (dflow.argo_objects.ArgoObjectDict, dflow.argo_objects.ArgoObjectList)) else \
                        completed_step.outputs.parameters[key].value
                except Exception as e:
                    logger.error(f"Failed to download output '{key}': {e}")
                    result[key] = None

            # Get parameters from output
            for key in parameter_outputs.keys():
                try:
                    result[key] = completed_step.outputs.parameters[key].value.recover() if isinstance(
                        completed_step.outputs.parameters[key].value,
                        (dflow.argo_objects.ArgoObjectDict, dflow.argo_objects.ArgoObjectList)) else \
                    completed_step.outputs.parameters[key].value
                except Exception as e:
                    logger.error(f"Failed to get parameter '{key}': {e}")
                    result[key] = None

            result["workflow_id"] = wf.id
            # result["status"] = "success"

            return result

        return wrapper

    return decorator