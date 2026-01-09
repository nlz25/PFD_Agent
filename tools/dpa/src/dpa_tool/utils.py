import os
from contextlib import (
    contextmanager,
)
from functools import (
    wraps,
)
from pathlib import (
    Path,
)
import importlib

import dflow
from dflow.config import (
    config,
    s3_config,
)
from dflow.plugins import (
    bohrium,
)


import time
import traceback
import uuid

@contextmanager
def set_directory(path: Path):
    """Sets the current working path within the context.

    Parameters
    ----------
    path : Path
        The path to the cwd

    Yields
    ------
    None

    Examples
    --------
    >>> with set_directory("some_path"):
    ...    do_something()
    """
    cwd = Path().absolute()
    path.mkdir(exist_ok=True, parents=True)
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(cwd)
        
        
def generate_work_path(create: bool = True) -> str:
	"""Return a unique work dir path and create it by default."""
	calling_function = traceback.extract_stack(limit=2)[-2].name
	current_time = time.strftime("%Y%m%d%H%M%S")
	random_string = str(uuid.uuid4())[:8]
	work_path = f"{current_time}.{calling_function}.{random_string}"
	if create:
		os.makedirs(work_path, exist_ok=True)
	return work_path


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