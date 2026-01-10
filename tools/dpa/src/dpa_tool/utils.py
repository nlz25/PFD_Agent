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

from typing import List, Tuple, Union, Optional
import subprocess
import sys
import shlex
import selectors
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