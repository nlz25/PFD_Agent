from typing import List, Union, Optional, Tuple, Dict, Any
import os
from pathlib import Path
import subprocess
import select


def run_command(
        cmd,
        shell=True
):
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=shell,
        executable='/bin/bash'
    )
    out = ""
    err = ""
    while True:
        readable, _, _ = select.select(
            [process.stdout, process.stderr], [], [])

        for fd in readable:
            if fd == process.stdout:
                line = process.stdout.readline()
                print(line.decode()[:-1])
                out += line.decode()
            elif fd == process.stderr:
                line = process.stderr.readline()
                print("STDERR:", line.decode()[:-1])
                err += line.decode()

        return_code = process.poll()
        if return_code is not None:
            break
    return return_code, out, err

def get_physical_cores():
    """
    """
    # 对于Linux系统，解析/proc/cpuinfo文件
    with open('/proc/cpuinfo', 'r') as f:
        cpuinfo = f.read()
    
    # 统计物理ID的数量和每个物理ID下的核心数
    physical_ids = set()
    cores_per_socket = {}
    
    for line in cpuinfo.split('\n'):
        if line.startswith('physical id'):
            phys_id = line.split(':')[1].strip()
            physical_ids.add(phys_id)
        elif line.startswith('cpu cores'):
            cores = int(line.split(':')[1].strip())
            if phys_id in cores_per_socket:
                cores_per_socket[phys_id] = max(cores_per_socket[phys_id], cores)
            else:
                cores_per_socket[phys_id] = cores
    
    # 计算总物理核心数
    if physical_ids and cores_per_socket:
        return sum(cores_per_socket.values())
    else:
        # 备选方法：使用lscpu命令
        output = subprocess.check_output('lscpu', shell=True).decode()
        for line in output.split('\n'):
            if line.startswith('Core(s) per socket:'):
                cores_per_socket = int(line.split(':')[1].strip())
            elif line.startswith('Socket(s):'):
                sockets = int(line.split(':')[1].strip())
        return cores_per_socket * sockets


def run_vasp(job_path: Path,
            log_file: Optional[str] = "vasp.log") -> None:
    """
    Run the vasp on the given job paths.
    If job_path is a list, it will run the command for each path.
    If job_path is a single Path, it will run the command for that path.
    """
    cwd = os.getcwd()

    # submit_type = os.environ.get("VASPAGENT_SUBMIT_TYPE", "local").lower()
    # if submit_type == "local":


    physical_cores = get_physical_cores()
    command_cmd = os.environ.get("VASP_COMMAND", 
                                    f"OMP_NUM_THREADS=1 mpirun -np {physical_cores} vasp") + f" > {log_file} 2>&1"     

    if not job_path.is_dir():
        raise ValueError(f"{job_path} is not a valid directory.")
    
    os.chdir(job_path)           
    return_code, out, err = run_command([command_cmd])
    os.chdir(cwd)
    if return_code != 0:
        raise RuntimeError(f"VASP command failed with error: {err}")
