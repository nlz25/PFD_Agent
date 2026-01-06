import uuid
import os
import argparse
from typing import Optional, Union, Literal, Dict, Any, List, Tuple
from pathlib import Path
import time
from matcreator.tools.vasp.calculation import (
    vasp_relaxation as vasp_relaxation,
    vasp_scf as vasp_scf,
    vasp_scf_results as vasp_scf_results,
)
import yaml
from pymatgen.core import Structure
import math
import numpy as np
from pymatgen.io.vasp import Kpoints
from ase.dft.kpoints import BandPath
from ase.io import read, write
from datetime import datetime
from dpdispatcher import Machine, Resources, Task, Submission
from dotenv import load_dotenv
load_dotenv(os.path.expanduser(".env"), override=True)
VASP_SERVER_WORK_PATH = "/tmp/vasp_server"

machine={
    "batch_type": "Bohrium",
    "context_type": "BohriumContext",
    "local_root" : "/tmp/vasp_server",
    "remote_profile":{
        "email": "2023400280@qq.com",
        "password": "20000612Zdj!", 
        "program_id": 29496,
        "keep_backup":True,
        "input_data":{
            "job_type": "container",
            "grouped":True,
            "job_name": "vasp_opt",
            "scass_type":"c32_m64_cpu",
            "platform": "ali",
            "image_name":"registry.dp.tech/dptech/prod-15454/vasp:5.4.4"
        }
}
}

machine = Machine.load_from_dict(machine)
resources={
    "group_size":4
}
resources = Resources.load_from_dict(resources)


def create_workpath(work_path=None):
    """
    Create the working directory for VaspAgent, and change the current working directory to it.

    Args:
        work_path (str, optional): The path to the working directory. If None, a default path will be used.
    Returns:
        str: The path to the working directory.
    """
    work_path = os.environ.get("VASP_SERVER_WORK_PATH", VASP_SERVER_WORK_PATH)
    os.makedirs(work_path, exist_ok=True)
    # os.chdir(work_path)
    print(f"Changed working directory to: {work_path}")
    return work_path    

def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Vasp_Agent Command Line Interface")
    
    parser.add_argument(
        "--transport",
        type=str,
        default="sse",
        choices=["sse", "streamable-http"],
        help="Transport protocol to use (default: sse), choices: sse, streamable-http"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="fastmcp",
        choices=["fastmcp", "dp"],
        help="Model to use (default: dp), choices: fastmcp, dp"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=50001,
        help="Port to run the MCP server on (default: 50001)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Host to run the MCP server on (default: localhost)"
    )
    args = parser.parse_args()
    return args


args = parse_args()  
if args.model == "dp":
    from dp.agent.server import CalculationMCPServer
    mcp = CalculationMCPServer(
            "VaspServer",
            host=args.host,
            port=args.port
        )
elif args.model == "fastmcp":
    from mcp.server.fastmcp import FastMCP
    mcp = FastMCP(
            "VaspServer",
            host=args.host,
            port=args.port
        )

current_dir = Path(__file__).parent
config_path = current_dir/"config.yaml"
    
with open(config_path, "r") as f:
    settings = yaml.safe_load(f)


    
@mcp.tool()
def vasp_relaxation_tool(structure_path: Path, incar_tags: Optional[Dict] = None, kpoint_num: Optional[tuple[int, int, int]] = None, frames: Optional[List[int]] = None, potcar_map: Optional[Dict] = None) -> Dict[str, Any]:
    """
        Submit VASP structural relaxation jobs.
        
        Args:
            structure_paths: Paths to the structure file (support extxyz etc.).
            incar_tags: Additional INCAR parameters to merge with defaults. Use None unless explicitly specified by the user.
            kpoint_num: K-point mesh as a tuple (nx, ny, nz). If not provided, an automatic density of 40 is used.
            frames: select specific frame indices, default: all
            potcar_map: POTCAR mapping as {element: potcar}, e.g., {"Bi": "Bi_d", "Se": "Se"}. Use None unless explicitly specified by the user.
        Returns:
            A dict containing the submission result with keys:
            - calculation_id: Unique calculation identifier
            - success: Whether submission succeeded
            - error: Error message, if any
    """
    # 转换输入参数
    cif_dir = Path("cif_dir")
    cif_dir.mkdir(exist_ok=True)
    cif_path_ls=[]
    for idx, atoms in enumerate(read(str(structure_path), index=':',format="extxyz")):
        if frames is not None and idx not in frames:
            continue
        cif_path = cif_dir / f"frame_{idx:04d}.cif"
        try:
            write(str(cif_path), atoms, format="cif")
        except Exception as e:
            results.append({"message": f"Failed to write CIF for frame {idx}: {e}", "frame_index": idx})
            continue
        cif_path_ls.append(str(cif_path))

    task_list = []
    calc_dir_ls = []
    for cifpath in cif_path_ls:
        # 生成随机UUID
        calculation_id = datetime.now().strftime("%Y%m%d%H%M%S_%f")
        struct = Structure.from_file(cifpath)
        if kpoint_num is None:
            factor = 40 * np.power(struct.lattice.a * struct.lattice.b * struct.lattice.c / struct.lattice.volume , 1/3)
            kpoint_float = (factor/struct.lattice.a, factor/struct.lattice.b, factor/struct.lattice.c)
            kpt_num_this = (max(math.ceil(kpoint_float[0]), 1), max(math.ceil(kpoint_float[1]), 1), max(math.ceil(kpoint_float[2]), 1))
        else:
        # 用户显式传了 kpoint_num：所有结构共用这一套
            kpt_num_this = kpoint_num
        # kpts = Kpoints.gamma_automatic(kpts = kpoint_num)
        kpts = Kpoints.gamma_automatic(kpts=kpt_num_this)
        incar = {}
        incar.update(settings['VASP_default_INCAR']['relaxation'])
        if incar_tags is not None:
            incar.update(incar_tags)
            
        # 执行计算
        task, calc_dir = vasp_relaxation(
            calculation_id=calculation_id,
            work_dir=settings['work_dir'],
            struct=struct,
            kpoints=kpts,
            incar_dict=incar,
            potcar_map=potcar_map
        )
        
        if not isinstance(task, Task):
            raise TypeError(f"vasp_scf must return Task, got {type(task)}: {task}")

        task_list.append(task)
        calc_dir_ls.append(calc_dir)

    submission = Submission(
        work_base="./",
        machine=machine,
        resources=resources,
        task_list=task_list,
        forward_common_files=[],
        backward_common_files=[],
    )

    submission.run_submission()    
    
    return {
        "success": True,
        "calc_dir_list":[str(vasp_relax_dir) for vasp_relax_dir in calc_dir_ls]
    }


@mcp.tool()
def vasp_scf_tool(structure_path: Path, restart_id: Optional[str] = None, soc: bool=False, incar_tags: Optional[Dict] = None, kpoint_num: Optional[tuple[int, int, int]] = None, frames: Optional[List[int]] = None, potcar_map: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Submit VASP self-consistent field (SCF) jobs.
    
    Args:
        restart_id: ID of a previous calculation. If provided, reuse its structure and charge density.
        structure_paths: Paths to the structure file; required when restart_id is not provided.
        soc: Whether to include spin–orbit coupling. Defaults to False.
        incar_tags: Additional INCAR parameters to merge with defaults. Use None unless explicitly specified by the user.
        kpoint_num: K-point mesh as a tuple (nx, ny, nz). If not provided, an automatic density of 40 is used.
        frames: select specific frame indices, default: all
        potcar_map: POTCAR mapping as {element: potcar}, e.g., {"Bi": "Bi_pv", "Se": "Se_pv"}. Use None unless explicitly specified by the user.
    Returns:
        A dict containing the submission result with keys:
        - calculation_id: Unique calculation identifier
        - success: Whether submission succeeded
        - error: Error message, if any
    """

    cif_dir = Path("cif_dir")
    cif_dir.mkdir(exist_ok=True)
    cif_path_ls=[]
    for idx, atoms in enumerate(read(str(structure_path), index=':',format="extxyz")):
        if frames is not None and idx not in frames:
            continue
        cif_path = cif_dir / f"frame_{idx:04d}.cif"
        try:
            write(str(cif_path), atoms, format="cif")
        except Exception as e:
            results.append({"message": f"Failed to write CIF for frame {idx}: {e}", "frame_index": idx})
            continue
        cif_path_ls.append(str(cif_path))

    task_list = []
    calc_dir_ls = []
    for cifpath in cif_path_ls:
        # 生成随机UUID
        calculation_id = datetime.now().strftime("%Y%m%d%H%M%S_%f")
        struct = Structure.from_file(cifpath)
        if kpoint_num is None:
            factor = 40 * np.power(struct.lattice.a * struct.lattice.b * struct.lattice.c / struct.lattice.volume , 1/3)
            kpoint_float = (factor/struct.lattice.a, factor/struct.lattice.b, factor/struct.lattice.c)
            kpt_num_this = (max(math.ceil(kpoint_float[0]), 1), max(math.ceil(kpoint_float[1]), 1), max(math.ceil(kpoint_float[2]), 1))
        else:
        # 用户显式传了 kpoint_num：所有结构共用这一套
            kpt_num_this = kpoint_num
        # kpts = Kpoints.gamma_automatic(kpts = kpoint_num)
        kpts = Kpoints.gamma_automatic(kpts=kpt_num_this)
        incar = {}
        if soc:
            incar.update(settings['VASP_default_INCAR']['scf_soc'])
        else:
            incar.update(settings['VASP_default_INCAR']['scf_nsoc'])
        if incar_tags is not None:
            incar.update(incar_tags)
            
        # 执行计算
        task, calc_dir = vasp_scf(
            calculation_id=calculation_id,
            work_dir=settings['work_dir'],
            struct=struct,
            kpoints=kpts,
            incar_dict=incar,
            potcar_map=potcar_map
        )

        if not isinstance(task, Task):
            raise TypeError(f"vasp_scf must return Task, got {type(task)}: {task}")

        task_list.append(task)
        calc_dir_ls.append(calc_dir)


    submission = Submission(
        work_base="./",
        machine=machine,
        resources=resources,
        task_list=task_list,
        forward_common_files=[],
        backward_common_files=[],
    )

    submission.run_submission()  
    return {
        "status": "success",
        "calc_dir_list":[str(vasp_scf_dir) for vasp_scf_dir in calc_dir_ls]
    }



@mcp.tool()
def vasp_scf_results_tool(
        scf_work_dir_ls: Union[List[Path], Path],
    ) -> Dict[str, Any]:
    """
        Collect results from VASP SCF calculation.

        Args:
            scf_work_dir_ls (List[Path]): A list of path to the directories containing the VASP SCF calculation output files.
        Returns:
            A dictionary containing the path to output file of VASP calculation in extxyz format. The extxyz file contains the atomic structure and the total energy, atomic forces, etc.
        """
    return vasp_scf_results(
            scf_work_dir_ls=scf_work_dir_ls
        )

if __name__ == "__main__":
    create_workpath()
    mcp.run(transport=args.transport)