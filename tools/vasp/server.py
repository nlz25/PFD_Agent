import uuid
import os
import argparse
from typing import Optional, Union, Literal, Dict, Any, List, Tuple
from pathlib import Path
import time
from vasp_tools.calculation import (
    vasp_relaxation as vasp_relaxation,
    vasp_scf as vasp_scf,
    vasp_scf_results as vasp_scf_results,
)

from vasp_tools.calculation import vasp_nscf, read_calculation_result
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

_script_dir = Path(__file__).parent
load_dotenv(_script_dir / ".env", override=True)

VASP_SERVER_WORK_PATH = "/tmp/vasp_server"

machine={
    "batch_type": "Bohrium",
    "context_type": "BohriumContext",
    "local_root" : "/tmp/vasp_server",
    "remote_profile":{
        "email": os.environ.get("BOHRIUM_USERNAME"),
        "password": os.environ.get("BOHRIUM_PASSWORD"),
        "program_id": int(os.environ.get("BOHRIUM_PROJECT_ID")),
        "keep_backup":True,
        "input_data":{
            "job_type": "container",
            "grouped":True,
            "job_name": "vasp_opt",
            "scass_type":os.environ.get("BOHRIUM_VASP_MACHINE"),
            "platform": "ali",
            "image_name":os.environ.get("BOHRIUM_VASP_IMAGE")
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
            structure_path: Path to the structure file (support extxyz etc.).
            incar_tags: Additional INCAR parameters to merge with defaults. Use None unless explicitly specified by the user.
            kpoint_num: K-point mesh as a tuple (nx, ny, nz). If not provided, an automatic density of 40 is used.
            frames: select specific frame indices, default: all
            potcar_map: POTCAR mapping as {element: potcar}, e.g., {"Bi": "Bi_d", "Se": "Se"}. Use None unless explicitly specified by the user.
        Returns:
            A dict containing the submission result with keys:
            - calculation_path: Unique calculation identifier
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
            cif_path_ls.append(str(cif_path))
        except Exception as e:
            print(f"Warning: Failed to write CIF for frame {idx}: {e}")
            continue

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
        structure_path: Path to the structure file; required when restart_id is not provided.
        soc: Whether to include spin–orbit coupling. Defaults to False.
        incar_tags: Additional INCAR parameters to merge with defaults. Use None unless explicitly specified by the user.
        kpoint_num: K-point mesh as a tuple (nx, ny, nz). If not provided, an automatic density of 40 is used.
        frames: select specific frame indices, default: all
        potcar_map: POTCAR mapping as {element: potcar}, e.g., {"Bi": "Bi_pv", "Se": "Se_pv"}. Use None unless explicitly specified by the user.
    Returns:
        A dict containing the submission result with keys:
        - calculation_path: Unique calculation identifier
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
            cif_path_ls.append(str(cif_path))
        except Exception as e:
            print(f"Warning: Failed to write CIF for frame {idx}: {e}")
            continue

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



@mcp.tool()
def vasp_nscf_kpath_tool(scf_dir_ls: Union[List[Path], Path], soc: bool=False, incar_tags: Optional[Dict] = None, kpath: Optional[str] = None, n_kpoints: Optional[int] = None, potcar_map: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Submit a VASP non-self-consistent field (NSCF) job for band structure. INCAR parameters should be consistent with the preceding SCF job where possible.
    
    Args:
        scf_dir_ls: path to the structure file; required when restart_id is not provided.
        soc: Whether to include spin–orbit coupling. Defaults to False.
        incar_tags: Additional INCAR parameters to merge with defaults. Use None unless explicitly specified by the user.
        kpath: K-point path. Options:
            - None: Use the auto-generated high-symmetry path from pymatgen.
            - str: User-specified path, e.g., "GMKG".
            Use None unless explicitly specified by the user.
        n_kpoints: Number of points per segment along the k-path.
        potcar_map: POTCAR mapping as {element: potcar}, e.g., {"Bi": "Bi_pv", "Se": "Se_pv"}. Use None unless explicitly specified by the user.
    Returns:
        A dict containing the submission result with keys:
        - calculation_path: Unique calculation identifier
        - success: Whether submission succeeded
        - error: Error message, if any
    """

    if isinstance(scf_dir_ls, Path):
        scf_dir_ls = [scf_dir_ls] 
    
    task_list = []
    calc_dir_ls = []
    n_kpoints = 16 if n_kpoints is None else n_kpoints
    for scf_dir in scf_dir_ls:
        # 生成随机UUID
        calculation_id = datetime.now().strftime("%Y%m%d%H%M%S_%f")
        chg_path = str(scf_dir.absolute()/"CHGCAR")
        wave_path = str(scf_dir.absolute()/"WAVECAR")
        struct = Structure.from_file(str(scf_dir.absolute()/"CONTCAR"))

        # 设置k点路径
        from pymatgen.symmetry.bandstructure import HighSymmKpath
        kpath_obj = HighSymmKpath(struct,symprec=0.01)
        if kpath_obj.kpath is None:
            return {"success": False, "error": "Failed to generate k-path for the structure"}


        if kpath is None:
            # 使用pymatgen自动生成的高对称路径
            kpts = Kpoints.automatic_linemode(n_kpoints, kpath_obj)
        else:
            # 使用用户指定的路径
            kpts_ase: BandPath = struct.to_ase_atoms().get_cell().bandpath(kpath, npoints=n_kpoints, eps=1e-2)
            high_sym_points = []
            labels = []
            high_sym_points.append(kpts_ase.special_points[kpath[0]])
            labels.append(kpath[0])
            kpath_list = list(kpath)
            for key in kpath_list[1:-1]:
                high_sym_points.append(kpts_ase.special_points[key])
                labels.append(key)
                high_sym_points.append(kpts_ase.special_points[key])
                labels.append(key)
            high_sym_points.append(kpts_ase.special_points[kpath[-1]])
            labels.append(kpath[-1])
            kpts = Kpoints(
                comment="User specified k-path",
                style=Kpoints.supported_modes.Line_mode,
                num_kpts=n_kpoints,
                kpts=high_sym_points,
                labels=labels,
                coord_type="Reciprocal"
            )
    
        # 设置INCAR
        incar = {}
        if soc:
            incar.update(settings['VASP_default_INCAR']['nscf_soc'])
        else:
            incar.update(settings['VASP_default_INCAR']['nscf_nsoc'])
        if incar_tags is not None:
            incar.update(incar_tags)

        # 执行计算
        task, calc_dir = vasp_nscf(
            calculation_id=calculation_id,
            work_dir=settings['work_dir'],
            struct=struct,
            kpoints=kpts,
            incar_dict=incar,
            chgcar_path=chg_path,
            wavecar_path=wave_path,
            potcar_map=potcar_map
        )

        if not isinstance(task, Task):
            raise TypeError(f"vasp_nscf must return Task, got {type(task)}: {task}")

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
        "calc_dir_list":[str(vasp_nscf_kpath_dir) for vasp_nscf_kpath_dir in calc_dir_ls]
    }



@mcp.tool()
def vasp_nscf_uniform_tool(scf_dir_ls: Union[List[Path], Path], soc: bool=False, incar_tags: Optional[Dict] = None, kpoint_num: Optional[tuple[int, int, int]] = None, potcar_map: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Submit a VASP non-self-consistent field (NSCF) job for band structure. INCAR parameters should be consistent with the preceding SCF job where possible.
    
    Args:
        scf_dir_ls: path of the preceding SCF calculation (required) to obtain converged charge density and wavefunction.
        soc: Whether to include spin–orbit coupling. Defaults to False.
        kpoint_num: K-point mesh as a tuple (nx, ny, nz). If not provided, an automatic density of 40 is used.
        incar_tags: Additional INCAR parameters to merge with defaults. Use None unless explicitly specified by the user.
        potcar_map: POTCAR mapping as {element: potcar}, e.g., {"Bi": "Bi_pv", "Se": "Se_pv"}. Use None unless explicitly specified by the user.
    Returns:
        A dict containing the submission result with keys:
        - calculation_path: Unique calculation identifier
        - success: Whether submission succeeded
        - error: Error message, if any
    """

    if isinstance(scf_dir_ls, Path):
        scf_dir_ls = [scf_dir_ls] 
    
    task_list = []
    calc_dir_ls = []
    for scf_dir in scf_dir_ls:
        # 生成随机UUID
        calculation_id = datetime.now().strftime("%Y%m%d%H%M%S_%f")
        chg_path = str(scf_dir.absolute()/"CHGCAR")
        wave_path = str(scf_dir.absolute()/"WAVECAR")
        struct = Structure.from_file(str(scf_dir.absolute()/"CONTCAR"))

        if kpoint_num is None:
            factor = 100 * np.power(struct.lattice.a * struct.lattice.b * struct.lattice.c / struct.lattice.volume , 1/3)
            kpoint_float = (factor/struct.lattice.a, factor/struct.lattice.b, factor/struct.lattice.c)
            kpt_num_this = (max(math.ceil(kpoint_float[0]), 1), max(math.ceil(kpoint_float[1]), 1), max(math.ceil(kpoint_float[2]), 1))
        else:
            # 用户显式传了 kpoint_num：所有结构共用这一套
            kpt_num_this = kpoint_num
        # kpts = Kpoints.gamma_automatic(kpts = kpoint_num)
        kpts = Kpoints.gamma_automatic(kpts=kpt_num_this)
        
        # 设置INCAR
        incar = {}
        if soc:
            incar.update(settings['VASP_default_INCAR']['nscf_soc'])
        else:
            incar.update(settings['VASP_default_INCAR']['nscf_nsoc'])
        if incar_tags is not None:
            incar.update(incar_tags)

        # 执行计算
        task, calc_dir = vasp_nscf(
            calculation_id=calculation_id,
            work_dir=settings['work_dir'],
            struct=struct,
            kpoints=kpts,
            incar_dict=incar,
            chgcar_path=chg_path,
            wavecar_path=wave_path,
            potcar_map=potcar_map
        )

        if not isinstance(task, Task):
            raise TypeError(f"vasp_nscf must return Task, got {type(task)}: {task}")

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
        "calc_dir_list":[str(vasp_nscf_uniform_dir) for vasp_nscf_uniform_dir in calc_dir_ls]
    }


@mcp.tool()
def plot_tool(calc_type: str, calculate_path: str) -> Dict[str, Any]:
    """
    根据计算类型读取结果用于作图
    Args:
    alc_type:计算类型
    calculate_path:文件路径
    Returns:
    A dict containing some results(eg.band_structure,dos,band_gap)
    """
    return read_calculation_result(calc_type,calculate_path)


if __name__ == "__main__":
    create_workpath()
    mcp.run(transport=args.transport)