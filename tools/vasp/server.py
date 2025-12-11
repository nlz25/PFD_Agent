import uuid
import os
import argparse
from typing import Optional, Union, Literal, Dict, Any, List, Tuple
from pathlib import Path
import time
from dotenv import load_dotenv
from matcreator.tools.vasp.struture_tool import (
    analyze_crystal_structure as analyze_crystal_structure,
    search_materials_project as search_materials_project,
    create_crystal_structure as create_crystal_structure,
    make_supercell as make_supercell,
    )
from matcreator.tools.vasp.calculation import (
    vasp_nscf as vasp_nscf,
    vasp_relaxation as vasp_relaxation,
    vasp_scf as vasp_scf,
    check_status as check_status,
)
from matcreator.tools.vasp.python_plot import safe_execute_plot_code
import yaml
from matcreator.tools.util.sqldata import VaspCalculationDB
from pymatgen.core import Structure
import math
import numpy as np
from pymatgen.io.vasp import Kpoints
from ase.dft.kpoints import BandPath
from ase.io import read, write
load_dotenv(os.path.expanduser(".env"), override=True)
from datetime import datetime

VASP_SERVER_WORK_PATH = "/tmp/vasp_server"

def create_workpath(work_path=None):
    """
    Create the working directory for VaspAgent, and change the current working directory to it.

    Args:
        work_path (str, optional): The path to the working directory. If None, a default path will be used.
    Returns:
        str: The path to the working directory.
    """
    work_path = os.environ.get("VASP_SERVER_WORK_PATH", VASP_SERVER_WORK_PATH) + f"/{time.strftime('%Y%m%d%H%M%S')}"
    os.makedirs(work_path, exist_ok=True)
    os.chdir(work_path)
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


db_path = settings['db_path']
structure_path = settings['structure_path']
mp_api_key = os.getenv("MP_API_KEY") or settings.get("mp_api_key")
# 初始化SQLite数据库
db = VaspCalculationDB(db_path=db_path)

def write_record(calculation_id: str, data: dict):
    """写入计算记录到SQLite数据库"""
    db.write_record(calculation_id, data)
    return

def read_record(calculation_id: str):
    """从SQLite数据库读取计算记录"""
    return db.read_record(calculation_id)

def extract_calculation_data(calculation_ids: List[str]) -> Dict[str, Any]:
    """从计算记录中提取数据"""
    data = {}
    for calc_id in calculation_ids:
        record = read_record(calc_id)
        if record is not None:
            data[calc_id] = record
        else:
            data[calc_id] = None
    return data
    
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
    results = []
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
        result = vasp_relaxation(
            calculation_id=calculation_id,
            work_dir=settings['work_dir'],
            struct=struct,
            kpoints=kpts,
            incar_dict=incar,
            potcar_map=potcar_map
        )
        results.append(result)
    return {
        "success": all(r.get("status") == "success" for r in results),
        "results": results,
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
    if restart_id is not None:
        calculation_id = datetime.now().strftime("%Y%m%d%H%M%S_%f")
        restart_record = read_record(restart_id)
        if restart_record is None:
            return {"success": False, "error": f"Restart record {restart_id} not found"}
        else:
            struct = restart_record['structure']
            chgcar_path = os.path.join(restart_record['calculate_path'], "CHGCAR")
            wavecar_path = os.path.join(restart_record['calculate_path'], "WAVECAR")
            if kpoint_num is None:
                factor = 40 * np.power(struct.lattice.a * struct.lattice.b * struct.lattice.c / struct.lattice.volume , 1/3)
                kpoint_float = (factor/struct.lattice.a, factor/struct.lattice.b, factor/struct.lattice.c)
                kpoint_num = (max(math.ceil(kpoint_float[0]), 1), max(math.ceil(kpoint_float[1]), 1), max(math.ceil(kpoint_float[2]), 1))
            kpts = Kpoints.gamma_automatic(kpts = kpoint_num)
            incar = {}
            if soc:
                incar.update(settings['VASP_default_INCAR']['scf_soc'])
            else:
                incar.update(settings['VASP_default_INCAR']['scf_nsoc'])
            if incar_tags is not None:
                incar.update(incar_tags)
            
            # 执行计算
            return vasp_scf(
                calculation_id=calculation_id,
                work_dir=settings['work_dir'],
                struct=struct,
                kpoints=kpts,
                incar_dict=incar,
                chgcar_path=chgcar_path,
                wavecar_path=wavecar_path,
                potcar_map=potcar_map
            )

    else:
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
        results = []
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
            result = vasp_scf(
                calculation_id=calculation_id,
                work_dir=settings['work_dir'],
                struct=struct,
                kpoints=kpts,
                incar_dict=incar,
                potcar_map=potcar_map
            )
            results.append(result)
        return {
            "success": all(r.get("status") == "success" for r in results),
            "results": results,
        }


@mcp.tool()
def vasp_nscf_kpath_tool(restart_id: str, soc: bool=False, incar_tags: Optional[Dict] = None, kpath: Optional[str] = None, n_kpoints: Optional[int] = None, potcar_map: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Submit a VASP non-self-consistent field (NSCF) job for band structure. INCAR parameters should be consistent with the preceding SCF job where possible.
    
    Args:
        restart_id: ID of the preceding SCF calculation (required) to obtain converged charge density and wavefunction.
        soc: Whether to include spin–orbit coupling. Defaults to True.
        incar_tags: Additional INCAR parameters to merge with defaults. Use None unless explicitly specified by the user.
        kpath: K-point path. Options:
            - None: Use the auto-generated high-symmetry path from pymatgen.
            - str: User-specified path, e.g., "GMKG".
            Use None unless explicitly specified by the user.
        n_kpoints: Number of points per segment along the k-path.
        potcar_map: POTCAR mapping as {element: potcar}, e.g., {"Bi": "Bi_pv", "Se": "Se_pv"}. Use None unless explicitly specified by the user.
    Returns:
        A dict containing the submission result with keys:
        - calculation_id: Unique calculation identifier
        - success: Whether submission succeeded
        - error: Error message, if any
    """
    # 生成随机UUID
    calculation_id = str(uuid.uuid4())
    
    # 获取结构和前序计算文件
    scf_record = read_record(restart_id)
    if scf_record is None:
        return {"success": False, "error": f"SCF record {restart_id} not found"}
    struct: Structure = scf_record['structure']
    chgcar_path = os.path.join(scf_record['calculate_path'], "CHGCAR")
    wavecar_path = os.path.join(scf_record['calculate_path'], "WAVECAR")
    
    # 设置k点路径
    from pymatgen.symmetry.bandstructure import HighSymmKpath
    kpath_obj = HighSymmKpath(struct)
    if kpath_obj.kpath is None:
        return {"success": False, "error": "Failed to generate k-path for the structure"}

    n_kpoints = 16 if n_kpoints is None else n_kpoints
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
    return vasp_nscf(
        calculation_id=calculation_id,
        work_dir=settings['work_dir'],
        struct=struct,
        kpoints=kpts,
        incar_dict=incar,
        chgcar_path=chgcar_path,
        wavecar_path=wavecar_path,
        potcar_map=potcar_map
    )


@mcp.tool()
def vasp_nscf_uniform_tool(restart_id: str, soc: bool=False, incar_tags: Optional[Dict] = None, kpoint_num: Optional[tuple[int, int, int]] = None, potcar_map: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Submit a VASP non-self-consistent field (NSCF) job for band structure. INCAR parameters should be consistent with the preceding SCF job where possible.
    
    Args:
        restart_id: ID of the preceding SCF calculation (required) to obtain converged charge density and wavefunction.
        soc: Whether to include spin–orbit coupling. Defaults to True.
        incar_tags: Additional INCAR parameters to merge with defaults. Use None unless explicitly specified by the user.
        kpoint_num: K-point mesh as a tuple (nx, ny, nz). If not provided, an automatic density of 40 is used.
        potcar_map: POTCAR mapping as {element: potcar}, e.g., {"Bi": "Bi_pv", "Se": "Se_pv"}. Use None unless explicitly specified by the user.
    Returns:
        A dict containing the submission result with keys:
        - calculation_id: Unique calculation identifier
        - success: Whether submission succeeded
        - error: Error message, if any
    """
    # 生成随机UUID
    calculation_id = str(uuid.uuid4())
    
    # 获取结构和前序计算文件
    scf_record = read_record(restart_id)
    if scf_record is None:
        return {"success": False, "error": f"SCF record {restart_id} not found"}
    struct: Structure = scf_record['structure']
    chgcar_path = os.path.join(scf_record['calculate_path'], "CHGCAR")
    wavecar_path = os.path.join(scf_record['calculate_path'], "WAVECAR")
    
    # 设置k点

    if kpoint_num is None:
        factor = 100 * np.power(struct.lattice.a * struct.lattice.b * struct.lattice.c / struct.lattice.volume , 1/3)
        kpoint_float = (factor/struct.lattice.a, factor/struct.lattice.b, factor/struct.lattice.c)
        kpoint_num = (max(math.ceil(kpoint_float[0]), 1), max(math.ceil(kpoint_float[1]), 1), max(math.ceil(kpoint_float[2]), 1))
    kpts = Kpoints.gamma_automatic(kpts = kpoint_num)
    
    # 设置INCAR
    incar = {}
    if soc:
        incar.update(settings['VASP_default_INCAR']['nscf_soc'])
    else:
        incar.update(settings['VASP_default_INCAR']['nscf_nsoc'])
    if incar_tags is not None:
        incar.update(incar_tags)
    
    # 执行计算
    result = vasp_nscf(
        calculation_id=calculation_id,
        work_dir=settings['work_dir'],
        struct=struct,
        kpoints=kpts,
        incar_dict=incar,
        chgcar_path=chgcar_path,
        wavecar_path=wavecar_path,
        potcar_map=potcar_map
    )


@mcp.tool()
def check_calculation_status_tool(calculation_ids: List[str]) -> Dict[str, Any]:
    """
    Check the status of calculation jobs and return results.
    
    Args:
        calculation_ids: List of calculation IDs to check
    
    Returns:
        A dict mapping each calculation_id to the following structure:
        {
            calculation_id: {
                "slurm_id": "12345",
                "calc_type": "relaxation",
                "calculate_path": "/path/to/calculation",
                "status": "running/completed/failed/error",
                ... other result fields
            }
        }
    """
    # 从记录中获取计算信息
    calc_dict = {}
    llm_friendly_result = {}
    
    # 收集有效的计算记录
    for calc_id in calculation_ids:
        record = read_record(calc_id)
        if record is not None:
            calc_dict[calc_id] = record
    
    # 对有记录的计算检查SLURM状态
    if calc_dict:
        updated_results = check_status(calc_dict)
        # 更新记录
        for calc_id, result in updated_results.items():
            write_record(calc_id, result)
            calc_dict[calc_id] = result
    
    # 为所有计算ID构建返回结果
    for calc_id in calculation_ids:
        if calc_id in calc_dict:
            data = calc_dict[calc_id]
            llm_friendly_result[calc_id] = extract_llm_friendly_result(data)
        else:
            llm_friendly_result[calc_id] = {
                "error": f"Calculation record {calc_id} not found",
                "status": "unknown"
            }
    
    return llm_friendly_result

@mcp.tool()
def python_plot_tool(calculation_ids: List[str], plot_code: str, description: Optional[str] = None) -> Dict[str, Any]:
    """
    Execute Python plotting code based on the specified calculation result data.
    Note: Do not use plt.show() or plt.savefig() in plot_code, and do NOT call plt.close().
    For multiple figures, call this tool multiple times, otherwise images will be overwritten.

    Args:
        calculation_ids: List of calculation IDs to extract result data
        plot_code: Python plotting code. The data will be provided as a dict named `data`,
                    where keys are calculation_id and values are the corresponding results.
                
                    Available libraries and objects:
                    - matplotlib.pyplot (as plt)
                    - numpy (as np)
                    - pandas (as pd)
                    - pymatgen.core.Structure (as Structure)
                    - pymatgen.core.Element (as Element)
                    - pymatgen.core.Lattice (as Lattice)
                    - pymatgen.electronic_structure.bandstructure.BandStructure (as BandStructure)
                    - pymatgen.electronic_structure.dos.CompleteDos (as CompleteDos)
                    - pymatgen.io.vasp.Vasprun (as Vasprun)
                
                    Data format notes:
                    - `data` is a dict mapping calculation_id to its result dict
                    - Results contain raw pymatgen objects; you can directly use pymatgen methods for processing and plotting
                    - Common fields include:
                        * 'structure': pymatgen.core.Structure
                        * 'band_structure': pymatgen.electronic_structure.bandstructure.BandStructure  
                        * 'dos': pymatgen.electronic_structure.dos.CompleteDos
                        * 'total_energy': float
                        * 'efermi': float
                        * 'band_gap': dict with keys "energy", "direct", "transition"
                        * 'stress': float
                        * 'eigen_values': numpy.ndarray
                
                    Example 1: Plot band structure (use this when there is no special requirement)
                    ```python
                    # Get the first calculation data
                    band_calc_id = list(data.keys())[0]
                    calc_data = data[band_calc_id]
                    
                    # Plot band structure
                    if 'band_structure' in calc_data:
                        bs = calc_data['band_structure']  # BandStructure object
                        plt.figure(figsize=(10, 6))
                        
                        # Use pymatgen built-in plotter
                        from pymatgen.electronic_structure.plotter import BSPlotter
                        plotter = BSPlotter(bs)
                        plotter.get_plot(ylim=[-2, 2])
                        plt.axhline(y=0, color='r', linestyle='--', label='Fermi level')
                        plt.ylabel('Energy - E_fermi (eV)')
                        plt.xlabel('k-point')
                        plt.legend()
                    ```
                    Example 2: Plot band structure (for special requirements, e.g., overlay two band structures)
                    ```python
                    # Get the first calculation data
                    band_calc_id = list(data.keys())[0]
                    calc_data = data[band_calc_id]
                    
                    # Plot band structure
                    if 'band_structure' in calc_data:
                        bs = calc_data['band_structure']  # BandStructure object
                        # Extract plot data via pymatgen
                        from pymatgen.electronic_structure.plotter import BSPlotter
                        plotter = BSPlotter(bs)
                        ax = plt.gca()
                        data = plotter.bs_plot_data(bs)
                        # Draw
                        for spin in bs.bands:
                            ls = "-" if str(spin) == "1" else "--"
                            for dist, ene in zip(data["distances"], data["energy"][str(spin)], strict=True):
                                ax.plot(dist, ene.T, ls=ls, c="blue")

                        ax.set_xticks(data["ticks"]["distance"], data["ticks"]["label"])
                        for distance in data["ticks"]["distance"]:
                            ax.axvline(x=distance, color='black', linewidth=0.5)
                        plt.axhline(y=0, color='r', linestyle='--', label='Fermi level')
                        plt.xlim(np.min(data["distances"]), np.max(data["distances"]))
                        plt.ylim(-5, 5)
                        plt.ylabel('Energy - E_fermi (eV)')
                        plt.xlabel('k points')
                    ```
                    Example 3: Plot DOS
                    ```python
                    # Get DOS calculation data
                    dos_calc_id = list(data.keys())[0]
                    dos_data = data[dos_calc_id]
                    # Plot DOS
                    if 'dos' in dos_data:
                        dos = dos_data['dos']  # CompleteDos object
                        plt.figure(figsize=(8, 6))
                        # Use pymatgen built-in plotter
                        from pymatgen.electronic_structure.plotter import DosPlotter
                        plotter = DosPlotter()
                        plotter.add_dos("Total DOS", dos)
                        plotter.get_plot(xlim=[-5, 5])
                    ```
        description: Optional description for the figure
    
    Returns:
        A dict with keys:
        - success: Whether execution succeeded
        - error: Error message, if any
        - plot_path: Path to the generated image file
        - calculation_data_summary: Summary of the data used
    """
    
    # 提取计算数据
    calculation_data = extract_calculation_data(calculation_ids)
    # 检查是否有有效的计算数据
    valid_data = {k: v for k, v in calculation_data.items() if v is not None}
    if not valid_data:
        return {
            'success': False,
            'error': f'没有找到有效的计算数据。提供的计算ID: {calculation_ids}',
            'plot_path': None,
            'image_base64': None,
            'calculation_data_summary': None
        }
    
    # 创建数据摘要
    data_summary = {}
    for calc_id, data in valid_data.items():
        if data:
            summary = {
                'calculation_id': calc_id,
                'success': data.get('success', False),
                'total_energy': data.get('total_energy', None),
                'available_keys': list(data.keys())
            }
            data_summary[calc_id] = summary
    
    # 执行画图代码
    success, result, image_base64 = safe_execute_plot_code(plot_code, valid_data, settings['work_dir'])
    if success:
        return {
            'success': True,
            'error': None,
            'plot_path': result,
            'description': description
        }
    else:
        return {
            'success': False,
            'error': result,
            'plot_path': None,
            'calculation_data_summary': data_summary,
            'description': description
        }

@mcp.tool()
def search_materials_project_tool(search_criteria: Dict[str, Any], limit: int = 10) -> Dict[str, Any]:
    """
    Args:
        search_criteria: Search filters with the following keys:
            - material_id: str, e.g., "mp-1234"
            - formula: str, e.g., "TiO2"
            - elements: List[str], e.g., ["Ti", "O"]
            - exclude_elements: List[str], elements to exclude
            - band_gap: Tuple[float, float], band gap range (min, max), e.g., (1.0, 3.0)
            - energy_above_hull: Tuple[float, float], formation energy range (min, max)
            - num_sites: Tuple[int, int], number of sites range (min, max)
            - spacegroup_number: int, space group number
            - crystal_system: str, one of "Triclinic", "Monoclinic", "Orthorhombic", "Tetragonal", "Trigonal", "Hexagonal", "Cubic"
            - is_gap_direct: bool, whether the band gap is direct
        limit: Maximum number of results to return
        
    Returns:
        Dict with search results and download status
    """
    result = search_materials_project(api_key=mp_api_key, search_criteria=search_criteria, download_path=structure_path, limit=limit)
    return result

@mcp.tool()
def analyze_crystal_structure_tool(struct_path: str) -> Dict[str, Any]:
    """
    Args:
        struct_path: Structure input; can be a file path or a pymatgen Structure object
    """
    return analyze_crystal_structure(struct_path)

@mcp.tool()
def create_crystal_structure_tool(positions: List[List[float]], elements: List[str], lattice_vectors: List[List[float]], cartesian: bool) -> Dict[str, Any]:
    """
    Create a crystal structure given atomic positions, elements, and lattice vectors, and save it to a file.
    Args:
        positions: Atomic positions, e.g., [[x1, y1, z1], [x2, y2, z2], ...]
        elements: List of element symbols, e.g., ["Li", "F"]
        lattice_vectors: Lattice vectors, e.g., [[a1, b1, c1], [a2, b2, c2], [a3, b3, c3]]
        cartesian: Whether positions are in Cartesian coordinates
    Returns:
        Dict containing the path to the created structure file
    """
    return create_crystal_structure(np.array(positions), elements, np.array(lattice_vectors), cartesian, structure_path)

@mcp.tool()
def make_supercell_tool(struct_path: str, supercell_matrix: List[List[int]]) -> Dict[str, Any]:
    """
    Args:
        struct_path: Structure input; can be a file path or a pymatgen Structure object
        supercell_matrix: 3 by 3 matrix, supercell matrix.
    return:
        Dict containing the path to the created structure file
    """
    return make_supercell(struct_path, supercell_matrix)

@mcp.tool()
def list_calculations_tool(calc_type: Optional[str] = None, status: Optional[str] = None, limit: Optional[int] = 50) -> Dict[str, Any]:
    """
    List calculation records.
    
    Args:
        calc_type: Filter by calculation type ("relaxation", "scf", "nscf")
        status: Filter by status ("pending", "running", "completed", "failed", "error")
        limit: Maximum number of records to return, default 50
    
    Returns:
        result_dict: A dict containing the list of records
    """
    try:
        calculations = db.list_calculations(calc_type=calc_type, status=status, limit=limit)
        return {
            "success": True,
            "calculations": calculations,
            "total_count": len(calculations)
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "calculations": []
        }

@mcp.tool()
def get_database_statistics_tool() -> Dict[str, Any]:
    """
    Get database statistics.
    
    Returns:
        Database statistics including total records, distribution by type and status, etc.
    """
    try:
        stats = db.get_statistics()
        return {
            "success": True,
            "statistics": stats
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "statistics": {}
        }

@mcp.tool()
def delete_calculation_tool(calculation_id: str) -> Dict[str, Any]:
    """
    Delete a calculation record.
    
    Args:
        calculation_id: ID of the record to delete
    
    Returns:
        Result of the deletion operation
    """
    try:
        success = db.delete_record(calculation_id)
        if success:
            return {
                "success": True,
                "message": f"成功删除计算记录 {calculation_id}"
            }
        else:
            return {
                "success": False,
                "error": f"计算记录 {calculation_id} 不存在"
            }
    except Exception as e:
        return {
            "success": False,
            "error": f"删除计算记录时出错: {str(e)}"
        }

@mcp.tool()
def check_files_exist(file_paths: list[str]) -> Dict[str, Any]:
    """
    Check whether each file in the list exists.
    
    Args:
        file_paths: List of file paths to check
    
    Returns:
        exist_dict: A dict of path -> bool, e.g.,
        {
            "file_path1": True,
            "file_path2": True,
            ......
        }
    """

    files_exist = {}
    for path in file_paths:
        files_exist[path] = os.path.exists(path)
    return files_exist

@mcp.tool()
def read_calc_results_from_db(calc_ids: list[str]) -> Dict[str, Any]:
    """
    Read calculation records.
    Args:
        calc_ids: List of calculation IDs to read
    
    Returns:
        A dict mapping each calculation_id to the following structure:
        {
            calculation_id: {
                "slurm_id": "12345",
                "calc_type": "relaxation",
                "calculate_path": "/path/to/calculation",
                "status": "running/completed/failed/error",
                ... other result fields
            }
        }
    """
    llm_friendly_results = {}
    for calc_id in calc_ids:
        data = read_record(calc_id)
        if data is not None:
            llm_friendly_results[calc_id] = extract_llm_friendly_result(data)
        else:
            llm_friendly_results[calc_id] = {"error": "calc_id not found!"}
    return llm_friendly_results

if __name__ == "__main__":
    create_workpath()
    mcp.run(transport=args.transport)