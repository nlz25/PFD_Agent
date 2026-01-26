import os
from pymatgen.core import Element, Structure
from pymatgen.io.vasp import VaspInput, Vasprun, Kpoints, Poscar, Chgcar, Potcar, Outcar
from pathlib import Path
from typing import Optional, Dict, Any, Union, List
from .common import run_vasp
import traceback
from dpdispatcher import Machine, Resources, Task, Submission
from ase.io import read, write
import dpdata
from .utils import dpdata2ase_single, generate_work_path
import shutil
from pymatgen.io.vasp import VaspInput, Vasprun, Kpoints, Poscar, Chgcar, Potcar, Outcar


def vasp_relaxation(calculation_id: str, work_dir: str, struct: Structure, 
                   kpoints: Kpoints, incar_dict: dict, potcar_map: Optional[Dict] = None) -> Dict[str, Any]:
    """
    提交VASP结构优化计算任务
    
    参数:
        calculation_id: 计算ID
        work_dir: 工作目录
        struct: 晶体结构
        kpoints: K点设置
        incar_dict: 额外的INCAR参数，会与默认设置合并。除非用户指定，不要擅自修改。
        
    返回:
        Dict包含success、error等信息
    """
    if potcar_map is None:
        potcar_map = {}
    try:
        Name = calculation_id
        calc_dir = os.path.abspath(f'{work_dir}/{Name}')
        calc_dir_1 = (f'tmp/vasp_server/{Name}')
        # 创建VASP输入文件
        # 手动获取元素列表，确保顺序与POSCAR一致
        poscar = Poscar(struct)
        unique_species = []
        for species in poscar.structure.species:
            species: Element
            if unique_species:
                if species.symbol != unique_species[-1]:
                    if species.symbol not in potcar_map:
                        potcar_map[species.symbol] = species.symbol
                    unique_species.append(species.symbol)
            else:
                if species.symbol not in potcar_map:
                    potcar_map[species.symbol] = species.symbol
                unique_species.append(species.symbol)
        potcar_symbols = []
        for symbol in unique_species:
            potcar_symbols.append(potcar_map[symbol])

        vasp_input = VaspInput(
            poscar=poscar,
            incar=incar_dict,
            kpoints=kpoints,
            potcar=Potcar(potcar_symbols)
        )
        
        # 准备结构优化目录
        vasp_input.write_input(calc_dir)
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Performing vasp calculation failed: {e}",
            "traceback": traceback.format_exc()}

    submit_type = os.environ.get("VASPAGENT_SUBMIT_TYPE", "local").lower()

    if submit_type == "local":
        run_vasp(calc_dir) 

    elif submit_type == "bohrium":
        task=Task(
            command="source /opt/intel/oneapi/setvars.sh && mpirun -n 32 vasp_std",
            task_work_path=Name,
            forward_files=["POSCAR","INCAR","POTCAR","KPOINTS"],
            backward_files=["OSZICAR","CONTCAR","OUTCAR","vasprun.xml"]
        )
        return task, calc_dir
    
    else:
        raise ValueError("Invalid VASPAGENT_SUBMIT_TYPE. Must be 'local' or 'bohrium'.")


def vasp_scf(calculation_id: str, work_dir: str, struct: Structure, 
            kpoints: Kpoints, incar_dict: dict, chgcar_path: Optional[str] = None, 
            wavecar_path: Optional[str] = None, potcar_map: Optional[Dict] = None):
    """
    提交VASP自洽场计算任务
    
    参数:
        calculation_id: 计算ID
        work_dir: 工作目录
        struct: 晶体结构
        kpoints: K点设置
        incar_dict: 额外的INCAR参数，会与默认设置合并。除非用户指定，不要擅自修改。
        chgcar_path: CHGCAR文件路径
        wavecar_path: WAVECAR文件路径
        
    返回:
        Dict包含success、error等信息
    """
    if potcar_map is None:
        potcar_map = {}
    try:
        Name = calculation_id
        calc_dir = os.path.abspath(f'{work_dir}/{Name}')
        # 创建VASP输入文件
        # 手动获取元素列表，确保顺序与POSCAR一致
        poscar = Poscar(struct)
        unique_species = []
        for species in poscar.structure.species:
            species: Element
            if unique_species:
                if species.symbol != unique_species[-1]:
                    if species.symbol not in potcar_map:
                        potcar_map[species.symbol] = species.symbol
                    unique_species.append(species.symbol)
            else:
                if species.symbol not in potcar_map:
                    potcar_map[species.symbol] = species.symbol
                unique_species.append(species.symbol)
        potcar_symbols = []
        for symbol in unique_species:
            potcar_symbols.append(potcar_map[symbol])

        vasp_input = VaspInput(
            poscar=poscar,
            incar=incar_dict,
            kpoints=kpoints,
            potcar=Potcar(potcar_symbols)
        )

        # 准备自洽场计算目录
        vasp_input.write_input(calc_dir)
    
    except Exception as e:
        return {
            "status": "error",
            "message": f"Performing vasp calculation failed: {e}",
            "traceback": traceback.format_exc()}


    submit_type = os.environ.get("VASPAGENT_SUBMIT_TYPE", "local").lower()

    if submit_type == "local":
        run_vasp(calc_dir) 

    elif submit_type == "bohrium":
        task=Task(
            command="source /opt/intel/oneapi/setvars.sh && mpirun -n 32 vasp_std",
            task_work_path=Name,
            forward_files=["POSCAR","INCAR","POTCAR","KPOINTS"],
            backward_files=["OSZICAR","CONTCAR","OUTCAR","vasprun.xml","CHGCAR","WAVECAR"]
        )
        return task, calc_dir
    
    else:
        raise ValueError("Invalid VASPAGENT_SUBMIT_TYPE. Must be 'local' or 'bohrium'.")




def vasp_scf_results(
    scf_work_dir_ls: Union[List[Path], Path],
) -> Dict[str, Any]:
    """
    Collect results from VASP SCF calculation.

    Args:
        scf_work_dir_ls (List[Path]): A list of path to the directories containing the VASP SCF calculation output files.
    Returns:
        A dictionary containing the path to output file of VASP calculation in extxyz format. The extxyz file contains the atomic structure and the total energy, atomic forces, etc., from the SCF calculation.
    """
    try:
        if isinstance(scf_work_dir_ls, Path):
            scf_work_dir_ls = [scf_work_dir_ls]
        atoms_ls=[]
        for scf_work_dir in scf_work_dir_ls:
            system=dpdata.LabeledSystem(str(scf_work_dir.absolute()/"OUTCAR"),fmt='vasp/outcar') 
            atoms=dpdata2ase_single(system)
            atoms_ls.append(atoms)
        work_path = Path(generate_work_path()).absolute()
        work_path.mkdir(parents=True, exist_ok=True)
        scf_result = work_path / "scf_result.extxyz"
        write(scf_result, atoms_ls, format="extxyz")
        return {
            "status": "success",
            "scf_result": str(scf_result.resolve())
            }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Collecting SCF results failed: {e}",
            "traceback": traceback.format_exc()}




def vasp_nscf(calculation_id: str, work_dir: str, struct: Structure, 
             kpoints: Kpoints, incar_dict: dict, chgcar_path: str, 
             wavecar_path: Optional[str] = None, 
             potcar_map: Optional[Dict] = None) -> Dict[str, Any]:
    """
    提交VASP非自洽场计算任务（能带计算）
    
    参数:
        calculation_id: 计算ID
        work_dir: 工作目录
        struct: 晶体结构
        kpoints: K点设置
        incar_dict: 额外的INCAR参数，会与默认设置合并。除非用户指定，不要擅自修改。
        chgcar_path: CHGCAR文件路径
        wavecar_path: WAVECAR文件路径
        potcar_map: POTCAR映射字典
        
    返回:
        Dict包含calculate_path、success、error等信息
    """
    Name = calculation_id
    calc_dir = os.path.abspath(f'{work_dir}/{Name}')
    if potcar_map is None:
        potcar_map = {}
    # 创建VASP输入文件
    # 手动获取元素列表，确保顺序与POSCAR一致
    poscar = Poscar(struct)
    unique_species = []
    for species in poscar.structure.species:
        species: Element
        if unique_species:
            if species.symbol != unique_species[-1]:
                if species.symbol not in potcar_map:
                    potcar_map[species.symbol] = species.symbol
                unique_species.append(species.symbol)
        else:
            if species.symbol not in potcar_map:
                potcar_map[species.symbol] = species.symbol
            unique_species.append(species.symbol)
    potcar_symbols = []
    for symbol in unique_species:
        potcar_symbols.append(potcar_map[symbol])

    vasp_input = VaspInput(
        poscar=poscar,
        incar=incar_dict,
        kpoints=kpoints,
        potcar=Potcar(potcar_symbols)
    )
    
    # 准备能带计算目录
    # band_dir = os.path.join(calc_dir, "band/")
    # os.makedirs(band_dir, exist_ok=True)
    vasp_input.write_input(calc_dir)
    
    # 复制相关文件
    if os.path.exists(chgcar_path):
        shutil.copy(chgcar_path, os.path.join(calc_dir, "CHGCAR"))
    if wavecar_path is not None and os.path.exists(wavecar_path):
        shutil.copy(wavecar_path, os.path.join(calc_dir, "WAVECAR"))


    submit_type = os.environ.get("VASPAGENT_SUBMIT_TYPE", "local").lower()
    if submit_type == "local":
        run_vasp(calc_dir) 

    elif submit_type == "bohrium":
        task=Task(
            command="source /opt/intel/oneapi/setvars.sh && mpirun -n 32 vasp_std",
            task_work_path=Name,
            forward_files=["POSCAR","INCAR","POTCAR","KPOINTS","CHGCAR","WAVECAR"],
            backward_files=["OSZICAR","CONTCAR","OUTCAR","vasprun.xml"]
        )
        return task, calc_dir
    
    else:
        raise ValueError("Invalid VASPAGENT_SUBMIT_TYPE. Must be 'local' or 'bohrium'.")




def read_calculation_result(calc_type: str, calculate_path: str) -> Dict[str, Any]:
    """
    根据计算类型读取计算结果
    """
    try:
        if calc_type == "relaxation":
            # 读取结构优化结果
            vasprun = Vasprun(os.path.join(calculate_path, "vasprun.xml"))
            contcar = Poscar.from_file(os.path.join(calculate_path, "CONTCAR"))
            
            return {
                "structure": contcar.structure,
                "total_energy": vasprun.final_energy,
                "max_force": np.max(np.linalg.norm(vasprun.ionic_steps[-1]['forces'], axis=1)),
                "stress": vasprun.ionic_steps[-1]['stress'],
                "ionic_steps": len(vasprun.ionic_steps),
            }
            
        elif calc_type == "scf":
            # 读取自洽场计算结果
            vasprun = Vasprun(os.path.join(calculate_path, "vasprun.xml"))
            
            return {
                "structure": vasprun.final_structure,
                "total_energy": vasprun.final_energy,
                "efermi": vasprun.efermi,
                "band_gap": vasprun.get_band_structure().get_band_gap(),
                "dos": vasprun.complete_dos,
                "eigen_values": vasprun.eigenvalues,
                "is_metal": vasprun.get_band_structure().is_metal(),
            }
            
        elif calc_type == "nscf":
            # 读取能带计算结果
            vasprun = Vasprun(os.path.join(calculate_path, "vasprun.xml"))
            bs = vasprun.get_band_structure()
            
            return {
                "structure": vasprun.final_structure,
                "band_structure": bs,
                "efermi": vasprun.efermi,
                "dos": vasprun.complete_dos,
                "eigen_values": vasprun.eigenvalues,
                "is_metal": bs.is_metal(),
                "band_gap": bs.get_band_gap(),
                "cbm": bs.get_cbm(),
                "vbm": bs.get_vbm(),
            }
        else:
            return {
                "success": False,
                "error": f"Unknown calculation type: {calc_type}"
            }
            
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }



