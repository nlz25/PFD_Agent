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
            backward_files=["OSZICAR","CONTCAR","OUTCAR","vasprun.xml"]
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