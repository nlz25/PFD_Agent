from typing import Optional, Union, Literal, Dict, Any, List, Tuple
import os
from pathlib import Path
from ase.io import read, write
from abacustest.lib_model.model_013_inputs import PrepInput
from abacustest.lib_model.comm import check_abacus_inputs as _check_abacus_inputs
from abacustest.lib_prepare.abacus import AbacusStru, ReadInput, WriteInput
from pfd_agent_tool.modules.util.common import generate_work_path
from pfd_agent_tool.init_mcp import mcp
from pfd_agent_tool.modules.util.ase2xyz import dpdata2ase_single
from pfd_agent_tool.modules.log.log import log_step
import dpdata
import numpy as np
from ._abacus import link_abacusjob, run_abacus, collect_metrics
#from abacusagent.modules.submodules.abacus import abacus_modify_input as _abacus_modify_input                                                 
import logging
import traceback


@mcp.tool()
@log_step(step_name="labeling_abacus_scf_preparation")
def abacus_prepare(
    structure_path: Path,
    job_type: Literal["scf", "relax", "cell-relax", "md"] = "scf",
    lcao: bool = True,
    nspin: Literal[1, 2, 4] = 1,
    soc: bool = False,
    dftu: bool = False,
    dftu_param: Optional[Union[Dict[str, Union[float, Tuple[Literal["p", "d", "f"], float]]], Literal['auto']]] = None,
    init_mag: Optional[Dict[str, float]] = None,
    afm: bool = False,
    extra_input: Optional[Dict[str, Any]] = None,
    frames: Optional[List[int]] = None  # select specific frame indices, default: all
) -> Dict[str, Any]:
    """
    Prepare mandatory input files for ABACUS calculation from a file that contain a list of structures.
    This function does not perform any actual calculation, but is necessary to use this function
    to prepare a list of directories containing necessary input files for ABACUS calculations in batch mode. 
    If there is error in preparing input files, try to use the abacus_modify_input or abacus_modify_stru tool to fix the INPUT file.
    
    Args:
        stru_file (Path): Structure file in extxyz format that contains multiple frame.
        job_type (Literal["scf", "relax", "cell-relax", "md"] = "scf"): The type of job to be performed, can be:
            'scf': Self-consistent field calculation, which is the default. 
            'relax': Geometry relaxation calculation, which wills relax the atomic position to the minimum energy configuration.
            'cell-relax': Cell relaxation calculation, which will relax the cell parameters and atomic positions to the minimum energy configuration.
            'md': Molecular dynamics calculation, which will perform molecular dynamics simulation.
        lcao (bool): Whether to use LCAO basis set, default is True.
        nspin (int): The number of spins, can be 1 (no spin), 2 (spin polarized), or 4 (non-collinear spin). Default is 1.
        soc (bool): Whether to use spin-orbit coupling, if True, nspin should be 4.
        dftu (bool): Whether to use DFT+U, default is False.
        dftu_param (dict): The DFT+U parameters, should be 'auto' or a dict
            If dft_param is set to 'auto', hubbard U parameters will be set to d-block and f-block elements automatically. For d-block elements, default U=4eV will
                be set to d orbital. For f-block elements, default U=6eV will be set to f orbital.
            If dft_param is a dict, the keys should be name of elements and the value has two choices:
                - A float number, which is the Hubbard U value of the element. The corrected orbital will be infered from the name of the element.
                - A list containing two elements: the corrected orbital (should be 'p', 'd' or 'f') and the Hubbard U value.
                For example, {"Fe": ["d", 4], "O": ["p", 1]} means applying DFT+U to Fe 3d orbital with U=4 eV and O 2p orbital with U=1 eV.
        init_mag ( dict or None): The initial magnetic moment for magnetic elements, should be a dict like {"Fe": 4, "Ti": 1}, where the key is the element symbol and the value is the initial magnetic moment.
        afm (bool): Whether to use antiferromagnetic calculation, default is False. If True, half of the magnetic elements will be set to negative initial magnetic moment.
        extra_input: Extra input parameters in the prepared INPUT file. Do not include any extra input otherwise being explicitly specified.
    
    Returns:
        A dictionary containing the list of job paths.
        - 'abacus_inputs_dir_list': A list of the absolute paths to the generated ABACUS input directory, containing INPUT, STRU, pseudopotential and orbital files.
        - 'input_content_list': A list of the content of the generated INPUT file.
    Raises:
        FileNotFoundError: If the structure file or pseudopotential path does not exist.
        ValueError: If LCAO basis set is selected but no orbital library path is provided.
        RuntimeError: If there is an error preparing input files.
    """
    try:
        structure_path = Path(structure_path).resolve()
        if not structure_path.is_file():
            raise FileNotFoundError(f"Structure file not found: {structure_path}")
        
        # Check if the pseudopotential path exists
        if soc:
            pp_path = os.environ.get("ABACUS_SOC_PP_PATH")
            orb_path = os.environ.get("ABACUS_SOC_ORB_PATH")
        else:
            pp_path = os.environ.get("ABACUS_PP_PATH")
            orb_path = os.environ.get("ABACUS_ORB_PATH")

        if not os.path.exists(pp_path):
            raise FileNotFoundError(f"Pseudopotential path {pp_path} does not exist.")

        if lcao and not os.path.exists(orb_path):
            raise FileNotFoundError(f"Orbital library path {orb_path} does not exist.")

        pwd = os.getcwd()
        work_path = Path(generate_work_path())
        os.chdir(work_path)

        results: List[Dict[str, Any]] = []
        cif_path_ls=[]
        cif_dir = Path("cif_dir")
        cif_dir.mkdir(exist_ok=True)
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
            #logging.INFO(f"Written CIF for frame {idx} to {cif_path}")                                                             
        try:
            extra_input_file = None
            if extra_input is not None:
                if 'out_chg' not in extra_input.keys():
                    extra_input['out_chg'] = -1
                # write extra input to the input file
                extra_input_file = Path("INPUT.tmp").resolve()
                WriteInput(extra_input, extra_input_file)
            
            # return a list of abacus input job dir
            _, abacus_inputs_dir_ls = PrepInput(
                files=cif_path_ls,
                filetype="cif",
                jobtype=job_type,
                pp_path=pp_path,
                orb_path=orb_path,
                input_file=extra_input_file,
                lcao=lcao,
                nspin=nspin,
                soc=soc,
                dftu=dftu,
                dftu_param=dftu_param,
                init_mag=init_mag,
                afm=afm,
                copy_pp_orb=True
            ).run() 
        except Exception as e:
            os.chdir(pwd)
            raise RuntimeError(f"Error preparing input files: {e}")

        if len(abacus_inputs_dir_ls) == 0:
            os.chdir(pwd)
            raise RuntimeError("No job path returned from PrepInput.")

        input_content_ls = [ReadInput(os.path.join(abacus_inputs_dir, "INPUT")) for abacus_inputs_dir in abacus_inputs_dir_ls]
        abacus_inputs_dir_ls = [str(Path(abacus_inputs_dir).resolve()) for abacus_inputs_dir in abacus_inputs_dir_ls]
        os.chdir(pwd)
        return {
            "status": "success",
            "abacus_inputs_dir_list": abacus_inputs_dir_ls,
            "input_content_list": input_content_ls}
    except Exception as e:
        return {
            "status": "error",
            "message": f"Batch prepare failed: {e}",
            "traceback": traceback.format_exc(),}


@mcp.tool()
def check_abacus_inputs(abacus_inputs_dir_ls: Union[List[Path], Path]) -> Dict[str, Any]:
    """
    Check if the ABACUS input files are valid. Always check the afer preparing input files with abacus_prepare_batch, 
    or after modifying them with abacus_modify_input_batch or abacus_modify_stru.
    Args:
        abacus_inputs_dir (str or Path): Path to the directory containing the ABACUS input files.

    Returns:
        Tuple[bool, str]: A tuple containing a boolean indicating validity and an error message if invalid.
    """
    try:
        if isinstance(abacus_inputs_dir_ls, Path):
            abacus_inputs_dir_ls = [abacus_inputs_dir_ls]
        for abacus_inputs_dir in abacus_inputs_dir_ls:
            is_valid, msg = _check_abacus_inputs(abacus_inputs_dir)
            if not is_valid:
                raise RuntimeError(f"Invalid ABACUS input files at {abacus_inputs_dir}: {msg}") 
        return {"check_passed": True, "message": "All ABACUS input files are valid."}
    except Exception as e:
        return {"check_passed": False, "message": str(e)}



@mcp.tool()
def abacus_modify_input(
    abacus_inputs_dir_list: Union[Path,List[Path]],
    dft_plus_u_settings: Optional[Dict[str, Union[float, Tuple[Literal["p", "d", "f"], float]]]] = None,
    extra_input: Optional[Dict[str, Any]] = None,
    remove_input: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Modify keywords in ABACUS INPUT file.
    Args:
        abacus_inputs_dir (str): Path to the directory containing the ABACUS input files.
        dft_plus_u_setting: Dictionary specifying DFT+U settings.  
            - Key: Element symbol (e.g., 'Fe', 'Ni').  
            - Value: A list with one or two elements:  
                - One-element form: float, representing the Hubbard U value (orbital will be inferred).  
                - Two-element form: [orbital, U], where `orbital` is one of {'p', 'd', 'f'}, and `U` is a float.
        extra_input: Additional key-value pairs to update the INPUT file. If the name of the key is already in the INPUT file, the value will be updated.
        remove_input: A list of parameter names to be removed in the INPUT file

    Returns:
        A dictionary containing:
        - modified_abacus_inputs_dir: the path of the modified INPUT file.
        - input_content: the content of the modified INPUT file as a dictionary.
    Raises:
        FileNotFoundError: If path of given INPUT file does not exist
        RuntimeError: If write modified INPUT file failed
    """
    try:
        if isinstance(abacus_inputs_dir_list, Path):
            abacus_inputs_dir_list = [abacus_inputs_dir_list]
        modified_abacus_inputs_dir = []
        modified_input_content_list = []
        for abacus_inputs_dir in abacus_inputs_dir_list:
            input_file = os.path.join(abacus_inputs_dir, "INPUT")
            if dft_plus_u_settings is not None:
                stru_file = os.path.join(abacus_inputs_dir, "STRU")
            if not os.path.isfile(input_file):
                raise FileNotFoundError(f"INPUT file {input_file} does not exist.")

            # Update simple keys and their values
            input_param = ReadInput(input_file)
            if extra_input is not None:
                for key, value in extra_input.items():
                    input_param[key] = value
    
            # Remove keys
            if remove_input is not None:
                for param in remove_input:
                    input_param.pop(param,None)

            # DFT+U settings
            main_group_elements = [
            "H", "He", 
            "Li", "Be", "B", "C", "N", "O", "F", "Ne",
        "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
        "K", "Ca", "Ga", "Ge", "As", "Se", "Br", "Kr",
        "Rb", "Sr", "In", "Sn", "Sb", "Te", "I", "Xe",
        "Cs", "Ba", "Tl", "Pb", "Bi", "Po", "At", "Rn",
        "Fr", "Ra", "Nh", "Fl", "Mc", "Lv", "Ts", "Og" ]
            transition_metals = [
        "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
        "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
        "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
        "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn"]
            lanthanides_and_acnitides = [
        "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
        "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr"]

            orbital_corr_map = {'p': 1, 'd': 2, 'f': 3}
            if dft_plus_u_settings is not None:
                input_param['dft_plus_u'] = 1

                stru = AbacusStru.ReadStru(stru_file)
                elements = stru.get_element(number=False,total=False)

                orbital_corr_param, hubbard_u_param = '', ''
                for element in elements:
                    if element not in dft_plus_u_settings:
                        orbital_corr_param += ' -1 '
                        hubbard_u_param += ' 0 '
                    else:
                        if type(dft_plus_u_settings[element]) is not float: # orbital_corr and hubbard_u are provided
                            orbital_corr = orbital_corr_map[dft_plus_u_settings[element][0]]
                            orbital_corr_param += f" {orbital_corr} "
                            hubbard_u_param += f" {dft_plus_u_settings[element][1]} "
                        else: #Only hubbard_u is provided, use default orbital_corr
                            if element in main_group_elements:
                                default_orb_corr = 1
                            elif element in transition_metals:
                                default_orb_corr = 2
                            elif element in lanthanides_and_acnitides:
                                default_orb_corr = 3

                            orbital_corr_param += f" {default_orb_corr} "
                            hubbard_u_param += f" {dft_plus_u_settings[element]} "

                input_param['orbital_corr'] = orbital_corr_param.strip()
                input_param['hubbard_u'] = hubbard_u_param.strip()

            WriteInput(input_param, input_file)
            modified_abacus_inputs_dir.append(Path(abacus_inputs_dir).resolve())
            modified_input_content_list.append(input_param)
        return {'modified_abacus_inputs_dir_list': modified_abacus_inputs_dir,
                'modified_input_content_list': modified_input_content_list}
    except Exception as e:
        return {'message': f"Modify ABACUS INPUT file failed: {e}"}

@mcp.tool()
def abacus_modify_stru(
    abacus_inputs_dir_ls: Union[Path, List[Path]],
    pp: Optional[Dict[str, str]] = None,
    orb: Optional[Dict[str, str]] = None,
    fix_atoms_idx: Optional[List[int]] = None,
    cell: Optional[List[List[float]]] = None,
    coord_change_type: Literal['scale', 'original'] = 'scale',
    movable_coords: Optional[List[bool]] = None,
    initial_magmoms: Optional[List[float]] = None,
    angle1: Optional[List[float]] = None,
    angle2: Optional[List[float]] = None
) -> Dict[str, Any]:
    """
    Modify pseudopotential, orbital, atom fixation, initial magnetic moments and initial velocities in ABACUS STRU file.
    Args:
        abacus_inputs_dir_ls (List[Path]): A list of paths to the directory containing the ABACUS input files.
        pp: Dictionary mapping element names to pseudopotential file paths.
            If not provided, the pseudopotentials from the original STRU file are retained.
        orb: Dictionary mapping element names to numerical orbital file paths.
            If not provided, the orbitals from the original STRU file are retained.
        fix_atoms_idx: List of indices of atoms to be fixed.
        cell: New cell parameters to be set in the STRU file. Should be a list of 3 lists, each containing 3 floats.
        coord_change_type: Type of coordinate change to apply.
            - 'scale': Scale the coordinates by the cell parameters. Suitable for most cases.
            - 'original': Use the original coordinates without scaling. Suitable for single atom or molecule in a large cell.
        movable_coords: For each fixed atom, specify which coordinates are allowed to move.
            Each entry is a list of 3 integers (0 or 1), where 1 means the corresponding coordinate (x/y/z) can move.
            Example: if `fix_atoms_idx = [1]` and `movable_coords = [[0, 1, 1]]`, the x-coordinate of atom 1 will be fixed.
        initial_magmoms: Initial magnetic moments for atoms.
            - For collinear calculations: a list of floats, shape (natom).
            - For non-collinear using Cartesian components: a list of 3-element lists, shape (natom, 3).
            - For non-collinear using angles: a list of floats, shape (natom), one magnetude of magnetic moment per atom.
        angle1: in non-colinear case, specify the angle between z-axis and real spin, in angle measure instead of radian measure
        angle2: in non-colinear case, specify angle between x-axis and real spin in projection in xy-plane , in angle measure instead of radian measure

    Returns:
        A dictionary containing:
        - modified_abacus_inputs_dir: the path of the modified ABACUS STRU file
        - stru_content: the content of the modified ABACUS STRU file as a string.
    Raises:
        ValueError: If `stru_file` is not path of a file, or dimension of initial_magmoms, angle1 or angle2 is not equal with number of atoms,
          or length of fixed_atoms_idx and movable_coords are not equal, or element in movable_coords are not a list with 3 bool elements
        KeyError: If pseudopotential or orbital are not provided for a element
    """
    try:
        if isinstance(abacus_inputs_dir_ls, Path):
            abacus_inputs_dir_ls = [abacus_inputs_dir_ls]
            
        for abacus_inputs_dir in abacus_inputs_dir_ls:
            stru_file = os.path.join(abacus_inputs_dir, "STRU")
            if os.path.isfile(stru_file):
                stru = AbacusStru.ReadStru(stru_file)
            else:
                raise ValueError(f"{stru_file} is not path of a file")

            # Set pp and orb
            elements = stru.get_element(number=False,total=False)
            if pp is not None:
                pplist = []
                for element in elements:
                    if element in pp:
                        pplist.append(pp[element])
                    else:
                        raise KeyError(f"Pseudopotential for element {element} is not provided")

                stru.set_pp(pplist)

            if orb is not None:
                orb_list = []
                for element in elements:
                    if element in orb:
                        orb_list.append(orb[element])
                    else:
                        raise KeyError(f"Orbital for element {element} is not provided")

                stru.set_orb(orb_list)

            # Set cell
            if cell is not None:
                if len(cell) != 3 or any(len(c) != 3 for c in cell):
                    raise ValueError("Cell should be a list of 3 lists, each containing 3 floats")

                if np.allclose(np.linalg.det(np.array(cell)), 0) is True:
                    raise ValueError("Cell cannot be a singular matrix, please provide a valid cell")
                if coord_change_type == "scale":
                    stru.set_cell(cell, bohr=False)
                elif coord_change_type == "original":
                    stru.set_cell(cell, bohr=False, change_coord=False)
                else:
                    raise ValueError("coord_change_type should be 'scale' or 'original'")

            # Set atomic magmom for every atom
            natoms = len(stru.get_coord())
            if initial_magmoms is not None:
                if len(initial_magmoms) == natoms:
                    stru.set_atommag(initial_magmoms)
                else:
                    raise ValueError("The dimension of given initial magmoms is not equal with number of atoms")
            if angle1 is not None and angle2 is not None:
                if len(initial_magmoms) == natoms:
                    stru.set_angle1(angle1)
                else:
                    raise ValueError("The dimension of given angle1 of initial magmoms is not equal with number of atoms")

                if len(initial_magmoms) == natoms:
                    stru.set_angle2(angle2)
                else:
                    raise ValueError("The dimension of given angle2 of initial magmoms is not equal with number of atoms")

            # Set atom fixations
            # Atom fixations in fix_atoms and movable_coors will be applied to original atom fixation
            if fix_atoms_idx is not None:
                atom_move = stru.get_move()
                for fix_idx, atom_idx in enumerate(fix_atoms_idx):
                    if fix_idx < 0 or fix_idx >= natoms:
                        raise ValueError("Given index of atoms to be fixed is not a integer >= 0 or < natoms")

                    if len(fix_atoms_idx) == len(movable_coords):
                        if len(movable_coords[fix_idx]) == 3:
                            atom_move[atom_idx] = movable_coords[fix_idx]
                        else:
                            raise ValueError("Elements of movable_coords should be a list with 3 bool elements")
                    else:
                        raise ValueError("Length of fix_atoms_idx and movable_coords should be equal")

                stru._move = atom_move

            stru.write(stru_file)
            stru_content = Path(stru_file).read_text(encoding='utf-8')
        
        return {'modified_abacus_inputs_dir': Path(abacus_inputs_dir).absolute(),
                'stru_content': stru_content}
    except Exception as e:
        return {'message': f"Modify ABACUS STRU file failed: {e}"}



@mcp.tool()
@log_step(step_name="labeling_abacus_scf_calculation")
def abacus_calculation_scf(
    abacus_inputs_dir_ls: Union[List[Path], Path],
) -> Dict[str, Any]:
    """
    Run ABACUS SCF calculation.

    Args:
        abacus_inputs_dir (Path): Path to the directory containing the ABACUS input files.
    Returns:
        A dictionary containing the path to output file of ABACUS calculation, and a dictionary containing whether the SCF calculation
        finished normally, the SCF is converged or not, the converged SCF energy and total time used.
    """
    try:
        if isinstance(abacus_inputs_dir_ls, Path):
            abacus_inputs_dir_ls = [abacus_inputs_dir_ls]
            
        # base work directory
        work_path_base = Path(generate_work_path()).absolute()
        work_path_ls=[]
        for abacus_inputs_dir in abacus_inputs_dir_ls:
            is_valid, msg = check_abacus_inputs(abacus_inputs_dir)
            if not is_valid:
                raise RuntimeError(f"Invalid ABACUS input files: {msg}")
            abacus_modify_input(abacus_inputs_dir,extra_input={'cal_force':True})
            work_path = work_path_base / abacus_inputs_dir.name
            work_path.mkdir(parents=True, exist_ok=True)
            link_abacusjob(src=abacus_inputs_dir, dst=work_path, copy_files=['INPUT', 'STRU'])
            input_params = ReadInput(os.path.join(work_path, "INPUT"))

            input_params['calculation'] = 'scf'
            WriteInput(input_params, os.path.join(work_path, "INPUT"))
            work_path_ls.append(work_path)
        
        print(work_path_ls)
        
        # running abacus calculation
        run_abacus(work_path_ls)

        return_dict = {
            'status': 'success',
            'scf_work_dir_list': [str(Path(work_path).expanduser().resolve()) for work_path in work_path_ls]}
        
        return_dict.update(
            {'scf_work_metrics_list':[collect_metrics(work_path,
                                                      metrics_names=['normal_end', 'converge', 'energy', 'total_time']) for work_path in work_path_ls]}
        )

        return return_dict
    except Exception as e:
        return {
            "status": "error",
            "message": f"Performing SCF calculation failed: {e}",
            "traceback": traceback.format_exc()}
                
    
@mcp.tool()
@log_step(step_name="labeling_abacus_scf_collect_results")
def collect_abacus_scf_results(
    scf_work_dir_ls: Union[List[Path], Path],
) -> Dict[str, Any]:
    """
    Collect results from ABACUS SCF calculation.

    Args:
        scf_work_dir_ls (List[Path]): A list of path to the directories containing the ABACUS SCF calculation output files.
    Returns:
        A dictionary containing the path to output file of ABACUS calculation in extxyz format. The extxyz file contains the atomic structure and the total energy, atomic forces, etc., from the SCF calculation.
    """
    try:
        if isinstance(scf_work_dir_ls, Path):
            scf_work_dir_ls = [scf_work_dir_ls]
        atoms_ls=[]
        for scf_work_dir in scf_work_dir_ls:
            system=dpdata.LabeledSystem(str(scf_work_dir.absolute()),fmt='abacus/scf') 
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