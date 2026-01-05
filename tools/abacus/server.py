import argparse
import os
import argparse
from typing import Optional, Union, Literal, Dict, Any, List, Tuple
from pathlib import Path
import importlib
import time
from dotenv import load_dotenv
from matcreator.tools.abacus import (
    abacus_prepare as _abacus_prepare,
    abacus_modify_stru as _abacus_modify_stru,
    abacus_modify_input as _abacus_modify_input,
    abacus_calculation_scf as _abacus_calculation_scf,
    collect_abacus_scf_results as _collect_abacus_scf_results,
    check_abacus_inputs as _check_abacus_inputs,
    )

_script_dir = Path(__file__).parent
load_dotenv(_script_dir / ".env", override=True)

ABACUS_SERVER_WORK_PATH = "/tmp/abacus_server"

# allowed modules of ABACUS agent
ALLOWED_MODULES=[
    "abacus","band","dos","scf","relax"
]

def load_tools():
    """
    Load all tools from the abacusagent package.
    """
    for py_file in ALLOWED_MODULES:
        module_name = f"abacusagent.modules.{py_file}"
        try:
            module = importlib.import_module(module_name)
            print(f"✅ Successfully loaded: {module_name}")
        except Exception as e:
            print(f"⚠️ Failed to load {module_name}: {str(e)}")


def create_workpath(work_path=None):
    """
    Create the working directory for AbacusAgent, and change the current working directory to it.
    
    Args:
        work_path (str, optional): The path to the working directory. If None, a default path will be used.
    
    Returns:
        str: The path to the working directory.
    """
    work_path = os.environ.get("ABACUS_SERVER_WORK_PATH", ABACUS_SERVER_WORK_PATH) + f"/{time.strftime('%Y%m%d%H%M%S')}"
    os.makedirs(work_path, exist_ok=True)
    os.chdir(work_path)
    print(f"Changed working directory to: {work_path}")
    return work_path    

def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="PFD_Agent Command Line Interface")
    
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

# set env variable to fit ABACUS AGENT    
os.environ["ABACUSAGENT_PORT"] = str(args.port)
os.environ["ABACUSAGENT_HOST"] = args.host
os.environ["ABACUSAGENT_MODEL"] = args.model

# compatibility with original ABACUS-agent project
from abacusagent.init_mcp import mcp

@mcp.tool()
def abacus_prepare_batch(
        structure_path: Path,
        job_type: Literal["scf", "relax", "cell-relax", "md"] = "scf",
        lcao: bool = True,
        nspin: Literal["1", "2", "4"] = "1",
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
    return _abacus_prepare(
            structure_path=structure_path,
            job_type=job_type,
            lcao=lcao,
            nspin=nspin,
            soc=soc,
            dftu=dftu,
            dftu_param=dftu_param,
            init_mag=init_mag,
            afm=afm,
            extra_input=extra_input,
            frames=frames
        )
    
@mcp.tool()
def check_abacus_inputs_batch(abacus_inputs_dir_ls: Union[List[Path], Path]) -> Dict[str, Any]:
    """
        Check if the ABACUS input files are valid. Always check the afer preparing input files with abacus_prepare_batch, 
        or after modifying them with abacus_modify_input_batch or abacus_modify_stru.
        Args:
            abacus_inputs_dir (str or Path): Path to the directory containing the ABACUS input files.

        Returns:
            Tuple[bool, str]: A tuple containing a boolean indicating validity and an error message if invalid.
        """
    return _check_abacus_inputs(abacus_inputs_dir_ls)
    
@mcp.tool()
def abacus_modify_input_batch(
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
    return _abacus_modify_input(
            abacus_inputs_dir_list=abacus_inputs_dir_list,
            dft_plus_u_settings=dft_plus_u_settings,
            extra_input=extra_input,
            remove_input=remove_input
        )
        
@mcp.tool()
def abacus_modify_stru_batch(
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
        Modify pseudopotential, orbital, atom fixation, initial magnetic moments and initial velocities in ABACUS STRU file in Batch Mode.
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
    return _abacus_modify_stru(
        abacus_inputs_dir_ls=abacus_inputs_dir_ls,
        pp=pp,
        orb=orb,
        fix_atoms_idx=fix_atoms_idx,
        cell=cell,
        coord_change_type=coord_change_type,
        movable_coords=movable_coords,
        initial_magmoms=initial_magmoms,
        angle1=angle1,
        angle2=angle2
    )
    
    
@mcp.tool()
def abacus_calculation_scf_batch(
        abacus_inputs_dir_ls: Union[List[str], str],
    ) -> Dict[str, Any]:
    """
        Run ABACUS SCF calculation in Batch Mode.

        Args:
            abacus_inputs_dir (Path): Path to the directory containing the ABACUS input files.
        Returns:
            A dictionary containing the path to output file of ABACUS calculation, and a dictionary containing whether the SCF calculation
            finished normally, the SCF is converged or not, the converged SCF energy and total time used.
        """
    return _abacus_calculation_scf(
            abacus_inputs_dir_ls=abacus_inputs_dir_ls
        )
        
@mcp.tool()
def collect_abacus_scf_results_batch(
        scf_work_dir_ls: Union[List[Path], Path],
    ) -> Dict[str, Any]:
    """
        Collect results from ABACUS SCF calculation in Batch Mode.

        Args:
            scf_work_dir_ls (List[Path]): A list of path to the directories containing the ABACUS SCF calculation output files.
        Returns:
            A dictionary containing the path to output file of ABACUS calculation in extxyz format. The extxyz file contains the atomic structure and the total energy, atomic forces, etc., from the SCF calculation.
        """
    return _collect_abacus_scf_results(
            scf_work_dir_ls=scf_work_dir_ls
        )

if __name__ == "__main__":
    create_workpath()
    load_tools(
    )
    mcp.run(transport=args.transport)