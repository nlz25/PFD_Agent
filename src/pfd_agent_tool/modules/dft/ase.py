from typing import Any, Dict, List, Union, Optional
from pathlib import Path
from ase import Atoms
from ase.io import write,read
from pfd_agent_tool.init_mcp import mcp
from pfd_agent_tool.modules.expl.calculator import CalculatorWrapper
from pfd_agent_tool.modules.util.common import generate_work_path
from pfd_agent_tool.modules.log.log import log_step
import traceback
import logging

@mcp.tool()
@log_step(step_name="labeling_ase_calculation")
def ase_calculation(
    structure_path: Union[List[Path], Path],
    model_style: str = "dpa",
    model_path: Optional[Path] = None,
    calc_args: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Perform energy and force (and stress) calculation on given structures.

    Parameters
    - structure_path: List[Path] | Path
        Path(s) to structure file(s) (extxyz/xyz/vasp/...). Can be a multi-frame file or a list of files.
    - model_style: str
        ASE calculator key (e.g., "dpa").
    - model_path: Path
        Model file(s) or URL(s) for ML calculators. 
    - calc_args (Dict[str, str], optional): Optional calculator initialization parameters passed directly to the calculator wrapper. For pretrained DPA multi-head models, an available head should be provided. 
        The head is defaulted to "MP_traj_v024_alldata_mixu" if not specified. Example: {"head": "MP_traj_v024_alldata_mixu"}

    Returns
    - Dict[str, Any]
        Dictionary containing paths to labeled data file and logs.
    """
    try:
        work_path=Path(generate_work_path())
        work_path = work_path.expanduser().resolve()
        work_path.mkdir(parents=True, exist_ok=True)
        
        if not calc_args:
            calc_args = {}
        calc=CalculatorWrapper.get_calculator(model_style)
        calc=calc().create(model_path=model_path, **calc_args)
        
        # read structures
        atoms_ls=[]
        if isinstance(structure_path, Path):
            structure_path = [structure_path]
        for path in structure_path:
            read_atoms = read(path, index=":")
            if isinstance(read_atoms, Atoms):
                atoms_ls.append(read_atoms)
            else:
                atoms_ls.extend(read_atoms)
        
        for atoms in atoms_ls:
            atoms.calc = calc
            energy= atoms.get_potential_energy()
            forces=atoms.get_forces()
            stress = atoms.get_stress()
            atoms.calc.results.clear()
            atoms.info['energy'] = energy
            atoms.set_array('force', forces)
            atoms.info['stress'] = stress
        labeled_data = work_path / "ase_results.extxyz"
        write(labeled_data, atoms_ls, format="extxyz")
        
        result = {
            "status": "success",
            "labeled_data": str(labeled_data.resolve()),
            "message": f"ASE calculation completed for {len(atoms_ls)} structures."
        }
    
    except Exception as e:
        logging.error(f"Error in ase_calculation: {str(e)}")
        result={
            "status": "error",
            "message": f"ASE calculation failed: {e}",
            "traceback": traceback.format_exc()
            }
    return result