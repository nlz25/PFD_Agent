import traceback
import time
import uuid
import os
from ase import Atoms
import dpdata
import numpy as np

def generate_work_path(create: bool = True) -> str:
    """
    Generate a unique working directory path based on call function and current time.
    
    directory = calling function name + current time + random string.
    
    Returns:
        str: The path to the working directory.
    """
    calling_function = traceback.extract_stack(limit=2)[-2].name
    current_time = time.strftime("%Y%m%d%H%M%S")
    random_string = str(uuid.uuid4())[:8]
    work_path = f"{current_time}.{calling_function}.{random_string}"
    if create:
        os.makedirs(work_path, exist_ok=True)
    
    return work_path

def dpdata2ase_single(
    sys: dpdata.System
    )->Atoms:
    """Convert dpdata System to ase.Atoms."""
    #atoms_list = []
    #for ii in range(len(sys)):
    atoms=Atoms(
        symbols=[sys.get_atom_names()[i] for i in sys.get_atom_types()],
        positions=sys.data["coords"][0].tolist(),
        cell=sys.data["cells"][0].tolist(),
        pbc= not sys.nopbc
        )
        # set the virials and forces
    if "virial" in sys.data:
        atoms.set_array("virial", sys.data["virial"][0])
    if "forces" in sys.data:
        atoms.set_array("forces", sys.data["forces"][0])
    if "energies" in sys.data:
        atoms.info["energy"] = sys.data["energies"][0]
    return atoms