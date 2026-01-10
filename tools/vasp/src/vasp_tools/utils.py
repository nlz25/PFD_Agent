import time
import traceback
import uuid
import os
import dpdata
import numpy as np     
from ase import Atoms

def generate_work_path(create: bool = True) -> str:
	"""Return a unique work dir path and create it by default."""
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