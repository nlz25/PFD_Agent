"""Utilities to create crystal structures with ASE."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
from ase.atoms import Atoms
from ase.build import bulk, make_supercell
from ase.io import write

from matcreator.tools.util.common import generate_work_path

logger = logging.getLogger(__name__)

## ==============================
## bulk crystal building functions
## ==============================

SupercellType = Union[int, Sequence[int], Sequence[Sequence[int]]]


def _sanitize_token(token: str) -> str:
	"""Return a filesystem-friendly token derived from a user string."""

	cleaned = re.sub(r"[^A-Za-z0-9_]+", "-", token.strip())
	return re.sub(r"-+", "-", cleaned).strip("-") or "structure"


def _apply_supercell(atoms: Atoms, size: SupercellType) -> Atoms:
	"""Expand *atoms* according to *size* definition."""

	if size in (None, 1):
		return atoms
	if isinstance(size, int):
		return atoms.repeat((size, size, size))
	if isinstance(size, Sequence):
		size_list = list(size)
		if len(size_list) == 3 and all(isinstance(val, int) for val in size_list):
			return atoms.repeat(tuple(size_list))
		if len(size_list) == 3 and all(
			isinstance(row, Sequence) and len(row) == 3 for row in size_list
		):
			matrix = np.array(size_list, dtype=int)
			return make_supercell(atoms, matrix)
	raise ValueError(
		"size must be an int, length-3 sequence of ints, or a 3x3 integer matrix"
	)


def _resolve_output_path(
	output_path: Optional[Union[str, Path]],
	formula: str,
	crystal_structure: str,
	file_extension: str,
) -> Path:
	"""Compute destination path for the generated structure."""

	if output_path:
		destination = Path(output_path).expanduser()
		destination.parent.mkdir(parents=True, exist_ok=True)
		return destination

	work_dir = Path(generate_work_path())
	work_dir.mkdir(parents=True, exist_ok=True)
	formula_token = _sanitize_token(formula)
	struct_token = _sanitize_token(crystal_structure)
	filename = f"{formula_token}-{struct_token}.{file_extension}"
	return (work_dir / filename).resolve()


def build_bulk_crystal(
	formula: str,
	crystal_structure: str,
	a: Optional[float] = None,
	c: Optional[float] = None,
	covera: Optional[float] = None,
	u: Optional[float] = None,
	spacegroup: Optional[int] = None,
	basis: Optional[Sequence[Sequence[float]]] = None,
	orthorhombic: bool = False,
	cubic: bool = False,
	size: SupercellType = 1,
	vacuum: Optional[float] = None,
	output_format: str = "extxyz",
	output_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
	"""Create a bulk crystal with ASE and persist it to disk."""

	builder_kwargs: Dict[str, Any] = {
		"name": formula,
		"crystalstructure": crystal_structure,
	}
	if a is not None:
		builder_kwargs["a"] = a
	if c is not None:
		builder_kwargs["c"] = c
	if covera is not None:
		builder_kwargs["covera"] = covera
	if u is not None:
		builder_kwargs["u"] = u
	if spacegroup is not None:
		builder_kwargs["spacegroup"] = spacegroup
	if basis is not None:
		builder_kwargs["basis"] = basis
	if orthorhombic:
		builder_kwargs["orthorhombic"] = True
	if cubic:
		builder_kwargs["cubic"] = True

	fmt = output_format.lower()
	extension_map = {
		"extxyz": "extxyz",
		"xyz": "xyz",
		"cif": "cif",
		"vasp": "vasp",
		"poscar": "vasp",
		"json": "json",
	}
	file_extension = extension_map.get(fmt, fmt)

	try:
		atoms = bulk(**builder_kwargs)
		atoms = _apply_supercell(atoms, size)
		if vacuum is not None:
			atoms.center(vacuum=vacuum)

		destination = _resolve_output_path(
			output_path=output_path,
			formula=formula,
			crystal_structure=crystal_structure,
			file_extension=file_extension,
		)
		write(destination, atoms, format=fmt)

		result = {
			"status": "success",
			"message": "Bulk crystal generated successfully.",
			"structure_path": str(destination),
			"chemical_formula": atoms.get_chemical_formula(empirical=True),
			"num_atoms": len(atoms),
			"cell": atoms.cell.tolist(),
			"pbc": atoms.get_pbc().tolist(),
		}
	except Exception as exc:
		logger.error("Failed to build crystal: %s", exc)
		result = {
			"status": "error",
			"message": f"Failed to build crystal: {exc}",
			"structure_path": "",
			"chemical_formula": "",
			"num_atoms": 0,
			"cell": [],
			"pbc": [],
		}

	return result

## ================================
## structure perturbation functions
## ================================
def _get_cell_perturb_matrix(cell_pert_fraction: float):
    """[Modified from dpdata]

    Args:
        cell_pert_fraction (float): The fraction of cell perturbation.

    Raises:
        RuntimeError: If cell_pert_fraction is negative.

    Returns:
        np.ndarray: A 3x3 cell perturbation matrix.
    """
    if cell_pert_fraction < 0:
        raise RuntimeError("cell_pert_fraction can not be negative")
    e0 = np.random.rand(6)
    e = e0 * 2 * cell_pert_fraction - cell_pert_fraction
    cell_pert_matrix = np.array(
        [
            [1 + e[0], 0.5 * e[5], 0.5 * e[4]],
            [0.5 * e[5], 1 + e[1], 0.5 * e[3]],
            [0.5 * e[4], 0.5 * e[3], 1 + e[2]],
        ]
    )
    return cell_pert_matrix


def _get_atom_perturb_vector(
    atom_pert_distance: float,
    atom_pert_style: str = "normal",
):
    """[Modified from dpdata] Perturb atom coord vectors.

    Args:
        atom_pert_distance (float): The distance to perturb the atom.
        atom_pert_style (str, optional): The style of perturbation. Defaults to "normal".

    Raises:
        RuntimeError: If atom_pert_distance is negative.
        RuntimeError: If atom_pert_style is not supported.

    Returns:
        np.ndarray: The perturbation vector for the atom.
    """
    random_vector = None
    if atom_pert_distance < 0:
        raise RuntimeError("atom_pert_distance can not be negative")

    if atom_pert_style == "normal":
        # return 3 numbers independently sampled from normal distribution
        e = np.random.randn(3)
        random_vector = (atom_pert_distance / np.sqrt(3)) * e
    elif atom_pert_style == "uniform":
        e = np.random.randn(3)
        while np.linalg.norm(e) < 0.1:
            e = np.random.randn(3)
        random_unit_vector = e / np.linalg.norm(e)
        v0 = np.random.rand(1)
        v = np.power(v0, 1 / 3)
        random_vector = atom_pert_distance * v * random_unit_vector
    elif atom_pert_style == "const":
        e = np.random.randn(3)
        while np.linalg.norm(e) < 0.1:
            e = np.random.randn(3)
        random_unit_vector = e / np.linalg.norm(e)
        random_vector = atom_pert_distance * random_unit_vector
    else:
        raise RuntimeError(f"unsupported options atom_pert_style={atom_pert_style}")
    return random_vector


def _perturb_atoms(
	atoms: Atoms,
	pert_num: int,
	cell_pert_fraction: float,
	atom_pert_distance: float,
	atom_pert_style: str = "normal",
	atom_pert_prob: float = 1.0,
):
	"""[Modified from dpdata] Generate perturbed structures for a single Atoms.

	Args:
		atoms: Input structure to perturb.
		pert_num: Number of perturbed structures to generate.
		cell_pert_fraction: Fractional cell distortion magnitude.
		atom_pert_distance: Max atomic displacement magnitude (Å).
		atom_pert_style: Displacement style ("normal", "uniform", or "const").
		atom_pert_prob: Probability each atom is selected for perturbation.

	Returns:
		List[Atoms]: List of perturbed structures.
	"""

	pert_atoms_ls = []
	for _ in range(pert_num):
		cell_perturb_matrix = _get_cell_perturb_matrix(cell_pert_fraction)
		pert_cell = np.matmul(atoms.get_cell().array, cell_perturb_matrix)
		pert_positions = atoms.get_positions().copy()
		pert_natoms = int(atom_pert_prob * len(atoms))
		pert_atom_id = sorted(
			np.random.choice(
				range(len(atoms)),
				pert_natoms,
				replace=False,
			).tolist()
		)

		for kk in pert_atom_id:
			atom_perturb_vector = _get_atom_perturb_vector(
				atom_pert_distance, atom_pert_style
			)
			pert_positions[kk] += atom_perturb_vector

		pert_atoms = Atoms(
			symbols=atoms.get_chemical_symbols(),
			positions=pert_positions,
			cell=pert_cell,
			pbc=atoms.get_pbc(),
		)
		pert_atoms_ls.append(pert_atoms)
	return pert_atoms_ls


def perturb_atoms(
	structure_path: Union[str, Path],
	pert_num: int,
	cell_pert_fraction: float,
	atom_pert_distance: float,
	atom_pert_style: str = "normal",
	atom_pert_prob: float = 1.0,
	output_format: str = "extxyz",
	output_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
	"""Public wrapper for `_perturb_atoms` that reads a structure file.

	The input is a single structure file readable by ASE (e.g. extxyz, xyz,
	cif, vasp). The function loads the first frame, generates ``pert_num``
	perturbed replicas, writes them as a multi-frame output file, and returns
	an MCP-compatible result dictionary similar to :func:`build_bulk_crystal`.

	Returns a dictionary with keys:
	- status: "success" or "error".
	- message: Short description of the outcome.
	- structure_path: Path to the written multi-frame file (or empty string).
	- num_structures: Number of perturbations written.
	- num_atoms_per_structure: List[int] with atom counts for each frame.
	"""

	fmt = output_format.lower()
	extension_map = {
		"extxyz": "extxyz",
		"xyz": "xyz",
		"cif": "cif",
		"vasp": "vasp",
		"poscar": "vasp",
		"json": "json",
	}
	file_extension = extension_map.get(fmt, fmt)

	try:
		from ase.io import read

		atoms = read(str(structure_path))
		perturbed = _perturb_atoms(
			atoms=atoms,
			pert_num=pert_num,
			cell_pert_fraction=cell_pert_fraction,
			atom_pert_distance=atom_pert_distance,
			atom_pert_style=atom_pert_style,
			atom_pert_prob=atom_pert_prob,
		)
		if not perturbed:
			raise RuntimeError("No perturbed structures were generated")

		if output_path:
			destination = Path(output_path).expanduser()
			destination.parent.mkdir(parents=True, exist_ok=True)
		else:
			work_dir = Path(generate_work_path())
			work_dir.mkdir(parents=True, exist_ok=True)
			formula = atoms.get_chemical_formula(empirical=True)
			filename = f"{_sanitize_token(formula)}-perturbed.{file_extension}"
			destination = (work_dir / filename).resolve()

		write(destination, perturbed, format=fmt)

		result = {
			"status": "success",
			"message": "Perturbed structures generated successfully.",
			"structure_path": str(destination),
			"num_structures": len(perturbed),
			"num_atoms_per_structure": [len(a) for a in perturbed],
		}
	except Exception as exc:  # pragma: no cover - defensive
		logger.error("Failed to perturb atoms: %s", exc)
		result = {
			"status": "error",
			"message": f"Failed to perturb atoms: {exc}",
			"structure_path": "",
			"num_structures": 0,
			"num_atoms_per_structure": [],
		}

	return result


def _build_supercell(
	atoms: Atoms,
	size: SupercellType,
) -> Atoms:
	"""Construct a supercell from an `Atoms` object using the same `size` logic.

	This is a small convenience around `_apply_supercell` so the same
	implementation is reused in other tools or MCP wrappers.
	"""

	return _apply_supercell(atoms, size)


def build_supercell(
	input_structure: Union[str, Path],
	size: SupercellType,
	output_format: str = "extxyz",
	output_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
	"""Build a supercell from a structure file and write it to disk.

	This helper reads a structure (any ASE-supported format), expands it
	according to `size`, and writes the resulting supercell to a file. The
	return dictionary is similar in spirit to `build_bulk_crystal` for easy
	integration as an MCP tool.

	- size: Same semantics as in `build_bulk_crystal` and `_apply_supercell`
	  (int, (nx, ny, nz) tuple/list, or 3x3 integer matrix).
	"""

	fmt = output_format.lower()
	extension_map = {
		"extxyz": "extxyz",
		"xyz": "xyz",
		"cif": "cif",
		"vasp": "vasp",
		"poscar": "vasp",
		"json": "json",
	}
	file_extension = extension_map.get(fmt, fmt)

	try:
		from ase.io import read

		atoms = read(str(input_structure))
		supercell = _build_supercell(atoms, size=size)

		if output_path:
			destination = Path(output_path).expanduser()
			destination.parent.mkdir(parents=True, exist_ok=True)
		else:
			work_dir = Path(generate_work_path())
			work_dir.mkdir(parents=True, exist_ok=True)
			formula = supercell.get_chemical_formula(empirical=True)
			filename = f"{_sanitize_token(formula)}-supercell.{file_extension}"
			destination = (work_dir / filename).resolve()

		write(destination, supercell, format=fmt)

		result = {
			"status": "success",
			"message": "Supercell generated successfully.",
			"structure_path": str(destination),
			"chemical_formula": supercell.get_chemical_formula(empirical=True),
			"num_atoms": len(supercell),
			"cell": supercell.cell.tolist(),
			"pbc": supercell.get_pbc().tolist(),
		}
	except Exception as exc:  # pragma: no cover - defensive
		logger.error("Failed to build supercell from file: %s", exc)
		result = {
			"status": "error",
			"message": f"Failed to build supercell: {exc}",
			"structure_path": "",
			"chemical_formula": "",
			"num_atoms": 0,
			"cell": [],
			"pbc": [],
		}

	return result


def inspect_structure(
	structure_path: Union[str, Path]
) -> Dict[str, Any]:
	"""Inspect an ASE-readable structure file and summarize its metadata."""

	try:
		from ase.io import read

		source = Path(structure_path).expanduser()
		if not source.exists():
			raise FileNotFoundError(f"Structure file not found: {source}")

		frames = read(str(source), index=":")
		if isinstance(frames, Atoms):
			frames = [frames]
		else:
			frames = list(frames)
		if not frames:
			raise RuntimeError("No frames could be read from the structure file")

		formulas = [frame.get_chemical_formula(empirical=True) for frame in frames]
		num_atoms_per_frame = [len(frame) for frame in frames]
		#cells = [frame.cell.tolist() for frame in frames]
		#pbc_flags = [frame.get_pbc().tolist() for frame in frames]
		info_keys = sorted({key for frame in frames for key in frame.info.keys()})
		array_keys = sorted({key for frame in frames for key in frame.arrays.keys()})

		unique_formulas = list(sorted(set(formulas)))
		unique_num_atoms = list(sorted(set(num_atoms_per_frame)))
		result = {
			"status": "success",
			"message": f"Read {len(frames)} frame(s) from structure file.",
			"structure_path": str(source.resolve()),
			"num_frames": len(frames),
			#"chemical_formula": formulas[0],
			"chemical_formulas": unique_formulas,
			#"num_atoms": num_atoms_per_frame[0],
			"num_atoms": unique_num_atoms,
			#"cells": cells,
			#"pbc": pbc_flags,
			"info_keys": info_keys,
			"array_keys": array_keys,
		}
	except Exception as exc:  # pragma: no cover - defensive
		logger.error("Failed to inspect structure: %s", exc)
		result = {
			"status": "error",
			"message": f"Failed to inspect structure: {exc}",
			"structure_path": "",
			"num_frames": 0,
			#"chemical_formula": "",
			"chemical_formulas": [],
			#"num_atoms": 0,
			"num_atoms": [],
			#"cells": [],
			#"pbc": [],
			"info_keys": [],
			"array_keys": [],
		}

	return result

