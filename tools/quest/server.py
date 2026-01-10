from __future__ import annotations

import argparse
import os
import traceback
import uuid
import time
import logging
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union
import numpy as np
from ase.atoms import Atoms
from ase.build import bulk, make_supercell
from ase.io import write
from dotenv import load_dotenv

"""Utilities to create crystal structures with ASE."""
def generate_work_path(create: bool = True) -> str:
	"""Return a unique work dir path and create it by default."""
	calling_function = traceback.extract_stack(limit=2)[-2].name
	current_time = time.strftime("%Y%m%d%H%M%S")
	random_string = str(uuid.uuid4())[:8]
	work_path = f"{current_time}.{calling_function}.{random_string}"
	if create:
		os.makedirs(work_path, exist_ok=True)
	return work_path

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


def build_bulk_crystal_impl(
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


def _perturb_atoms_impl(
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


def perturb_atoms_impl(
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
		perturbed = _perturb_atoms_impl(
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


def _build_supercell_impl(
	atoms: Atoms,
	size: SupercellType,
) -> Atoms:
	"""Construct a supercell from an `Atoms` object using the same `size` logic.

	This is a small convenience around `_apply_supercell` so the same
	implementation is reused in other tools or MCP wrappers.
	"""

	return _apply_supercell(atoms, size)


def build_supercell_impl(
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
		supercell = _build_supercell_impl(atoms, size=size)

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


def inspect_structure_impl(
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

## ================================
## entropy-based filtering functions
## ===============================


def _h_filter_cpu(
	iter_confs: List[Atoms],
	dset_confs: List[Atoms]=[],
	chunk_size: int = 10,
	max_sel: int = 100,
	k: int = 32,
	cutoff: float = 5.0,
	batch_size: int = 1000,
	h: float = 0.015,
	dtype: str = "float32",
):
	"""Entropy-based selection on CPU via quests."""
	from quests.descriptor import get_descriptors
	from quests.entropy import entropy, delta_entropy

	num_ref = len(dset_confs)
	if len(dset_confs) == 0:
		if chunk_size >= len(iter_confs):
			return iter_confs, {"num_confs": len(iter_confs)}
		random.shuffle(iter_confs)
		dset_confs = iter_confs[:chunk_size]
		iter_confs = iter_confs[chunk_size:]
		num_ref = 0
		max_sel -= chunk_size

	max_iter = min(
		max_sel // chunk_size + (max_sel % chunk_size > 0),
		len(iter_confs) // chunk_size + (len(iter_confs) % chunk_size > 0),
	)
	iter_desc = get_descriptors(iter_confs, k=k, cutoff=cutoff, dtype=dtype)
	dset_desc = get_descriptors(dset_confs, k=k, cutoff=cutoff, dtype=dtype)

	num_atoms_per_structure_iter = [atoms.get_number_of_atoms() for atoms in iter_confs]
	atom_indices_iter = []
	start = 0
	for n in num_atoms_per_structure_iter:
		end = start + n
		atom_indices_iter.append((start, end))
		start = end

	H_list = []
	H = entropy(dset_desc, h=h, batch_size=batch_size)
	logging.info(
		"Initial entropy with %s reference configurations: %.4f",
		len(dset_confs),
		H,
	)
	H_list.append(H)
	result = {"iter_00": H, "num_confs": len(dset_confs)}
	indices = []
	for ii in range(max_iter):
		re_indices = [i for i in range(len(iter_confs)) if i not in indices]
		re_confs = [iter_confs[i] for i in re_indices]
		re_desc = [
			iter_desc[atom_indices_iter[i][0] : atom_indices_iter[i][1]]
			for i in re_indices
		]
		x = np.vstack(re_desc)
		delta = delta_entropy(x, dset_desc, h=h, batch_size=batch_size)
		num_atoms_per_structure = [atoms.get_number_of_atoms() for atoms in re_confs]
		atom_indices = []
		start = 0
		for n in num_atoms_per_structure:
			end = start + n
			atom_indices.append((start, end))
			start = end
		delta_sums = [delta[start:end].sum() for start, end in atom_indices]
		sorted_pairs = sorted(zip(re_indices, delta_sums), key=lambda x: x[1], reverse=True)
		sorted_re_indices = [idx for idx, _ in sorted_pairs]
		selected_indices = sorted_re_indices[:chunk_size]
		dset_desc_ls = [dset_desc]
		for idx in selected_indices:
			indices.append(idx)
			dset_confs.append(iter_confs[idx])
			dset_desc_ls.append(iter_desc[atom_indices_iter[idx][0] : atom_indices_iter[idx][1]])
		dset_desc = np.vstack(dset_desc_ls)
		H = entropy(dset_desc, h=h, batch_size=batch_size)
		dH = H - H_list[-1]
		H_list.append(H)
		logging.info(
			"Iteration %s/%s, selected %s configurations, entropy %.4f",
			ii + 1,
			max_iter,
			len(dset_confs),
			H,
		)
		result.update({f"iter_{ii+1:02d}": H, "num_confs": len(dset_confs)})
		if dH < 1e-2:
			logging.info("Entropy increase %.4f < 1e-2, stopping selection.", dH)
			break
	return dset_confs[num_ref:], result


def _h_filter_gpu(
	iter_confs: List[Atoms],
	dset_confs: List[Atoms]=[],
	chunk_size: int = 10,
	max_sel: int = 100,
	k: int = 32,
	cutoff: float = 5.0,
	batch_size: int = 1000,
	h: float = 0.015,
	dtype: str = "float32",
):
	import torch
	from quests.descriptor import get_descriptors
	from quests.gpu.entropy import delta_entropy, entropy

	device = "cuda" if torch.cuda.is_available() else "cpu"
	result = {}
	num_ref = len(dset_confs)
	if len(dset_confs) == 0:
		if chunk_size >= len(iter_confs):
			return iter_confs, {"num_confs": len(iter_confs)}
		random.shuffle(iter_confs)
		dset_confs = iter_confs[:chunk_size]
		iter_confs = iter_confs[chunk_size:]
		num_ref = 0
		max_sel -= chunk_size

	max_iter = min(
		max_sel // chunk_size + (max_sel % chunk_size > 0),
		len(iter_confs) // chunk_size + (len(iter_confs) % chunk_size > 0),
	)
	iter_desc = get_descriptors(iter_confs, k=k, cutoff=cutoff, dtype=dtype)
	dset_desc = get_descriptors(dset_confs, k=k, cutoff=cutoff, dtype=dtype)

	num_atoms_per_structure_iter = [atoms.get_number_of_atoms() for atoms in iter_confs]
	atom_indices_iter = []
	start = 0
	for n in num_atoms_per_structure_iter:
		end = start + n
		atom_indices_iter.append((start, end))
		start = end

	H_list = []
	x = torch.tensor(dset_desc, device=device, dtype=torch.float32)
	H = entropy(x, h=h, batch_size=batch_size, device=device)
	H_list.append(float(H.cpu().numpy()))
	result.update({"num_confs": len(dset_confs), "iter_00": float(H.cpu().numpy())})
	indices = []
	for ii in range(max_iter):
		re_indices = [i for i in range(len(iter_confs)) if i not in indices]
		re_confs = [iter_confs[i] for i in re_indices]
		re_desc = [
			iter_desc[atom_indices_iter[i][0] : atom_indices_iter[i][1]]
			for i in re_indices
		]
		x = torch.tensor(np.vstack(re_desc), device=device, dtype=torch.float32)
		y = torch.tensor(dset_desc, device=device, dtype=torch.float32)
		delta = delta_entropy(x, y, h=h, batch_size=batch_size, device=device)
		delta = delta.cpu().numpy()
		num_atoms_per_structure = [atoms.get_number_of_atoms() for atoms in re_confs]
		atom_indices = []
		start = 0
		for n in num_atoms_per_structure:
			end = start + n
			atom_indices.append((start, end))
			start = end
		delta_sums = [delta[start:end].sum() for start, end in atom_indices]
		sorted_pairs = sorted(zip(re_indices, delta_sums), key=lambda x: x[1], reverse=True)
		sorted_re_indices = [idx for idx, _ in sorted_pairs]
		selected_indices = sorted_re_indices[:chunk_size]
		dset_desc_ls = [dset_desc]
		for idx in selected_indices:
			indices.append(idx)
			dset_confs.append(iter_confs[idx])
			dset_desc_ls.append(iter_desc[atom_indices_iter[idx][0] : atom_indices_iter[idx][1]])
		dset_desc = np.vstack(dset_desc_ls)
		y = torch.tensor(dset_desc, device=device, dtype=torch.float32)
		H = entropy(y, h=h, batch_size=batch_size, device=device)
		dH = H - H_list[-1]
		H_list.append(float(H.cpu().numpy()))
		result.update({f"iter_{ii+1:02d}": float(H.cpu().numpy()), "num_confs": len(dset_confs)})
		if dH < 1e-2:
			break
	return dset_confs[num_ref:], result


def filter_by_entropy_impl(
	iter_confs: Union[List[Union[Path, str]], Union[Path, str]],
	reference: Union[List[Union[Path, str]], Union[Path, str]] = [],
	chunk_size: int = 10,
	k: int = 32,
	cutoff: float = 5.0,
	batch_size: int = 1000,
	h: float = 0.015,
	max_sel: int = 50,
):
	"""Entropy-based subset selection; tries GPU first, falls back to CPU."""
	try:
		from ase.io import read

		if isinstance(iter_confs, list):
			iter_confs = [read(p, index=":") for p in iter_confs]
			iter_confs = [atom for sublist in iter_confs for atom in sublist]
		else:
			iter_confs = read(iter_confs, index=":")

		if isinstance(reference, (Path, str)):
			reference = read(reference, index=":")
		elif isinstance(reference, list):
			reference = [read(p, index=":") for p in reference]
			reference = [atom for sublist in reference for atom in sublist]
		try:
			import torch  # noqa: F401
			select_atoms, select_result = _h_filter_gpu(
				iter_confs,
				reference,
				chunk_size=chunk_size,
				max_sel=max_sel,
				k=k,
				cutoff=cutoff,
				batch_size=batch_size,
				h=h,
			)
		except ImportError:
			select_atoms, select_result = _h_filter_cpu(
				iter_confs,
				reference,
				chunk_size=chunk_size,
				max_sel=max_sel,
				k=k,
				cutoff=cutoff,
				batch_size=batch_size,
				h=h,
			)

		work_path = Path(generate_work_path())
		work_path = work_path.expanduser().resolve()
		work_path.mkdir(parents=True, exist_ok=True)
		select_atoms_path = work_path / "selected.extxyz"
		write(select_atoms_path, select_atoms)

		result = {
			"status": "success",
			"message": "Filter by entropy completed.",
			"selected_atoms": str(select_atoms_path.resolve()),
			"entropy": select_result,
		}

	except Exception as e:
		logging.error(
			"Error in filter_by_entropy: %s. Traceback: %s",
			str(e),
			traceback.format_exc(),
		)
		result = {
			"status": "error",
			"message": f"Filter by entropy failed: {str(e)}",
			"selected_atoms": "",
			"entropy": {},
		}

	return result


QUEST_SERVER_WORK_PATH= "/tmp/quest_server"

_script_dir = Path(__file__).parent
load_dotenv(_script_dir / ".env", override=True)

def create_workpath(work_path=None):
	"""
	Create the working directory for AbacusAgent, and change the current working directory to it.
	
	Args:
		work_path (str, optional): The path to the working directory. If None, a default path will be used.
	
	Returns:
		str: The path to the working directory.
	"""
	work_path = QUEST_SERVER_WORK_PATH + f"/{time.strftime('%Y%m%d%H%M%S')}"
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
if args.model == "dp":
	from dp.agent.server import CalculationMCPServer
	mcp = CalculationMCPServer(
			"QuestServer",
			host=args.host,
			port=args.port
		)
elif args.model == "fastmcp":
	from mcp.server.fastmcp import FastMCP
	mcp = FastMCP(
			"QuestServer",
			host=args.host,
			port=args.port
		)


@mcp.tool()
def filter_by_entropy(
		iter_confs: Union[List[str], str],
		reference: Union[List[str], str] = [],
		chunk_size: int = 10,
		k:int =32,
		cutoff: float =5.0,
		batch_size: int = 1000,
		h: float = 0.015,
		max_sel: int =50,
		):
	"""Select a diverse subset of configurations by maximizing dataset entropy.

	This tool performs iterative, entropy-based subset selection from a pool of candidate
	configurations ("iterative set") against an optional reference set. At each iteration,
	it scores remaining candidates by their incremental contribution to the dataset entropy
	and picks the top `chunk_size`. Selection stops when either `max_sel` is reached or the
	entropy increment falls below a small threshold.

		Backend and acceleration
		- If PyTorch is available, a GPU-accelerated path is used (quests.gpu.*); otherwise a CPU path is used.
		- Descriptors and entropy are computed via the `quests` package.

		Parameters
		- iter_confs: List[Path] | Path
			Candidate configurations to select from. Typically:
			• A path to a multi-frame extxyz/xyz file, or
			• A list of paths to structure files.
			Internally, the backend expects ASE Atoms sequences; ensure your inputs are
			compatible with the descriptor/entropy functions in `quests`.
		- reference: List[Path] | Path, default []
			Optional existing dataset to seed the selection. If empty and `chunk_size` is
			smaller than the candidate pool, the first iteration seeds the dataset with one
			chunk chosen from the candidates.
		- chunk_size: int, default 10
			Number of configurations to add in each selection iteration.
		- k: int, default 32
			Neighborhood/descriptor parameter forwarded to `quests.descriptor.get_descriptors`.
		- cutoff: float, default 5.0
			Cutoff radius for descriptor construction (Å), forwarded to `quests`.
		- batch_size: int, default 1000
			Batch size for entropy computations.
		- h: float, default 0.015
			Kernel bandwidth (smoothing) parameter for entropy estimation.
		- max_sel: int, default 50
			Upper bound on the total number of configurations to select.

	Algorithm (high-level)
	1) Initialize the reference set (from `reference` or by taking an initial chunk from
	   `iter_confs` when empty).
	2) Compute descriptors for candidates and reference with (k, cutoff).
	3) Compute initial entropy H of the reference set.
	4) Loop up to ceil(max_sel / chunk_size):
	   a) For each remaining candidate structure, compute delta-entropy w.r.t. the current
		  reference descriptors and sum per-structure contributions.
	   b) Pick the top `chunk_size` structures, append them to the reference set, and update
		  H. Stop early if the entropy gain is below ~1e-2.

	Outputs
	- Returns a TypedDict with:
		• select_atoms (Path): Path to an extxyz file ("selected.extxyz") containing the
		  selected configurations in order of selection.
		• entroy (Dict[str, Any]): Iteration log with entries like
		  {"iter_00": H0, "iter_01": H1, ..., "num_confs": N}, where Ht is the entropy
		  after iteration t and num_confs is the growing dataset size.

	Notes
	- If any error occurs, this function returns an empty Path and an empty log.
	- Memory/performance: descriptor computation scales with total atoms; consider tuning
	  `k`, `cutoff`, and `batch_size` for large datasets.
	- The result key name `entroy` is preserved for compatibility, even though it is a typo
	  of "entropy".

	Examples
	- Basic selection from a multi-frame file:
		filter_by_entropy(iter_confs=Path("candidates.extxyz"), chunk_size=20, max_sel=200)

	- Seed with an existing set and use GPU if available:
		filter_by_entropy(
			iter_confs=[Path("pool1.extxyz"), Path("pool2.extxyz")],
			reference=Path("seed.extxyz"),
			chunk_size=10, k=32, cutoff=5.0, h=0.015, max_sel=100
		)
		"""
	return filter_by_entropy_impl(
		iter_confs=iter_confs,
		reference=reference,
		chunk_size=chunk_size,
		k=k,
		cutoff=cutoff,
		batch_size=batch_size,
		h=h,
		max_sel=max_sel,
	)
	
@mcp.tool()
def build_bulk_crystal(
				formula: str,
				crystal_structure: str,
				a: float | None = None,
				c: float | None = None,
				covera: float | None = None,
				u: float | None = None,
				spacegroup: int | None = None,
				basis: List[List[float]] | None = None,
				orthorhombic: bool = False,
				cubic: bool = False,
				size: Union[int, List[int], List[List[int]]] = 1,
				vacuum: float | None = None,
				output_format: str = "extxyz",
		):
	"""Build and save a bulk crystal structure using ASE. It constructs a
		bulk crystal from a chemical formula and crystal prototype, optionally
		expands it to a supercell, and writes the result to disk.

		Key arguments
		- formula: Chemical formula understood by ASE (e.g. "Si", "Al2O3").
		- crystal_structure: Prototype string for ASE `bulk`, such as
			"fcc", "bcc", "hcp", "rocksalt", "zincblende", etc.
		- a, c, covera, u, spacegroup, basis: Optional lattice parameters and
			internal coordinates passed directly to ASE `bulk`.
		- orthorhombic, cubic: Geometry flags forwarded to ASE `bulk`.
		- size: Supercell expansion; can be an integer N (NxNxN), a 3-int list
			like [2,2,1], or a 3x3 integer matrix for a general supercell.
		- vacuum: Extra vacuum padding (in Å) added via `atoms.center`.
		- output_format: Output file format, typically "extxyz" (default),
			"xyz", "cif", or "vasp".

		Returns
		- A dictionary with:
			- status: "success" or "error".
			- message: Short description of the outcome.
			- structure_path: Absolute path to the written structure file
				(empty string on error).
			- chemical_formula: Empirical formula of the generated structure.
			- num_atoms: Number of atoms in the final supercell.
			- cell: 3x3 cell matrix as a nested list.
			- pbc: Periodic boundary condition flags as a length-3 list.

		Example
		- Create a 2x2x2 fcc Al supercell and save to extxyz:
				build_bulk_crystal(formula="Al", crystal_structure="fcc", size=[2,2,2])
		"""

	return build_bulk_crystal_impl(
				formula=formula,
				crystal_structure=crystal_structure,
				a=a,
				c=c,
				covera=covera,
				u=u,
				spacegroup=spacegroup,
				basis=basis,
				orthorhombic=orthorhombic,
				cubic=cubic,
				size=size,
				vacuum=vacuum,
				output_format=output_format,
		)

@mcp.tool()
def build_supercell(
		input_structure: str,
		size: Union[int, List[int], List[List[int]]] = 1,
		output_format: str = "extxyz",
):
	"""Build a supercell from an input structure file and save it.

		Parameters
		- input_structure: Path to the input structure file (or list of paths).
		- size: Supercell expansion. Either an int (N -> N x N x N), a 3-int
			list/tuple ([nx,ny,nz]), or a 3x3 integer matrix for arbitrary
			supercell transforms.
		- output_format: Output format, e.g. "extxyz", "xyz", "cif", "vasp".

		Returns:
		- A dictionary with:
			- status: "success" or "error".
			- message: Short description of the outcome.
			- structure_path: Absolute path to the written structure file
				(empty string on error).
			- chemical_formula: Empirical formula of the generated structure.
			- num_atoms: Number of atoms in the final supercell.
			- cell: 3x3 cell matrix as a nested list.
			- pbc: Periodic boundary condition flags as a length-3 list.
		"""

	return build_supercell_impl(
				input_structure=input_structure,
				size=size,
				output_format=output_format,
		)

@mcp.tool()
def perturb_atoms(
		structure_path: Union[str, List[str]],
		pert_num: int,
		cell_pert_fraction: float,
		atom_pert_distance: float,
		atom_pert_style: str = "normal",
		atom_pert_prob: float = 1.0,
		output_format: str = "extxyz",
		output_path: str | None = None,
):
	"""Generate perturbed configurations from a structure file and write them out.

		Arguments
		- structure_path: ASE-readable structure path (or list with the first entry used).
		- pert_num: Number of perturbed structures to generate.
		- cell_pert_fraction: Fractional cell distortion magnitude.
		- atom_pert_distance: Maximum per-atom displacement (Å).
		- atom_pert_style: Displacement distribution (`normal`, `uniform`, `const`).
		- atom_pert_prob: Probability that each atom is perturbed.
		- output_format: Output format such as `extxyz` (default), `xyz`, `cif`, `vasp`.
		- output_path: Optional explicit output file path; auto-generated when omitted.

		Returns
		- A dictionary with:
			-status: "success" or "error".
			-message: Short description of the outcome.
			-structure_path: Absolute path to the written structure file
				(empty string on error).
			-num_structures: Number of perturbed structures generated.
			-num_atoms_per_structure: Number of atoms in each perturbed structure.
		"""

	if isinstance(structure_path, list):
		structure_path = structure_path[0]

	return perturb_atoms_impl(
				structure_path=structure_path,
				pert_num=pert_num,
				cell_pert_fraction=cell_pert_fraction,
				atom_pert_distance=atom_pert_distance,
				atom_pert_style=atom_pert_style,
				atom_pert_prob=atom_pert_prob,
				output_format=output_format,
				output_path=output_path,
		)

@mcp.tool()
def inspect_structure(
	structure_path: str,
):
	"""Read an ASE-compatible structure file and return useful metadata.

	Args:
		structure_path (str): Path to an ASE-readable structure file (extxyz, xyz, cif, vasp, ...).

	Returns:
		dict: A result dictionary with keys similar to the implementation:
		- ``status``: "success" or "error".
		- ``message``: Short description or error message.
		- ``structure_path``: Absolute path to the input file (or empty string on error).
		- ``num_frames``: Number of frames read from the file.
		- ``chemical_formulas``: List of unique chemical formulas found.
		- ``num_atoms``: List of unique atom counts per frame.
		- ``info_keys`` / ``array_keys``: Lists of metadata and array keys present in frames.

	Example:
		inspect_structure("structures.extxyz")

	"""

	return inspect_structure_impl(
		structure_path=structure_path,
	)


if __name__ == "__main__":
	create_workpath()
	mcp.run(transport=args.transport)