import logging
from pathlib import Path
from typing import  List, Union
from ase.io import write,read
from ase.atoms import Atoms
import numpy as np
import random
import logging
import traceback
from matcreator.tools.util.common import generate_work_path
import torch 

def h_filter_cpu_gpu(
    use_torch:bool,
    iter_confs,
    dset_confs = [],
    chunk_size: int = 10,
    max_sel: int = 100,
    k=32,
    cutoff=5.0,
    batch_size: int = 1000,
    h = 0.015,
    dtype='float32',
    **kwargs
):
    """Filter configurations based on entropy.

    Args:
        iter_confs (List[Atoms]): The configurations to iterate over.
        dset_confs (List[Atoms], optional): The reference configurations. Defaults to [].
        chunk_size (int, optional): The number of configurations to process at once. Defaults to 10.
        max_sel (int, optional): _description_. Defaults to 100.
        k (int, optional): _description_. Defaults to 32.
        cutoff (float, optional): _description_. Defaults to 5.0.
        batch_size (int, optional): _description_. Defaults to 1000.
        h (float, optional): _description_. Defaults to 0.015.
        dtype (str, optional): _description_. Defaults to 'float32'.

    Returns:
        _type_: _description_
    """
    if use_torch:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        entropy = entropy_gpu.entropy
        delta_entropy = entropy_gpu.delta_entropy
    else:
        device = 'cpu'
        entropy = entropy_cpu.entropy
        delta_entropy = entropy_cpu.delta_entropy

    
    result = {}
    last_H = 0
    ndset = len(dset_confs)

    atoms_list = dset_confs + iter_confs 
    ndata = len(atoms_list)
    flag = np.zeros((ndata,), dtype=bool)
    num_ref=len(dset_confs)

    if ndset == 0:
        selected_indices = np.random.choice(ndata, chunk_size, replace=False)
    else:
        selected_indices = np.arange(ndset)

    flag[selected_indices] = True
    list_of_indices = []
    list_of_indices.append(selected_indices)
    
    descriptors      = get_descriptors(atoms_list, k=k, cutoff=cutoff, dtype=dtype)
    natoms_list      = np.array([len(atoms) for atoms in atoms_list])
    natoms_sum_list  = np.hstack([0, np.cumsum(natoms_list)])

    max_iter = int(round(ndata / chunk_size) - 1)
    for ii in range(max_iter):
        selected_stru_indices = np.where(flag)[0]
        if np.all(flag) or len(selected_stru_indices) > max_sel:
            break
        selected_atom_indices = np.concatenate(
            [np.arange(natoms_sum_list[i], natoms_sum_list[i+1]) for i in selected_stru_indices]
        )
        selected_atom_descriptors = descriptors[selected_atom_indices]

        if use_torch:
            selected_atom_descriptors = torch.tensor(selected_atom_descriptors, device=device)

        H = entropy(selected_atom_descriptors, h=h, batch_size=batch_size)
        
        logging.info(f"Iteration {ii+1}/{max_iter}, selected {len(selected_stru_indices)} configurations, entropy {H:.4f}")
        result.update({f"iter_{ii+1:02d}": H, "num_confs": len(selected_stru_indices)})
        dH = H - last_H
        last_H = H
        if dH < 1e-2:
            logging.info(f"Entropy increase {dH:.4f} is less than 1e-2, stopping selection.")
            break

        remain_stru_indices = np.where(~flag)[0]
        remain_atom_indices = np.concatenate(
            [np.arange(natoms_sum_list[i], natoms_sum_list[i+1]) for i in remain_stru_indices]
        )
        remain_atom_descriptors = descriptors[remain_atom_indices]

        tmp_sum_list = np.hstack([0, np.cumsum(natoms_list[remain_stru_indices])])

        if use_torch:
            remain_atom_descriptors = torch.tensor(remain_atom_descriptors, device=device)

        delta = delta_entropy(remain_atom_descriptors, selected_atom_descriptors, h=h,batch_size=batch_size, **kwargs)

        if use_torch:
            delta = delta.cpu().numpy()
        
        delta_sums = np.array([np.sum(delta[tmp_sum_list[i]: tmp_sum_list[i+1]]) for i in range(len(remain_stru_indices))])

        sorted_index = np.argsort(delta_sums)

        if (len(sorted_index) < chunk_size):
            selected_indices = remain_stru_indices
        else:
            selected_indices = remain_stru_indices[sorted_index[-chunk_size:]]
        flag[selected_indices] = True
        list_of_indices.append(selected_indices)
    
    all_the_indices = np.concatenate(list_of_indices)
    return iter_confs[all_the_indices[num_ref:]], result

#@mcp.tool()
#@log_step(step_name="explore_filter_by_entropy")
def filter_by_entropy(
    iter_confs: Union[List[str], str],
    reference: Union[List[str], str] = [],
    chunk_size: int = 10,
    k:int=32,
    cutoff:float = 5.0,
    batch_size: int = 1000,
    h: float = 0.015,
    max_sel: int =50,
    #**kwargs
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
    try:
        if isinstance(iter_confs, str):
            iter_confs = read(iter_confs, index=":")
        elif isinstance(iter_confs, list) and all(isinstance(p, str) for p in iter_confs):
            iter_confs = [read(p, index=":") for p in iter_confs]
            iter_confs = [atom for sublist in iter_confs for atom in sublist] # flatten
        
        if isinstance(reference, str):
            reference = read(reference, index=":")
        elif isinstance(reference, list) and all(isinstance(p, str) for p in reference):
            reference = [read(p, index=":") for p in reference]
            reference = [atom for sublist in reference for atom in sublist] # flatten
        
        try:
            import torch
            logging.info("Using torch entropy calculation")
            select_atoms, select_result = h_filter_cpu_gpu(use_torch=True,
                                                           iter_confs=iter_confs,
                                                           dset_confs=reference,
                                                           chunk_size=chunk_size,
                                                           max_sel=max_sel,
                                                           k=k,
                                                           cutoff=cutoff,
                                                           batch_size=batch_size,
                                                           h=h)
        except ImportError:
            logging.info("Using CPU entropy (torch not available)")
            select_atoms, select_result = h_filter_cpu_gpu(use_torch=False,
                                                           iter_confs=iter_confs,
                                                           dset_confs=reference,
                                                           chunk_size=chunk_size,
                                                           max_sel=max_sel,
                                                           k=k,
                                                           cutoff=cutoff,
                                                           batch_size=batch_size,
                                                           h=h)
        work_path=Path(generate_work_path())
        work_path=work_path.expanduser().resolve()
        work_path.mkdir(parents=True,exist_ok=True)
        select_atoms_path = work_path / "selected.extxyz"
        write(select_atoms_path, select_atoms)
        
        result={
            "status":"success",
            "message":"Filter by entropy completed.",
            "selected_atoms": str(select_atoms_path.resolve()),
            "entropy": select_result
        }
        
    except Exception as e:
        logging.error(f"Error in filter_by_entropy: {str(e)}. Traceback: {traceback.format_exc()}")
        
        result={
            "status":"error",
            "message":f"Filter by entropy failed: {str(e)}",
            "selected_atoms": "",
            "entropy": {}
        }
        
    return result
