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


def _h_filter_cpu(
    iter_confs: List[Atoms],
    dset_confs: List[Atoms]=[],
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
    from quests.descriptor import get_descriptors
    from quests.entropy import entropy,delta_entropy
    num_ref=len(dset_confs)
    if len(dset_confs) == 0:
        if chunk_size >= len(iter_confs):
            return iter_confs
        random.shuffle(iter_confs)
        dset_confs = iter_confs[:chunk_size]
        iter_confs = iter_confs[chunk_size:]
        num_ref=0
        max_sel-= chunk_size
        
    max_iter = min(max_sel//chunk_size+(max_sel%chunk_size>0), 
                   len(iter_confs)//chunk_size+(len(iter_confs)%chunk_size>0))
    iter_desc = get_descriptors(iter_confs, k=k, cutoff=cutoff,dtype=dtype)
    dset_desc = get_descriptors(dset_confs, k=k, cutoff=cutoff,dtype=dtype)
    num_atoms_per_structure_iter = [atoms.get_number_of_atoms() for atoms in iter_confs]
    atom_indices_iter = []
    start = 0
    for n in num_atoms_per_structure_iter:
        end = start + n
        atom_indices_iter.append((start, end))
        start = end
        
    H_list = []
    # initial entropy
    H= entropy(dset_desc, h=h, batch_size=batch_size)
    logging.info(f"Initial entropy with {len(dset_confs)} reference configurations: {H:.4f}")
    H_list.append(H)
    result = {}
    result.update({"iter_00": H, "num_confs": len(dset_confs)})
    indices = []
    for ii in range(max_iter):
        re_indices = [i for i in range(len(iter_confs)) if i not in indices]
        re_confs = [iter_confs[i] for i in re_indices]
        re_desc = [iter_desc[atom_indices_iter[i][0]:atom_indices_iter[i][1]] for i in re_indices]
        x = np.vstack(re_desc)
        delta = delta_entropy(x, dset_desc, h=h,batch_size=batch_size)
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
        dset_desc_ls=[dset_desc]
        for idx in selected_indices:
            indices.append(idx)
            dset_confs.append(iter_confs[idx])
            dset_desc_ls.append(iter_desc[atom_indices_iter[idx][0]:atom_indices_iter[idx][1]])
        dset_desc = np.vstack(dset_desc_ls)
        H = entropy(dset_desc, h=h, batch_size=batch_size)
        dH = H - H_list[-1]
        H_list.append(H)
        logging.info(f"Iteration {ii+1}/{max_iter}, selected {len(dset_confs)} configurations, entropy {H:.4f}")
        result.update({f"iter_{ii+1:02d}": H, "num_confs": len(dset_confs)})
        if dH < 1e-2:
            logging.info(f"Entropy increase {dH:.4f} is less than 1e-2, stopping selection.")
            break
    return dset_confs[num_ref:], result # return only the newly selected ones

def _h_filter_gpu(
    iter_confs: List[Atoms],
    dset_confs: List[Atoms]=[],
    chunk_size: int = 10,
    max_sel: int = 100,
    k=32,
    cutoff=5.0,
    batch_size: int = 1000,
    h = 0.015,
    dtype='float32',
    **kwargs
):
    import torch
    from quests.descriptor import get_descriptors
    from quests.gpu.entropy import delta_entropy,entropy
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # logging results for filtering
    result = {}
    num_ref=len(dset_confs)
    if len(dset_confs) == 0:
        if chunk_size >= len(iter_confs):
            return iter_confs
        random.shuffle(iter_confs)
        dset_confs = iter_confs[:chunk_size]
        iter_confs = iter_confs[chunk_size:]
        num_ref=0
        max_sel-= chunk_size

    max_iter = min(max_sel//chunk_size+(max_sel%chunk_size>0), 
                   len(iter_confs)//chunk_size+(len(iter_confs)%chunk_size>0))
    
    iter_desc = get_descriptors(iter_confs, k=k, cutoff=cutoff,dtype=dtype)
    dset_desc = get_descriptors(dset_confs, k=k, cutoff=cutoff,dtype=dtype)

    num_atoms_per_structure_iter = [atoms.get_number_of_atoms() for atoms in iter_confs]
    atom_indices_iter = []
    start = 0
    for n in num_atoms_per_structure_iter:
        end = start + n
        atom_indices_iter.append((start, end))
        start = end
    
    H_list = []
    x= torch.tensor(dset_desc,device=device, dtype=torch.float32)
    H= entropy(x, h=h, batch_size=batch_size,device=device)
    logging.info(f"Initial entropy with {len(dset_confs)} reference configurations: {H:.4f}")
    H_list.append(float(H.cpu().numpy()))
    result.update({"num_confs": len(dset_confs),"iter_00": float(H.cpu().numpy())})
    indices = []
    for ii in range(max_iter):
        re_indices = [i for i in range(len(iter_confs)) if i not in indices]
        re_confs = [iter_confs[i] for i in re_indices]
        re_desc = [iter_desc[atom_indices_iter[i][0]:atom_indices_iter[i][1]] for i in re_indices]
        x = torch.tensor(np.vstack(re_desc),device=device, dtype=torch.float32)
        y = torch.tensor(dset_desc, device=device, dtype=torch.float32)
        delta = delta_entropy(x, y, h=h,batch_size=batch_size, device=device)
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
        dset_desc_ls=[dset_desc]
        for idx in selected_indices:
            indices.append(idx)
            dset_confs.append(iter_confs[idx])
            dset_desc_ls.append(iter_desc[atom_indices_iter[idx][0]:atom_indices_iter[idx][1]])
        dset_desc = np.vstack(dset_desc_ls)
        y = torch.tensor(dset_desc, device=device, dtype=torch.float32)
        H = entropy(y, h=h, batch_size=batch_size,device=device)
        dH = H - H_list[-1]
        H_list.append(float(H.cpu().numpy()))
        logging.info(f"Iteration {ii+1}/{max_iter}, selected {len(dset_confs)} configurations, entropy {H:.4f}")
        result.update({f"iter_{ii+1:02d}": float(H.cpu().numpy()), "num_confs": len(dset_confs)})
        if dH < 1e-2:
            logging.info(f"Entropy increase {float(dH.cpu().numpy()):.4f} is less than 1e-2, stopping selection.")
            break
    return dset_confs[num_ref:], result # return only the newly selected ones

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
            select_atoms, select_result = _h_filter_gpu(iter_confs,reference,chunk_size=chunk_size,max_sel=max_sel,
                                 k=k,cutoff=cutoff,batch_size=batch_size,h=h,#**kwargs
                                 )
        except ImportError:
            logging.info("Using CPU entropy (torch not available)")
            select_atoms, select_result = _h_filter_cpu(iter_confs,reference,chunk_size=chunk_size,max_sel=max_sel,
                                 k=k,cutoff=cutoff,batch_size=batch_size,h=h,#**kwargs
                                 )
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
