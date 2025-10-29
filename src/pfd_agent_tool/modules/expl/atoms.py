import logging
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, TypedDict, Union
from ase.io import write,read
from .filter import _h_filter_cpu, _h_filter_gpu
from pfd_agent_tool.init_mcp import mcp
from pfd_agent_tool.modules.util.common import generate_work_path
from pfd_agent_tool.modules.log.log import log_step

@mcp.tool()
@log_step(step_name="explore_filter_by_entropy")
def filter_by_entropy(
    iter_confs: Union[List[Path], Path],
    reference: Union[List[Path], Path] = [],
    chunk_size: int = 10,
    k=32,
    cutoff=5.0,
    batch_size: int = 1000,
    h = 0.015,
    max_sel: int =100,
    **kwargs
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
    - max_sel: int, default 100
        Upper bound on the total number of configurations to select.
    - **kwargs: Any
        Advanced options forwarded to the GPU delta-entropy path (e.g., device hints).

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
        if isinstance(iter_confs, Path):
            iter_confs = read(iter_confs, index=":")
        elif isinstance(iter_confs, list) and all(isinstance(p, Path) for p in iter_confs):
            iter_confs = [read(p, index=":") for p in iter_confs]
            iter_confs = [atom for sublist in iter_confs for atom in sublist] # flatten
        
        if isinstance(reference, Path):
            reference = read(reference, index=":")
        elif isinstance(reference, list) and all(isinstance(p, Path) for p in reference):
            reference = [read(p, index=":") for p in reference]
            reference = [atom for sublist in reference for atom in sublist] # flatten
        
        try:
            import torch
            logging.info("Using torch entropy calculation")
            select_atoms, select_result = _h_filter_gpu(iter_confs,reference,chunk_size=chunk_size,max_sel=max_sel,
                                 k=k,cutoff=cutoff,batch_size=batch_size,h=h,**kwargs)
        except ImportError:
            logging.info("Using CPU entropy (torch not available)")
            select_atoms, select_result = _h_filter_cpu(iter_confs,reference,chunk_size=chunk_size,max_sel=max_sel,
                                 k=k,cutoff=cutoff,batch_size=batch_size,h=h,**kwargs)
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
        logging.error(f"Error in filter_by_entropy: {str(e)}")
        
        result={
            "status":"error",
            "message":f"Filter by entropy failed: {str(e)}",
            "selected_atoms": "",
            "entropy": {}
        }
        
    return result
