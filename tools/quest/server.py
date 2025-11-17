import argparse
import os
import argparse
from typing import Union, List
import time
from matcreator.tools.quest import (
    filter_by_entropy as _filter_by_entropy,
    )

QUEST_SERVER_WORK_PATH= "/tmp/quest_server",


def create_workpath(work_path=None):
    """
    Create the working directory for AbacusAgent, and change the current working directory to it.
    
    Args:
        work_path (str, optional): The path to the working directory. If None, a default path will be used.
    
    Returns:
        str: The path to the working directory.
    """
    work_path = QUEST_SERVER_WORK_PATH  + f"/{time.strftime('%Y%m%d%H%M%S')}"
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
    return _filter_by_entropy(
            iter_confs=iter_confs,
            reference=reference,
            chunk_size=chunk_size,
            k=k,
            cutoff=cutoff,
            batch_size=batch_size,
            h=h,
            max_sel=max_sel,
        )
    

if __name__ == "__main__":
    create_workpath()
    mcp.run(transport=args.transport)