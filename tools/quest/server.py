import argparse
import os
import argparse
from typing import Union, List
import time
from dotenv import load_dotenv
from pathlib import Path
from matcreator.tools.structure_builder import (
    build_bulk_crystal as _build_bulk_crystal,
    build_supercell as _build_supercell,
    inspect_structure as _inspect_structure,
    perturb_atoms as _perturb_atoms,
)
from matcreator.tools.quest import (
    filter_by_entropy as _filter_by_entropy,
    )

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

        return _build_bulk_crystal(
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

        return _build_supercell(
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

        return _perturb_atoms(
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
    """Read an ASE-compatible structure file and report metadata such as frame count and properties.
    """

    return _inspect_structure(
        structure_path=structure_path,
    )


if __name__ == "__main__":
    create_workpath()
    mcp.run(transport=args.transport)