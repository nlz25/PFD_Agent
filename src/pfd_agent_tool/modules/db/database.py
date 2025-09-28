import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, TypedDict, Union
from dotenv import load_dotenv
from ase.db import connect
from ase.io import write,read

from pfd_agent_tool.init_mcp import mcp
load_dotenv()

# Globals configured at runtime
DEFAULT_DB_PATH: Path = Path(os.environ.get("ASE_DB_PATH")).resolve()

logger = logging.getLogger("ase_db_server")

def _resolve_db_path(db_path: Optional[Path]) -> Path:
    path = (db_path or DEFAULT_DB_PATH).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"ASE database not found at {path}")
    return path

class AtomsInfoResult(TypedDict):
    """Result structure for model training"""
    formulas: List[str]
    formulas_full: List[str]
    query_atoms_path: Union[Path, str]

@mcp.tool()
def read_user_structure(
    structures: Union[List[Path], Path],
):
    """Query the ASE database by atomic structure.
    
    This tool allows users to query the ASE database for entries that match a given atomic structure.
    The matching is performed based on structural similarity, with an optional tolerance parameter
    to define the acceptable deviation in atomic positions or lattice parameters.
    
    Args:
        structure (Path): Path to the atomic structure file (e.g., CIF, XYZ, or other supported formats).
        db_path (Optional[Path]): Path to the ASE database file. If not provided, the default database path
            will be used (configured via the ASE_DB_PATH environment variable or a default value).
        tolerance (float): Tolerance for structural matching. This defines the acceptable deviation
            in atomic positions or lattice parameters for a match. Default is 0.1.
        limit (Optional[int]): Maximum number of matching entries to return. If not provided, all matches
            will be returned.
    """
    try:
        # get atoms ls 
        atoms_ls=[]
        if isinstance(structures, Path):
            structures = [structures]
        for structure in structures:
            atoms = read(structure,index=':')
            atoms_ls.extend(atoms)
        formulas=[]
        formulas_full=[]  
        for atoms in atoms_ls:
            formula = atoms.get_chemical_formula(empirical=True)
            formula_full = atoms.get_chemical_formula()
            if formula_full:
                formulas_full.append(formula_full)
            if formula:
                formulas.append(formula)
                
        query_atoms_path = Path('query_atoms.extxyz')
        #sorted_structures = sorted([str(s.resolve()) for s in structures])
        write(query_atoms_path, atoms_ls, format="extxyz")
        return AtomsInfoResult(
            formulas=formulas,
            formulas_full=formulas_full,
            query_atoms_path=query_atoms_path
        )       
    except Exception as e:
        logger.error("Error in atoms_info: %s", e)
        return AtomsInfoResult(
            formulas=[],formulas_full=[],query_atoms_path=""
        )        

class QueryResult(TypedDict):
    """Result structure for model training"""
    query: str
    count: int
    ids: List[int]
    formulas: List[str]
    results: List[Dict[str, Any]]

@mcp.tool()
def query_compounds(
    selectors: dict,
    *,
    limit: Optional[int] = None,
    db_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Query the ASE database with a list of selectors. 
    example: selectors = {
        "formula": "Si32",
        }
    By default, there is no limit on the number of results returned.
    
    
    """
    path = _resolve_db_path(db_path)
    results: List[Dict[str, Any]] = []
    seen_ids: set[int] = set()
    formulas: set[str] = set()
    try:
        with connect(path) as db:
            for row in db.select(**selectors,limit=limit):
                if row.id in seen_ids:
                    continue
                seen_ids.add(row.id)
                formulas.add(row.get("formula"))
                results.append(
                {
                        "id": row.id,
                        "name": row.get("name"),
                        "formula": row.get("formula"),
                        "tags": row.get("tags"),
                        "key_value_pairs": dict(row.key_value_pairs or {}),
                    }
                )
            return  QueryResult(
                query=selectors,count=len(results),results=results,ids=seen_ids,formulas=formulas
            )
    except Exception as e:
        logger.error("Error querying database: %s", e)
        return QueryResult(
            query=selectors,count=0,results=[],ids=[],formulas=[]
        )


class ExportResult(TypedDict):
    """Result structure for export_entries"""
    output_file: str
    metadata_file: str
    counts: Dict[str, int]

@mcp.tool()
def export_entries(
    ids: List[int],
    *,
    output_dir: Path,
    fmt: Literal["extxyz", "cif", "traj"] = "extxyz",
    db_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Export selected ASE database entries to a single structure file with summary stats."""
    if not ids:
        raise ValueError("ids must contain at least one entry id")

    path = _resolve_db_path(db_path)
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    atoms_collection: List[Any] = []
    formulas: set[str] = set()
    total_exported = 0

    metadata_path = output_dir / "exported_metadata.json"
    try:
        with metadata_path.open("w", encoding="utf-8") as meta_fp:
            with connect(path) as db:
                for entry_id in ids:
                    row = db.get(id=entry_id)
                    atoms = row.toatoms()
                    atoms_collection.append(atoms)
                    formula = row.get("formula") or atoms.get_chemical_formula(empirical=True)
                    if formula:
                        formulas.add(formula)
                    record = {
                        "id": entry_id,
                        "name": row.get("name"),
                        "formula": row.get("formula"),
                        "tags": row.get("tags"),
                    }
                    meta_fp.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                    total_exported += 1

        combined_filename = f"exported_structures.{fmt}"
        combined_path = output_dir / combined_filename

        payload = atoms_collection[0] if len(atoms_collection) == 1 else atoms_collection
        write(combined_path, payload, format=fmt)

        counts = {
            "total_exported": total_exported,
            "unique_formulas": len(formulas),
        }

        return ExportResult(
            output_file=combined_path.as_uri(),
            metadata_file=metadata_path.as_uri(),
            counts=counts,
        )
    except Exception as e:
        logger.error("Error exporting entries: %s", e)
        return ExportResult(
            output_file="",
            metadata_file="",
            counts={},
        )
