import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, TypedDict, Union, Tuple, Callable
from ase.db import connect
from ase.io import write,read

from matcreator.tools.util.common import generate_work_path

# Globals configured at runtime
DEFAULT_DB_PATH: Optional[Path] = Path(os.environ.get("ASE_DB_PATH","")).resolve()


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
    
class QueryResult(TypedDict):
    """Result structure for model training"""
    query: str
    count: int
    ids: List[int]
    formulas: List[str]
    results: List[Dict[str, Any]]

#@mcp.tool()
def read_user_structure(
    structures: Union[List[Path], Path],
):
    """Extract chemical compositions from user-provided structure file(s) for downstream DB queries.

    Purpose:
        This tool does NOT query the ASE database. Instead, it parses one or more input structure
        files (each may contain multiple frames) to extract chemical compositions. The aggregated
        frames are written into a single extxyz file, and the list of compositions is returned so
        the agent can build a query with `query_compounds`.

    Args:
        structures (Path | List[Path]): One or more structure files to read (e.g., .cif, .xyz, .extxyz,
            POSCAR). Each file may contain multiple frames; all frames will be aggregated.

    Returns:
        AtomsInfoResult:
            - formulas: List[str] of empirical formulas (e.g., "NaCl", "SiO2").
            - formulas_full: List[str] of full chemical formulas as reported by ASE.
            - query_atoms_path: Path to an extxyz file containing all parsed frames, which can be
              used for inspection or further processing.

    Notes:
        - Use the returned `formulas`/`formulas_full` to construct selectors for `query_compounds`.
          For example, to search for entries containing Na and Cl, build a selector like 'Na,Cl' or a
          more specific expression using the database’s supported fields.
        - This function performs no database I/O.
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
        logging.error("Error in atoms_info: %s", e)
        return AtomsInfoResult(
            formulas=[],formulas_full=[],query_atoms_path=""
        )        



#@mcp.tool()
def query_compounds(
    selection: Union[dict,int,str,List[Union[str,Tuple]]]=None,
    exclusive_elements: Union[str, List[str]] = None,
    limit: Optional[int] = None,
    db_path: Optional[Path] = None,
    custom_args: Dict[str, Any] = {},
) -> Dict[str, Any]:
    """Query an ASE database for structures using flexible selectors and optional filters.

    Overview:
        Wraps `ase.db.connect(...).select(...)` and returns a compact summary of matching rows.
        The database path is resolved from `db_path` or the `ASE_DB_PATH` environment variable.

    Parameters:
        selection (int | str | list[str | tuple] | None):
            Selector(s) passed to ASE DB. Supported forms include:
            - int: single row id, e.g. `123`.
            - str: a single expression or a comma-separated list of expressions:
                • no-key: 'Si' # Note: these would select any entry with 'Si' in formula, inclduing 'SiO2', etc.
                • comparisons: 'key=value', 'key!=value', 'key<value', 'key<=value',
                  'key>value', 'key>=value'
                • combined: 'formula=Si32,pbc=True,energy<-1.0' or 'Si,O'
            - list[str]: list of string expressions, e.g. `['formula=Si32', 'pbc=True']`.
            - list[tuple]: list of `(key, op, value)` tuples, e.g. `[("energy", "<", -1.0)]`.

        exclusive_elements (str | set[str] | None):
            Optional post-filtering by chemical elements. Only entries whose structures within the chemical space specified can
            be included in the results. examples: "Ba,Ti,O" or {"Ba", "Ti", "O"}.
        
        limit (int | None):
            Maximum number of rows to return (applied during ASE selection).

        db_path (Path | None):
            Path to the ASE database. Defaults to `ASE_DB_PATH` if not provided.

        Other key arguments that may be forwarded to `ase.db.Select` (common options):
            - explain (bool): Print query plan.
            - verbosity (int): 0, 1 or 2.
            - offset (int): Skip initial rows.
            - sort (str): e.g. 'energy' or '-energy' for descending.
            - include_data (bool): False to skip reading data payloads.
            - columns ('all' | list[str]): Restrict SQL columns for speed.

    Returns:
        QueryResult:
            - query (str): Echo of the selection input (stringified).
            - count (int): Number of unique rows returned.
            - ids (List[int]): Unique row ids.
            - formulas (List[str]): Unique empirical formulas (if available).
            - results (List[Dict[str, Any]]): One dict per row with keys:
                { 'id', 'name', 'formula', 'tags', 'key_value_pairs' }.

    Examples (selection):
        # 1) Single id
        >>> query_compounds(123)

        # 2) Single condition (string)
        >>> query_compounds('Si') # matches any entry with 'Si' in formula
        >>> query_compounds('formula=Si32')
        >>> query_compounds('energy<-1.0')
        >>> query_compounds('pbc=True')

        # 3) Comma-separated conditions (string)
        >>> query_compounds('Si,O') # matches any entry with 'Si' and 'O' in formula
        >>> query_compounds('formula=Si32,pbc=True,energy<-1.0')

        # 4) List of string conditions
        >>> query_compounds(['formula=Si32', 'pbc=True', 'energy<-1.0'])

        # 5) List of (key, op, value) tuples
        >>> query_compounds([('energy', '<', -1.0), ('pbc', '=', True)])

        # 6) Dict selector (advanced; forwarded to ASE)
        >>> query_compounds({'calculator': 'deepmd'})


    Notes:
        - Use `sort='-energy'` and `limit=K` to quickly retrieve low/high energy candidates.
        - Set `include_data=False` for faster metadata-only scans.
    """
    path = _resolve_db_path(db_path)
    logging.info(f"Querying ASE database at {path} with selection: {selection}")
    results: List[Dict[str, Any]] = []
    seen_ids: set[int] = set()
    formulas: set[str] = set()
    try:
        if exclusive_elements:
            filter = _exclusive_elements(exclusive_elements)
        else:
            filter = None
            
        with connect(path) as db:
            for row in db.select(selection,filter=filter,
                                 limit=limit,**custom_args):
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
                query=selection,count=len(results),results=results,ids=seen_ids,formulas=formulas
            )
    except Exception as e:
        logging.error("Error querying database: %s", e)
        return QueryResult(
            query=selection,count=0,results=[],ids=[],formulas=[]
        )


class ExportResult(TypedDict):
    """Result structure for export_entries"""
    output_file: Path
    metadata_file: Path
    counts: Dict[str, int]

#@mcp.tool()
def export_entries(
    ids: List[int],
    *,
    fmt: Literal["extxyz", "cif", "traj"] = "extxyz",
    db_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Export selected ASE database entries to a single structure file with summary stats."""
    if not ids:
        raise ValueError("ids must contain at least one entry id")

    path = _resolve_db_path(db_path)
    work_path=Path(generate_work_path())
    work_path = work_path.expanduser().resolve()
    work_path.mkdir(parents=True, exist_ok=True)

    atoms_collection: List[Any] = []
    formulas: set[str] = set()
    total_exported = 0

    metadata_path = work_path / "exported_metadata.json"
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
        combined_path = work_path / combined_filename

        payload = atoms_collection[0] if len(atoms_collection) == 1 else atoms_collection
        write(combined_path, payload, format=fmt)

        counts = {
            "total_exported": total_exported,
            "unique_formulas": len(formulas),
        }

        return ExportResult(
            output_file=combined_path,
            metadata_file=metadata_path,
            counts=counts,
        )
    except Exception as e:
        logging.error("Error exporting entries: %s", e)
        return ExportResult(
            output_file=Path(""),
            metadata_file=Path(""),
            counts={},
        )

def _exclusive_elements(elements:Union[str, List[str]]) -> Callable:
    """Return True if the row's structure contains only elements in the allowed set."""
    if isinstance(elements, str):
        elements = elements.split(',')
    return lambda row: set(row.symbols) <= set(elements)