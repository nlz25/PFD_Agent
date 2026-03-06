import argparse
import json
import logging
import os
import random
import sqlite3
import time
import re
import traceback
import uuid
from datetime import datetime
from dotenv import load_dotenv
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, TypedDict, Union
from ase.db import connect
from ase.io import read, write
from mcp.server.fastmcp import FastMCP

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    filename='agent.log', 
    filemode='w'         
)
## ===========================
## Database tool implementations
## ==========================

def generate_work_path(create: bool = True) -> str:
	"""Return a unique work dir path and create it by default."""
	calling_function = traceback.extract_stack(limit=2)[-2].name
	current_time = time.strftime("%Y%m%d%H%M%S")
	random_string = str(uuid.uuid4())[:8]
	work_path = f"{current_time}.{calling_function}.{random_string}"
	if create:
		os.makedirs(work_path, exist_ok=True)
	return work_path

class AtomsInfoResult(TypedDict):
    """Result structure for model training"""
    formulas: List[str]
    formulas_full: List[str]
    query_atoms_path: Union[Path, str]
    
class QueryResult(TypedDict):
    """Result structure for query_compounds - brief summary only"""
    query: str
    count: int
    unique_formulas: List[str]
    sample_ids: List[int]  # First few ids as examples

# ---------------------------------------------------------------------------
# Helpers for the normalized nodes/datasets/dataset_elements schema
# ---------------------------------------------------------------------------

def _ensure_schema(conn: sqlite3.Connection) -> None:
    """Create new tables if they don't exist yet (idempotent)."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS nodes (
            node_id      INTEGER PRIMARY KEY AUTOINCREMENT,
            name         TEXT NOT NULL UNIQUE,
            description  TEXT,
            functional   TEXT,
            code         TEXT,
            cutoff_eV    REAL,
            pseudopot    TEXT,
            kspacing     REAL,
            spin_pol     INTEGER DEFAULT 0,
            vdw          TEXT,
            extra_params TEXT,
            created_at   TEXT
        );
        CREATE TABLE IF NOT EXISTS datasets (
            dataset_id   INTEGER PRIMARY KEY AUTOINCREMENT,
            node_id      INTEGER NOT NULL REFERENCES nodes(node_id),
            elements     TEXT NOT NULL,
            n_elements   INTEGER,
            system_type  TEXT,
            field        TEXT,
            entries      INTEGER,
            source       TEXT,
            path         TEXT,
            has_forces   INTEGER DEFAULT 0,
            has_stress   INTEGER DEFAULT 0,
            has_energy   INTEGER DEFAULT 0,
            energy_min   REAL,
            energy_max   REAL,
            created_at   TEXT
        );
        CREATE TABLE IF NOT EXISTS dataset_elements (
            dataset_id  INTEGER NOT NULL REFERENCES datasets(dataset_id),
            element     TEXT NOT NULL,
            PRIMARY KEY (dataset_id, element)
        );
        CREATE INDEX IF NOT EXISTS idx_de_element  ON dataset_elements(element);
        CREATE INDEX IF NOT EXISTS idx_ds_elements ON datasets(elements);
        CREATE INDEX IF NOT EXISTS idx_ds_node     ON datasets(node_id);
    """)


def _get_or_create_user_node(conn: sqlite3.Connection, now: str) -> int:
    """Return node_id for the 'User Upload' node, creating it if needed."""
    cur = conn.cursor()
    cur.execute("SELECT node_id FROM nodes WHERE name = 'User Upload'")
    row = cur.fetchone()
    if row:
        return row[0]
    cur.execute(
        "INSERT INTO nodes (name, description, created_at) VALUES (?, ?, ?)",
        ("User Upload", "Frames uploaded by the user; DFT settings unknown.", now),
    )
    return cur.lastrowid


def _save_extxyz_to_db(extxyz_path: str,
                      info_db_path: str,
                      ase_db_path: str,
                      db_path_info: str):

    db = connect(ase_db_path)
    images = read(extxyz_path, format="extxyz", index=":")

    elements_set: set[str] = set()
    has_forces = has_stress = has_energy = False
    energy_vals: list[float] = []
    for item in images:
        elements_set.update(item.get_chemical_symbols())
        db.write(item)
        if item.calc is not None:
            results = item.calc.results
            if "forces" in results:
                has_forces = True
            if "stress" in results:
                has_stress = True
            if "energy" in results:
                has_energy = True
                energy_vals.append(results["energy"])

    elements_list = sorted(elements_set)
    elements_str = "-".join(elements_list)
    now = datetime.now().isoformat()

    with sqlite3.connect(info_db_path) as conn:
        _ensure_schema(conn)
        node_id = _get_or_create_user_node(conn, now)
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO datasets
                (node_id, elements, n_elements, system_type, field,
                 entries, source, path,
                 has_forces, has_stress, has_energy,
                 energy_min, energy_max, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                node_id, elements_str, len(elements_list),
                "Bulk", "User Upload",
                len(images), "User Calculation", db_path_info,
                int(has_forces), int(has_stress), int(has_energy),
                min(energy_vals) if energy_vals else None,
                max(energy_vals) if energy_vals else None,
                now,
            ),
        )
        dataset_id = cur.lastrowid
        for element in elements_list:
            cur.execute(
                "INSERT OR IGNORE INTO dataset_elements (dataset_id, element) VALUES (?, ?)",
                (dataset_id, element),
            )
        conn.commit()


SQL_FORBIDDEN_KEYWORDS_QUERY = re.compile(
    r"\b(UPDATE|INSERT|DELETE|DROP|ALTER|CREATE|ATTACH|DETACH|REPLACE|VACUUM|PRAGMA)\b",
    re.IGNORECASE,
)

def validate_sql_query(sql_code: str) -> str:
    """Validate and normalize a single non-destructive SELECT statement for dataset query.

    Returns the sanitized SQL or raises ValueError if invalid.
    """
    if not sql_code:
        raise ValueError("Empty SQL code")
    statements = [segment.strip() for segment in sql_code.strip().split(';') if segment.strip()]
    if len(statements) != 1:
        raise ValueError("Only a single SQL statement is allowed")
    stmt = statements[0]
    if not re.match(r"^\s*SELECT\b", stmt, re.IGNORECASE):
        raise ValueError("Only SELECT statements are permitted")
    if SQL_FORBIDDEN_KEYWORDS_QUERY.search(stmt):
        raise ValueError("Destructive or schema-altering keywords detected")
    return stmt


def _query_information_database(sql_code:str, db_path:str)->List[Dict[str, Any]]:
    
    """
    Execute sql command on the information database.

    Args:
        sql_code(str): Validated SELECT statement generated by the SQL helper agent.
        db_path(str): The path of the information database. 

    Returns:
        A list containing the query results.
    """

    if len(sql_code) == 0:
        return []
    db_path = Path(db_path)
    parent_path = db_path.parent

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(sql_code)
        records = cursor.fetchall() 
        results  = []
        for record in records:
            item = {key: record[key] for key in record.keys()}
            # resolve relative path -> absolute; handle both 'path' and 'Path'
            path_key = next((k for k in item if k.lower() == "path"), None)
            if path_key and item[path_key]:
                item[path_key] = str(parent_path / item[path_key])
            results.append(item)

        return results


def _read_user_structure(
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


def _query_compounds(
    selection: Union[dict,int,str,List[Union[str,Tuple]]],
    db_path: str,
    #exclusive_elements: Union[str, List[str]] = None,
    limit: Optional[int] = None,
    custom_args: Dict[str, Any] = {},
) -> Dict[str, Any]:
    """Query an ASE database for structures using flexible selectors and optional filters.

    Overview:
        Wraps `ase.db.connect(...).select(...)` and returns a compact summary of matching rows.
        The database path is resolved from `db_path`.

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
        
        db_path (str):
            Path to the ASE database.
        
        limit (int | None):
            Maximum number of rows to return (applied during ASE selection).

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
    path = db_path
    logging.info(f"Querying ASE database at {path} with selection: {selection}")
    seen_ids: List[int] = []
    formulas: set[str] = set()
    try:
        with connect(path) as db:
            for row in db.select(selection, limit=limit, **custom_args):
                seen_ids.append(row.id)
                formula = row.get("formula")
                if formula:
                    formulas.add(formula)
            
            # Return brief summary: count, unique formulas, and first few ids as samples
            sample_ids = seen_ids[:10]  # Only return first 10 ids as examples
            
            return QueryResult(
                query=str(selection),
                count=len(seen_ids),
                unique_formulas=sorted(formulas),
                sample_ids=sample_ids
            )
    except Exception as e:
        logging.error("Error querying database: %s", e)
        return QueryResult(
            query=str(selection),
            count=0,
            unique_formulas=[],
            sample_ids=[]
        )


class ExportResult(TypedDict):
    """Result structure for export_entries"""
    output_file: Path
    metadata_file: Path
    counts: Dict[str, int]

def _export_entries(
    ids: Optional[List[int]] = None,
    db_path: str = "",
    fmt: Literal["extxyz", "cif", "traj"] = "extxyz",
    mode: Literal["ids", "all", "random", "selection"] = "ids",
    sample_size: Optional[int] = None,
    random_seed: Optional[int] = None,
    selection: Optional[Union[dict, int, str, List[Union[str, Tuple]]]] = None,
    limit: Optional[int] = None,
    custom_args: Dict[str, Any] = {},
) -> Dict[str, Any]:
    """Export ASE database entries with flexible selection strategies.

    Args:
        ids: Explicit entry ids (required when ``mode='ids'``).
        db_path: Absolute path to the ASE database file.
        fmt: Output structure format.
        mode: Selection strategy: ``'ids'``, ``'all'``, ``'random'``, or ``'selection'``.
        sample_size: Number of entries to sample when ``mode='random'``.
        random_seed: Optional seed to make random sampling reproducible.
        selection: Selection criteria (required when ``mode='selection'``). 
            Same format as query_compounds: int, str, list[str], list[tuple], or dict.
        limit: Maximum number of entries to export when ``mode='selection'``.
        custom_args: Additional arguments forwarded to ASE db.select() when ``mode='selection'``.

    Returns:
        ExportResult with paths to the structure bundle, metadata JSON, and aggregate counts.
    """

    if mode not in {"ids", "all", "random", "selection"}:
        raise ValueError("mode must be one of: 'ids', 'all', 'random', 'selection'")
    if mode == "ids" and not ids:
        raise ValueError("Provide at least one id when mode='ids'")
    if mode == "random" and (sample_size is None or sample_size <= 0):
        raise ValueError("sample_size must be a positive integer when mode='random'")
    if mode == "selection" and selection is None:
        raise ValueError("Provide selection criteria when mode='selection'")

    path = db_path
    work_path = Path(generate_work_path())
    work_path = work_path.expanduser().resolve()
    work_path.mkdir(parents=True, exist_ok=True)

    atoms_collection: List[Any] = []
    formulas: set[str] = set()
    total_exported = 0

    metadata_path = work_path / "exported_metadata.json"
    combined_path = work_path / f"exported_structures.{fmt}"

    try:
        with metadata_path.open("w", encoding="utf-8") as meta_fp:
            with connect(path) as db:
                target_ids: Optional[List[int]] = None

                if mode == "ids":
                    target_ids = list(ids or [])
                elif mode == "random":
                    available_ids = [row.id for row in db.select()]
                    if not available_ids:
                        target_ids = []
                    else:
                        sample_n = min(sample_size or 0, len(available_ids))
                        rng = random.Random(random_seed)
                        target_ids = rng.sample(available_ids, sample_n)
                # mode == "all" or "selection" stream rows directly

                if mode == "all":
                    row_iter = db.select()
                elif mode == "selection":
                    row_iter = db.select(selection, limit=limit, **custom_args)
                else:
                    row_iter = (
                        db.get(id=entry_id)
                        for entry_id in (target_ids or [])
                    )

                for row in row_iter:
                    if row is None:
                        continue
                    atoms = row.toatoms()
                    atoms_collection.append(atoms)
                    formula = row.get("formula") or atoms.get_chemical_formula(empirical=True)
                    if formula:
                        formulas.add(formula)
                    record = {
                        "id": row.id,
                        "name": row.get("name"),
                        "formula": row.get("formula"),
                        "tags": row.get("tags"),
                    }
                    meta_fp.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                    total_exported += 1

        if total_exported == 0:
            metadata_path.unlink(missing_ok=True)
            return ExportResult(
                output_file=Path(""),
                metadata_file=Path(""),
                counts={"total_exported": 0, "unique_formulas": 0},
            )

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



_script_dir = Path(__file__).parent
load_dotenv(_script_dir / ".env", override=True)

DATABASE_SERVER_WORK_PATH = "/tmp/database_server"
INFO_DB_PATH = ""
 
info_db_path = os.environ.get("INFO_DB_PATH", INFO_DB_PATH)
 
 
def create_workpath(work_path=None):
    """
    Create the working directory for DataBase agent, and change the current working directory to it.
    
    Args:
        work_path (str, optional): The path to the working directory. If None, a default path will be used.
    
    Returns:
        str: The path to the working directory.
    """
    if work_path is None:
        work_path = os.environ.get("DATABASE_SERVER_WORK_PATH", "/tmp/database_server") + f"/{time.strftime('%Y%m%d%H%M%S')}"
    os.makedirs(work_path, exist_ok=True)
    os.chdir(work_path)
    print(f"Changed working directory to: {work_path}")
    return work_path    

def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Database server CLI")
    
    parser.add_argument(
        "--transport",
        type=str,
        default="sse",
        choices=["sse", "streamable-http"],
        help="Transport protocol to use (default: sse), choices: sse, streamable-http"
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
mcp = FastMCP(
        "DatabaseServer",
        host=args.host,
        port=args.port
        )
    
@mcp.tool()
def save_extxyz_to_db(extxyz_path:str):
    """
        Convert an extxyz file to an ASE db dataset and save its information to the information database.

        Args:
            extxyz_path (str): The path to the extxyz file.
        
        """
    root_dir = Path(info_db_path)
    user_data_dir = root_dir.parent / "user_data"
    if not user_data_dir.is_dir():
        user_data_dir.mkdir()
    
    db_name = f"{time.strftime('%Y%m%d%H%M%S')}.db"
    db_file_path = user_data_dir / db_name
    db_path_info = "user_data" + "/" + db_name
    _save_extxyz_to_db(extxyz_path, str(info_db_path), str(db_file_path), db_path_info)


@mcp.tool()
def validate_sql_code_query(sql_code: str) -> Dict[str, Any]:
    """Validate a SELECT statement before execution.

    Returns dict with normalized SQL (if valid) plus status/error info.
    """
    try:
        normalized = validate_sql_query(sql_code)
        return {"valid": True, "sql": normalized, "error": None}
    except ValueError as exc:
        return {"valid": False, "sql": "", "error": str(exc)}

         
@mcp.tool()
def query_information_database(sql_code: str) -> Dict[str, Any]:
    """Execute a SELECT statement on the information database.

    The SQL agent generates the query targeting the nodes/datasets/dataset_elements schema.
    All columns selected are returned; 'path' values are resolved to absolute paths.

    Args:
        sql_code (str): Validated single SELECT statement.

    Returns:
        Dict[str, Any]:
            - ``query`` (str): The SQL string that was executed.
            - ``count`` (int): Number of rows returned.
            - ``datasets`` (List[Dict]): One dict per row; columns depend on the SELECT.
              Always includes ``path`` resolved to absolute when present.
    """
    query_result = _query_information_database(sql_code, info_db_path)
    return {
        "query": sql_code.strip(),
        "count": len(query_result),
        "datasets": query_result,
    }
    
@mcp.tool()
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
    return _read_user_structure(
            structures=structures,
        )
    
@mcp.tool()
def query_compounds(
        selection: Union[dict,int,str,List[Union[str,Tuple]]],
        db_path: str,
        #exclusive_elements: Union[str, List[str]] = None,
        limit: Optional[int] = None,
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
                         'energy' # Note: this would select any entry with 'energy' key present. 
                • comparisons: 'key=value', 'key!=value', 'key<value', 'key<=value',
                  'key>value', 'key>=value'
                • combined: 'formula=Si32,pbc=True,energy<-1.0' or 'Si,O'
            - list[str]: list of string expressions, e.g. `['formula=Si32', 'pbc=True']`.
            - list[tuple]: list of `(key, op, value)` tuples, e.g. `[("energy", "<", -1.0)]`.
        
        limit (int | None):
            Maximum number of rows to return (applied during ASE selection).

        db_path (str):
            Path to a dataset (an ASE database file).

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
            - count (int): Total number of entries matching the selection.
            - unique_formulas (List[str]): Unique empirical formulas found.
            - sample_ids (List[int]): First few entry ids as examples (max 10).

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
    return _query_compounds(
            selection=selection,
            db_path=db_path,
            #exclusive_elements=exclusive_elements,
            limit=limit,
            custom_args=custom_args,
        )
        
@mcp.tool()        
def export_entries(
    ids: Optional[List[int]] = None,
    db_path: str = "",
    fmt: Literal["extxyz", "cif", "traj"] = "extxyz",
    mode: Literal["ids", "all", "random", "selection"] = "ids",
    sample_size: Optional[int] = None,
    random_seed: Optional[int] = None,
    selection: Optional[Union[dict, int, str, List[Union[str, Tuple]]]] = None,
    limit: Optional[int] = None,
    custom_args: Dict[str, Any] = {},
    ) -> Dict[str, Any]:
    """Export ASE entries by ids, selection criteria, entire dataset, or random sample.

    Args:
        ids (List[int] | None): Specific row identifiers when ``mode='ids'``.
        db_path (str): Absolute path to the ASE database file.
        fmt (Literal["extxyz", "cif", "traj"]): Output structure format.
        mode (Literal["ids", "all", "random", "selection"]): Selection strategy.
        sample_size (int | None): Number of entries to sample when ``mode='random'``.
        random_seed (int | None): Optional seed for deterministic sampling.
        selection: Selection criteria when ``mode='selection'``. Same format as query_compounds:
            int, str, list[str], list[tuple], or dict.
        limit (int | None): Maximum entries to export when ``mode='selection'``.
        custom_args (Dict[str, Any]): Additional arguments for ASE db.select() when ``mode='selection'``.

    Returns:
        Dict[str, Any]: Paths to the exported structure file and metadata JSON plus summary counts.
        
    Examples:
        # Export by selection (similar to query_compounds)
        >>> export_entries(db_path="path/to/db.db", mode="selection", selection="Si,energy<-1.0")
        >>> export_entries(db_path="path/to/db.db", mode="selection", selection="formula=Si32", limit=100)
    """
    return _export_entries(
        ids=ids,
        db_path=db_path,
        fmt=fmt,
        mode=mode,
        sample_size=sample_size,
        random_seed=random_seed,
        selection=selection,
        limit=limit,
        custom_args=custom_args,
    )

if __name__ == "__main__":
    create_workpath()
    mcp.run(transport=args.transport)