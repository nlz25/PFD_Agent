import argparse
import os
import argparse
from typing import Optional, Union, Literal, Dict, Any, List, Tuple
from pathlib import Path
import time
from dotenv import load_dotenv
import logging

from mcp.server.fastmcp import FastMCP
from database import (
    read_user_structure as _read_user_structure,
    query_compounds as _query_compounds,
    export_entries as _export_entries
    )

from database import query_information_database as _query_information_database
from database import save_extxyz_to_db as _save_extxyz_to_db
from database import validate_sql_query

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    filename='agent.log', 
    filemode='w'         
)

load_dotenv(os.path.expanduser(".env"), override=True)

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
    """Execute a SELECT statement on the information database and summarize the datasets.

    Args:
        sql_code (str): Validated single SELECT statement targeting ``dataset_info``.

    Returns:
        Dict[str, Any]:
            - ``query`` (str): Normalized SQL string that was executed.
            - ``count`` (int): Number of datasets returned by the query.
            - ``datasets`` (List[Dict[str, Any]]): Minimal per-dataset metadata with keys
              ``ID``, ``Elements``, ``Type``, ``Fields``, ``Entries``, and absolute ``Path``.
    """
    
    query_result = _query_information_database(sql_code, info_db_path)
    
    dataset_summaries: List = [
        {
            "ID": row["ID"],
            "Elements": row["Elements"],
            "Type": row["Type"],
            "Fields": row["Fields"],
            "Entries": row["Entries"],
            "Path": row["Path"],
        }
        for row in query_result#["results"]
    ]
    return {
        "query": sql_code.strip(),
        "count": len(dataset_summaries),
        "datasets": dataset_summaries,
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
        exclusive_elements: Union[str, List[str]] = None,
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
    return _query_compounds(
            selection=selection,
            db_path=db_path,
            exclusive_elements=exclusive_elements,
            limit=limit,
            custom_args=custom_args,
        )
        
@mcp.tool()        
def export_entries(
    ids: Optional[List[int]] = None,
    db_path: str = "",
    fmt: Literal["extxyz", "cif", "traj"] = "extxyz",
    mode: Literal["ids", "all", "random"] = "ids",
    sample_size: Optional[int] = None,
    random_seed: Optional[int] = None,
    ) -> Dict[str, Any]:
    """Export ASE entries by ids, the entire dataset, or a random sample.

    Args:
        ids (List[int] | None): Specific row identifiers when ``mode='ids'``.
        db_path (str): Absolute path to the ASE database file.
        fmt (Literal["extxyz", "cif", "traj"]): Output structure format.
        mode (Literal["ids", "all", "random"]): Selection strategy.
        sample_size (int | None): Number of entries to sample when ``mode='random'``.
        random_seed (int | None): Optional seed for deterministic sampling.

    Returns:
        Dict[str, Any]: Paths to the exported structure file and metadata JSON plus summary counts.
    """
    return _export_entries(
        ids=ids,
        db_path=db_path,
        fmt=fmt,
        mode=mode,
        sample_size=sample_size,
        random_seed=random_seed,
    )

if __name__ == "__main__":
    create_workpath()
    mcp.run(transport=args.transport)