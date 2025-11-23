import argparse
import os
import argparse
from typing import Optional, Union, Literal, Dict, Any, List, Tuple
from pathlib import Path
import json
import time
import logging

from google.adk.models.lite_llm import LiteLlm
from mcp.server.fastmcp import FastMCP
from database import (
    read_user_structure as _read_user_structure,
    query_compounds as _query_compounds,
    export_entries as _export_entries,
    )

from database import get_sql_codes_from_llm as _get_sql_codes_from_llm
from database import query_information_database as _query_information_database

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    filename='agent.log', 
    filemode='w'         
)

ENVS = {
    "DATABASE_SERVER_WORK_PATH": "/tmp/database_server",
    "ASE_DB_PATH": "",
}
 
def set_envs():
    envjson_file = os.path.expanduser("~/.database_server/env.json")
    if os.path.isfile(envjson_file):
        envjson = json.load(open(envjson_file, "r"))
    else:
        envjson = {}
    update_envjson = False    
    for key, value in ENVS.items():
        if key not in envjson:
            envjson[key] = value
            update_envjson = True
        
    for key, value in envjson.items():
        os.environ[key] = str(value)
    
    if update_envjson:
        os.makedirs(os.path.dirname(envjson_file), exist_ok=True)
        json.dump(
            envjson,
            open(envjson_file, "w"),
            indent=4
        )
    return envjson

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
    cwd = os.getcwd()
    os.chdir(work_path)
    print(f"Changed working directory to: {work_path}")
    # write the environment variables to a file
    json.dump({
        k: os.environ.get(k) for k in ENVS.keys()
    }.update({"DATABASE_SERVER_START_PATH": cwd}), 
        open("env.json", "w"), indent=4)
    
    return work_path    

def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Database server CLI")
    
    parser.add_argument(
        "--transport",
        type=str,
        default=None,
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

def main():
    args = parse_args()  
    mcp = FastMCP(
            "DatabaseServer",
            host=args.host,
            port=args.port)
    
    env_file = os.path.expanduser("~/.database_agent/env.json")
    if os.path.isfile(env_file):
        env = json.load(open(env_file, "r"))
    else:
        env = {}
    model_name = env.get("LLM_MODEL", os.environ.get("LLM_MODEL", ""))
    model_api_key = env.get("LLM_API_KEY", os.environ.get("LLM_API_KEY", ""))
    model_base_url = env.get("LLM_BASE_URL", os.environ.get("LLM_BASE_URL", ""))
    info_db_path = env.get("INFO_DB_PATH", os.environ.get("INFO_DB_PATH", ""))
    
    # logging.info()
    # logging.info(f"info_db_path = {info_db_path}")

    llm = LiteLlm( 
        model    = model_name,
        base_url = model_base_url,
        api_key  = model_api_key
    )


    @mcp.tool()
    async def get_sql_codes_from_llm(user_prompts:str) -> str: 
        """
        Generate sql codes from the user's inputs. If there is no sql code block in the response, return an empty string.
        Args:
            user_prompts (str): The input prompts of the user.
        Returns:
            The sql code. If there is no sql code block in the response, return an empty string.
        """
        return await _get_sql_codes_from_llm(llm, user_prompts)
    

    @mcp.tool()
    def query_information_database(sql_code:str)->Tuple[str, Dict[str, Any]]:
        """
        Execute sql command on the information database. The function return a tuple of two elements:
        - The descriptive string of the query result in markdown format. It can be represented to the user.
        - The query result in a structured format.

        Args:
            sql_code(str): The sql code returned by `get_sql_codes_from_llm`.

        Returns:
            str: The descriptive string of the query result in a markdown table format. You can directly return it to the user.

            QueryResult:
                - query (str): Echo of the sql code (stringified).
                - count (int): Number of rows returned.
                - ids (List[int]): Unique row ids.
                - formulas (List[str]): Unique empirical formulas (if available).
                - results (List[Dict[str, Any]]): One dict per row with keys: { 'ID', 'Elements', 'Type', 'Fields', 'Entries', 'Source', 'Path' }. The meaning of these keys are
                
                    - ID: The global id of the dataset. An integer.
                    - Elements: All the elements in this dataset, seperated by hyphen. A string, e.g. Al-Si-Fe.
                    - Type: The system type of this dataset, such as Cluster, Bluk, Surface, Interface and so on. A string.
                    - Fields: The related field of this dataset, such as Alloy, Catalysis, Semi Conductor and so on. A string.
                    - Entries: The number of entries in this dataset. An integer.
                    - Source: Where does this dataset come from. A string.
                    - Path: The absolute path of the dataset file (*.db). A string. 
        """

        #logging.info(f"inside query_information_database info_db_path = {info_db_path}")
        query_result = _query_information_database(sql_code, info_db_path)
        

        # The summery string is a markdown table, which contains the 
        # ID, Elements, Type, Fields, Entries and Source, Path is omitted.
        summary_str = "# Summary of Query Results\n\n"
        summary_str += "You can specifically request entries by their IDs.\n\n"
        summary_str += "| ID | Elements | Type | Fields | Entries |\n"
        summary_str += "|----|----------|------|--------|---------|\n"
        for row in query_result["results"]:
            summary_str += f"| {row['ID']} | {row['Elements']} | {row['Type']} | {row['Fields']} | {row['Entries']} |\n"
        return summary_str, query_result
         

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
        ids: List[int],
        db_path: str,
        fmt: Literal["extxyz", "cif", "traj"] = "extxyz",
        
    ) -> Dict[str, Any]:
        """Export selected ASE database entries to a single structure file with summary stats."""
        return _export_entries(
            ids,
            db_path=db_path,
            fmt=fmt
        )
    set_envs()
    create_workpath()
    mcp.run(transport="sse")

if __name__ == "__main__":
    main()