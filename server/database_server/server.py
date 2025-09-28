import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, TypedDict

from ase.db import connect
from ase.io import write
from mcp.server.fastmcp import FastMCP

# Globals configured at runtime
DEFAULT_DB_PATH: Path = Path(os.environ.get("ASE_DB_PATH", "./database.db")).resolve()

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ASE database MCP server")
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH, help="Path to ase.db file")
    parser.add_argument("--host", default="0.0.0.0", help="Host for SSE transport")
    parser.add_argument("--port", type=int, default=50001, help="Port for SSE transport")
    parser.add_argument(
        "--transport",
        choices=["sse", "stdio"],
        default="sse",
        help="MCP transport (sse listens on host/port, stdio uses stdin/stdout)",
    )
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    return parser.parse_args()
    
args = parse_args()
mcp = FastMCP("AseDatabaseServer", host=args.host, port=args.port)
logger = logging.getLogger("ase_db_server")


def _resolve_db_path(db_path: Optional[Path]) -> Path:
    path = (db_path or DEFAULT_DB_PATH).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"ASE database not found at {path}")
    return path

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
                #if len(results) >= limit:
                #    break
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

if __name__ == "__main__":
    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(message)s")
    globals()["DEFAULT_DB_PATH"] = args.db_path.expanduser().resolve()
    logger.info("Using ASE database at %s", DEFAULT_DB_PATH)
    if args.transport == "sse":
        mcp.run(transport="sse")
    else:
        mcp.run(transport="stdio")