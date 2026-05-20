"""Lightweight FastAPI server that exposes agent graph data to the frontend.

Runs on port 8001 alongside the ADK backend (port 8000).

Endpoints
---------
GET /api/agent-graph/{session_id}
    Returns the JSON graph file for the session, or an empty graph if not found.
GET /api/workspace/files?path=<path>
    Serves any file from the workspace root (absolute or relative path).
    Returns 403 if the path escapes the workspace root.
GET /api/sessions/{session_id}/files
    Lists all files under the session's working directory.

The vite dev server proxies /api/* here and /run_sse + /apps/* to the ADK server.
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# Allow importing from agents/ and src/ when running as script
ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.MatCreator.workspace import get_session_workdir, get_workspace_root  # noqa: E402
from agents.MatCreator.agents.cancellation import (  # noqa: E402
    request_cancellation,
    is_cancellation_requested,
    get_cancellation_reason,
    clear_cancellation,
    request_step_cancellation,
)
from agents.MatCreator.agents.graph_logger import AgentGraphLogger  # noqa: E402

app = FastAPI(title="MatCreator Graph API", version="1.0.0")
APP_NAME = "MatCreator"
SESSION_DB_PATH = ROOT / "agents" / "MatCreator" / ".adk" / "session.db"
DEFAULT_ADMIN_USERS = {"admin"}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)


def _admin_users() -> set[str]:
    raw_value = os.environ.get("MATCREATOR_ADMIN_USERS")
    if raw_value is None:
        return DEFAULT_ADMIN_USERS.copy()

    return {
        item.strip()
        for item in raw_value.split(",")
        if item.strip()
    }


def _is_admin(user_id: str) -> bool:
    return user_id in _admin_users()


def _session_row_to_summary(row: sqlite3.Row) -> dict:
    return {
        "id": row["id"],
        "appName": row["app_name"],
        "userId": row["user_id"],
        "createTime": row["create_time"],
        "lastUpdateTime": row["update_time"],
    }


def _query_session_summaries(user_id: str | None = None) -> list[dict]:
    if not SESSION_DB_PATH.exists():
        return []

    where_clause = "WHERE app_name = ?"
    params: tuple[str, ...] = (APP_NAME,)
    if user_id is not None:
        where_clause += " AND user_id = ?"
        params = (APP_NAME, user_id)

    try:
        with sqlite3.connect(SESSION_DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                f"""
                SELECT app_name, user_id, id, create_time, update_time
                FROM sessions
                {where_clause}
                ORDER BY update_time DESC
                """,
                params,
            ).fetchall()
    except sqlite3.Error as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read sessions: {exc}")

    return [_session_row_to_summary(row) for row in rows]


def _load_json_field(raw_value: str | None, fallback):
    if not raw_value:
        return fallback
    try:
        return json.loads(raw_value)
    except json.JSONDecodeError:
        return fallback


def _load_agent_graph_data(session_id: str) -> dict:
    graph_path = get_workspace_root() / "agent_graphs" / f"{session_id}.json"
    if not graph_path.exists():
        return {}
    try:
        return json.loads(graph_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _safe_upload_filename(filename: str) -> str:
    cleaned = Path(filename or "upload").name.strip()
    if not cleaned or cleaned in {".", ".."}:
        cleaned = "upload"
    return "".join(ch if ch.isalnum() or ch in "._- " else "_" for ch in cleaned)


def _available_upload_path(upload_dir: Path, filename: str) -> Path:
    candidate = upload_dir / filename
    if not candidate.exists():
        return candidate

    stem = candidate.stem or "upload"
    suffix = candidate.suffix
    for i in range(1, 10000):
        next_candidate = upload_dir / f"{stem}-{i}{suffix}"
        if not next_candidate.exists():
            return next_candidate

    raise HTTPException(status_code=409, detail="Too many files with the same name")


@app.get("/api/session-access/{user_id}")
async def get_session_access(user_id: str) -> JSONResponse:
    return JSONResponse({"user_id": user_id, "is_admin": _is_admin(user_id)})


@app.get("/api/users/{user_id}/sessions")
async def list_user_sessions(user_id: str) -> JSONResponse:
    return JSONResponse(_query_session_summaries(user_id))


@app.get("/api/users/{user_id}/sessions/{session_id}")
async def get_user_session(user_id: str, session_id: str) -> JSONResponse:
    if not SESSION_DB_PATH.exists():
        raise HTTPException(status_code=404, detail="Session not found")

    try:
        with sqlite3.connect(SESSION_DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            session = conn.execute(
                """
                SELECT app_name, user_id, id, state, create_time, update_time
                FROM sessions
                WHERE app_name = ? AND user_id = ? AND id = ?
                """,
                (APP_NAME, user_id, session_id),
            ).fetchone()
            if session is None:
                raise HTTPException(status_code=404, detail="Session not found")

            event_rows = conn.execute(
                """
                SELECT event_data
                FROM events
                WHERE app_name = ? AND user_id = ? AND session_id = ?
                ORDER BY timestamp ASC
                """,
                (APP_NAME, user_id, session_id),
            ).fetchall()
    except sqlite3.Error as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read session: {exc}")

    summary = _session_row_to_summary(session)
    summary["state"] = _load_json_field(session["state"], {})
    events = [
        _load_json_field(row["event_data"], {})
        for row in event_rows
    ]
    # Return the canonical session history as-is so the frontend reflects only
    # what was actually persisted in the session DB.
    summary["events"] = events
    return JSONResponse(summary)


@app.get("/api/admin/sessions")
async def list_all_sessions(user_id: str = Query(..., description="Current signed-in user")) -> JSONResponse:
    if not _is_admin(user_id):
        raise HTTPException(status_code=403, detail="Admin access required")
    return JSONResponse(_query_session_summaries())


@app.get("/api/agent-graph/{session_id}")
async def get_agent_graph(session_id: str) -> JSONResponse:
    data = _load_agent_graph_data(session_id)
    if not data:
        return JSONResponse({"session_id": session_id, "nodes": {}, "edges": [], "updated_at": None})
    return JSONResponse(data)


@app.get("/api/workspace/files")
async def serve_workspace_file(path: str = Query(..., description="Absolute or workspace-relative file path")) -> FileResponse:
    ws_root = get_workspace_root().resolve()
    p = Path(path)
    resolved = p.resolve() if p.is_absolute() else (ws_root / p).resolve()
    if not resolved.is_relative_to(ws_root):
        raise HTTPException(status_code=403, detail="Access denied: path is outside workspace")
    if not resolved.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(resolved)


@app.get("/api/structure/view")
async def view_structure(path: str = Query(..., description="Absolute or workspace-relative structure file path")) -> JSONResponse:
    from io import StringIO

    try:
        from ase.io import read as ase_read
        from ase.io import write as ase_write
    except ImportError:
        raise HTTPException(status_code=500, detail="ASE is not installed")

    ws_root = get_workspace_root().resolve()
    p = Path(path)
    resolved = p.resolve() if p.is_absolute() else (ws_root / p).resolve()
    if not resolved.is_relative_to(ws_root):
        raise HTTPException(status_code=403, detail="Access denied: path is outside workspace")
    if not resolved.is_file():
        raise HTTPException(status_code=404, detail="File not found")

    try:
        atoms = ase_read(str(resolved))
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Cannot parse structure: {exc}")

    buf = StringIO()
    ase_write(buf, atoms, format="xyz")
    xyz_data = buf.getvalue()

    return JSONResponse({
        "xyz": xyz_data,
        "formula": atoms.get_chemical_formula(),
        "n_atoms": len(atoms),
        "periodic": bool(atoms.pbc.any()),
        "cell": atoms.cell.tolist() if atoms.pbc.any() else None,
    })


@app.get("/api/sessions/{session_id}/files")
async def list_session_files(session_id: str) -> JSONResponse:
    session_dir = get_session_workdir(session_id)
    if not session_dir.exists():
        return JSONResponse({"files": []})
    files = [
        {"name": f.name, "path": str(f), "size": f.stat().st_size}
        for f in sorted(session_dir.rglob("*"))
        if f.is_file()
    ]
    return JSONResponse({"files": files})


@app.post("/api/sessions/{session_id}/files")
async def upload_session_file(session_id: str, file: UploadFile = File(...)) -> JSONResponse:
    session_dir = get_session_workdir(session_id)
    upload_dir = session_dir / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    filename = _safe_upload_filename(file.filename or "")
    target = _available_upload_path(upload_dir, filename)

    size = 0
    try:
        with target.open("wb") as fh:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                fh.write(chunk)
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Failed to save upload: {exc}")
    finally:
        await file.close()

    return JSONResponse({
        "name": target.name,
        "path": str(target),
        "size": size,
    })


@app.delete("/api/sessions/{session_id}/files")
async def delete_session_file(
    session_id: str,
    path: str = Query(..., description="Absolute or session-relative file path"),
) -> JSONResponse:
    session_dir = get_session_workdir(session_id).resolve()
    p = Path(path)
    resolved = p.resolve() if p.is_absolute() else (session_dir / p).resolve()

    if not resolved.is_relative_to(session_dir):
        raise HTTPException(status_code=403, detail="Access denied: path is outside session")
    if not resolved.is_file():
        raise HTTPException(status_code=404, detail="File not found")

    try:
        resolved.unlink()
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Failed to delete file: {exc}")

    return JSONResponse({"deleted": True, "path": str(resolved)})


@app.post("/api/sessions/{session_id}/cancel")
async def cancel_session_execution(
    session_id: str,
    reason: str = Query(default="user_requested", description="Cancellation reason"),
) -> JSONResponse:
    """Request cancellation of any ongoing execution for this session.

    The agent runner checks this flag before each step (graceful) and
    periodically during a step (force). The flag is cleared automatically
    when the orchestrator routes back to the planner.
    """
    request_cancellation(session_id, reason)
    AgentGraphLogger(session_id).mark_running_nodes_cancelled(
        summary=f"Cancelled by user ({reason})"
    )
    return JSONResponse({
        "status": "ok",
        "session_id": session_id,
        "message": "Cancellation requested. The running step will stop at the next checkpoint.",
    })


@app.get("/api/sessions/{session_id}/cancel")
async def get_cancellation_status(session_id: str) -> JSONResponse:
    """Check whether a cancellation is currently pending for this session."""
    flagged = is_cancellation_requested(session_id)
    return JSONResponse({
        "session_id": session_id,
        "cancellation_requested": flagged,
        "reason": get_cancellation_reason(session_id) if flagged else None,
    })


@app.delete("/api/sessions/{session_id}/cancel")
async def clear_cancellation_flag(session_id: str) -> JSONResponse:
    """Manually clear a pending cancellation flag."""
    clear_cancellation(session_id)
    return JSONResponse({
        "status": "ok",
        "session_id": session_id,
        "message": "Cancellation flag cleared.",
    })


@app.post("/api/sessions/{session_id}/cancel-step/{step_number}")
async def cancel_individual_step(
    session_id: str,
    step_number: int,
    reason: str = Query(default="user_requested", description="Cancellation reason"),
) -> JSONResponse:
    """Cancel a specific running step without stopping the whole session.

    The step executor polls the per-step flag and exits at the next checkpoint.
    The graph node for that step is updated immediately so the frontend reflects
    the cancellation before the executor polls.
    """
    request_step_cancellation(session_id, step_number, reason)
    found = AgentGraphLogger(session_id).cancel_step_node_by_number(
        step_number, summary=f"Cancelled by user ({reason})"
    )
    return JSONResponse({
        "status": "ok",
        "session_id": session_id,
        "step_number": step_number,
        "graph_updated": found,
        "message": f"Step {step_number} cancellation requested.",
    })


# Serve built frontend in production
_dist = Path(__file__).parent / "vite-frontend" / "dist"
if _dist.exists():
    app.mount("/", StaticFiles(directory=str(_dist), html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)
