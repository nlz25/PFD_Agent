"""Lightweight FastAPI server that exposes agent graph data to the frontend.

Runs alongside the ADK backend.  Ports are resolved via
:func:`~matcreator.ports.get_web_port` and
:func:`~matcreator.ports.get_adk_port` at call time
(env vars > config.yaml > defaults).

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

import asyncio
import fcntl
import json
import os
import pty
import re
import shutil
import signal
import socket
import sqlite3
import struct
import subprocess
import sys
import termios
import threading
import time
from pathlib import Path
from typing import List
from urllib.parse import unquote

import httpx
import yaml

from dotenv import dotenv_values, load_dotenv
from dotenv import set_key as dotenv_set_key
from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
# Allow importing users_db from the web/ directory
ROOT = Path(__file__).parent.parent
_WEB_DIR = Path(__file__).parent
_MATCREATOR_MODE = os.environ.get("MATCREATOR_MODE", "local")
_SERVER_DATA_ROOT = Path(
    os.environ.get("MATCREATOR_DATA_ROOT", str(ROOT / "server-data"))
).expanduser()
_CONTROL_PLANE_HOME_ENV = Path(
    os.environ.get(
        "MATCREATOR_CONTROL_PLANE_HOME",
        str(_SERVER_DATA_ROOT / "control-plane" / ".matcreator"),
    )
).expanduser()
if _MATCREATOR_MODE == "server":
    os.environ.setdefault("MATCREATOR_HOME", str(_CONTROL_PLANE_HOME_ENV))
if str(_WEB_DIR) not in sys.path:
    sys.path.insert(0, str(_WEB_DIR))

import users_db  # noqa: E402

from matcreator.workspace import get_session_workdir, get_workspace_root, workspace_skills_dir  # noqa: E402
from matcreator.agents.cancellation import (  # noqa: E402
    request_cancellation,
    is_cancellation_requested,
    get_cancellation_reason,
    clear_cancellation,
    request_step_cancellation,
)
from matcreator.agents.graph_logger import AgentGraphLogger  # noqa: E402
from matcreator.agents.session_log import build_session_log_export  # noqa: E402
from matcreator.skill import ALL_SKILLS, PLANNING_SKILL_NAMES, refresh_skills, get_default_skill_names  # noqa: E402
from matcreator.config import load_config, save_config, get_disabled_skills  # noqa: E402
from matcreator.config import ENV_TO_YAML, YAML_TO_ENV, SENSITIVE_YAML_KEYS  # noqa: E402
from matcreator.constants import GRAPH_AGENT_MODEL  # noqa: E402
from matcreator.knowledge.query import _get_kg  # noqa: E402
from matcreator.knowledge.review import run_review_pipeline  # noqa: E402
from matcreator.ports import get_adk_port, get_local_adk_command, get_web_port, get_worker_base_port  # noqa: E402

app = FastAPI(title="MatCreator Graph API", version="1.0.0")
APP_NAME = "MatCreator"
_SERVER_HOST_DATA_ROOT = Path(
    os.environ.get("MATCREATOR_HOST_DATA_ROOT", str(_SERVER_DATA_ROOT))
).expanduser()
_CONTROL_PLANE_HOME = _CONTROL_PLANE_HOME_ENV
_LOCAL_MATCREATOR_HOME = Path("~/.matcreator").expanduser()
_MATCREATOR_HOME = _CONTROL_PLANE_HOME if _MATCREATOR_MODE == "server" else _LOCAL_MATCREATOR_HOME
SESSION_DB_PATH = _MATCREATOR_HOME / ".adk" / "session.db"
_ADK_DIR = _MATCREATOR_HOME / ".adk"
ENV_PATH = _MATCREATOR_HOME / ".env"
_USERS_DATA_ROOT = _SERVER_DATA_ROOT / "users"
_USERS_HOST_ROOT = _SERVER_HOST_DATA_ROOT / "users"
_WORKER_MATCREATOR_HOME = Path("/root/.matcreator")
_WORKER_WORKSPACE_ROOT = _WORKER_MATCREATOR_HOME / "workspace"
DEFAULT_ADMIN_USERS = {"admin"}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

if ENV_PATH.exists():
    load_dotenv(ENV_PATH, override=False)
SUMMARIES_PATH = ROOT / "agents" / "MatCreator" / ".adk" / "session_summaries.json"
_SENSITIVE_FIELDS = frozenset({"LLM_API_KEY", "BOHRIUM_PASSWORD"})
_ENV_FIELDS = [
    "LLM_MODEL", "LLM_API_KEY", "LLM_BASE_URL", "EMBEDDING_MODEL",
    "BOHRIUM_EMAIL", "BOHRIUM_PASSWORD", "BOHRIUM_PROJECT_ID",
    "BOHRIUM_VASP_IMAGE", "BOHRIUM_VASP_MACHINE",
    "BOHRIUM_DEEPMD_IMAGE", "BOHRIUM_DEEPMD_MACHINE", "DEEPMD_MODEL_PATH",
]

_adk_process: subprocess.Popen | None = None
_knowledge_review_lock = threading.Lock()
_knowledge_review_task: asyncio.Task | None = None
_knowledge_review_state = {
    "status": "idle",
    "trigger_session_id": None,
    "progress": {"completed": 0, "total": 0, "percent": 0},
    "results": [],
    "errors": [],
    "summary": "",
}
_LEGACY_ENV_ALIASES = {
    "LLM_API_KEY": "MINIMAX_API_KEY",
    "LLM_BASE_URL": "MINIMAX_API_BASE",
}


def _config_value_for_env_key(env_key: str) -> str:
    yaml_key = ENV_TO_YAML.get(env_key)
    if not yaml_key:
        return ""
    parts = yaml_key.split(".", 1)
    config = load_config()
    if len(parts) == 1:
        value = config.get(parts[0], "")
    else:
        value = config.get(parts[0], {}).get(parts[1], "")
    return "" if value is None else str(value)


def _dotenv_value(env_key: str) -> str:
    if not ENV_PATH.exists():
        return ""
    value = dotenv_values(ENV_PATH).get(env_key, "")
    return "" if value is None else str(value)


def _runtime_env_value(env_key: str) -> str:
    """Resolve a setting from the active runtime plus persisted UI settings."""
    mode = os.environ.get("MATCREATOR_MODE", "local")
    if mode == "local":
        value = (
            _config_value_for_env_key(env_key)
            or os.environ.get(env_key, "")
            or _dotenv_value(env_key)
        )
    else:
        value = _dotenv_value(env_key) or os.environ.get(env_key, "")
    if value:
        return value
    legacy_key = _LEGACY_ENV_ALIASES.get(env_key)
    if not legacy_key:
        return ""
    return _dotenv_value(legacy_key) or os.environ.get(legacy_key, "")

# ---------------------------------------------------------------------------
# Server-mode worker management
# ---------------------------------------------------------------------------
# In server mode each user gets a dedicated Docker container running the ADK
# API server.  The control plane (this process) proxies /run_sse and /apps/*
# to the correct worker and manages the container lifecycle.

# _ADK_LOCAL_PORT is resolved at call time via get_adk_port() (env > config.yaml > default).
_WORKER_IMAGE = os.environ.get("MATCREATOR_WORKER_IMAGE", "matcreator:latest")
_WORKER_NETWORK = os.environ.get("MATCREATOR_WORKER_NETWORK", "matcreator-net")
_WORKER_CONNECT_MODE = os.environ.get("MATCREATOR_WORKER_CONNECT_MODE", "network").lower()
_WORKER_BASE_PORT = get_worker_base_port()
_WORKER_IDLE_TIMEOUT_SECONDS = int(os.environ.get("MATCREATOR_WORKER_IDLE_TIMEOUT_SECONDS", "0"))
_WORKER_MEM_LIMIT = os.environ.get("MATCREATOR_WORKER_MEM_LIMIT", "")
_WORKER_CPUS = os.environ.get("MATCREATOR_WORKER_CPUS", "")
_WORKER_PIDS_LIMIT = os.environ.get("MATCREATOR_WORKER_PIDS_LIMIT", "")
_WORKSPACE_CLI_TIMEOUT_SECONDS = int(os.environ.get("MATCREATOR_WORKSPACE_CLI_TIMEOUT_SECONDS", "30"))
_WORKSPACE_CLI_OUTPUT_LIMIT = int(os.environ.get("MATCREATOR_WORKSPACE_CLI_OUTPUT_LIMIT", "20000"))

_worker_registry: dict[str, str] = {}   # user_id → base URL
_worker_ports: dict[str, int] = {}      # user_id → host port (host-port connect mode)
_worker_last_used: dict[str, float] = {}
_worker_registry_lock = threading.Lock()
_docker_client = None


def _safe_user_dir_name(user_id: str) -> str:
    """Return a path-safe user directory name, rejecting path traversal."""
    candidate = (user_id or "").strip()
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", candidate):
        raise ValueError(f"Invalid user_id for filesystem path: {user_id!r}")
    return candidate


def _user_matcreator_home(user_id: str, *, host: bool = False) -> Path:
    root = _USERS_HOST_ROOT if host else _USERS_DATA_ROOT
    return root / _safe_user_dir_name(user_id) / ".matcreator"


def _user_adk_dir(user_id: str) -> Path:
    return _user_matcreator_home(user_id) / ".adk"


def _user_workspace_root(user_id: str) -> Path:
    return _user_matcreator_home(user_id) / "workspace"


def _worker_target_url(user_id: str, port: int | None = None) -> str:
    if _WORKER_CONNECT_MODE == "host-port":
        if port is None:
            raise RuntimeError("host-port worker routing requires a host port")
        return f"http://127.0.0.1:{port}"
    return f"http://{_worker_container_name(user_id)}:{get_adk_port()}"


def _iter_session_db_paths(user_id: str | None = None):
    if _MATCREATOR_MODE != "server":
        if SESSION_DB_PATH.exists():
            yield user_id or "", SESSION_DB_PATH
        return

    if user_id:
        db_path = _user_adk_dir(user_id) / "session.db"
        if db_path.exists():
            yield user_id, db_path
        return

    if not _USERS_DATA_ROOT.exists():
        return
    for db_path in sorted(_USERS_DATA_ROOT.glob("*/.matcreator/.adk/session.db")):
        yield db_path.parents[2].name, db_path


def _load_session_state(session_id: str, user_id: str | None = None) -> tuple[str | None, dict]:
    for owner_id, db_path in _iter_session_db_paths(user_id):
        try:
            with sqlite3.connect(db_path) as conn:
                row = conn.execute(
                    "SELECT state FROM sessions WHERE app_name = ? AND id = ?",
                    (APP_NAME, session_id),
                ).fetchone()
        except sqlite3.Error:
            continue
        if row:
            return owner_id, _load_json_field(row[0], {})
    return None, {}


def _get_docker():
    global _docker_client
    if _docker_client is None:
        try:
            import docker as _docker
            _docker_client = _docker.from_env()
        except Exception as exc:
            raise RuntimeError(f"Docker unavailable: {exc}") from exc
    return _docker_client


def _worker_container_name(user_id: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_-]", "-", user_id)[:64]
    return f"matcreator-worker-{safe}"


def _next_free_port() -> int:
    used = set(_worker_ports.values())
    port = _WORKER_BASE_PORT
    while port in used:
        port += 1
    return port


def _worker_env_vars() -> dict[str, str]:
    """Credentials to forward into each worker container."""
    keys = [
        "LLM_MODEL", "LLM_API_KEY", "LLM_BASE_URL", "EMBEDDING_MODEL",
        "GRAPH_AGENT_MODEL", "REVIEW_AGENT_MODEL",
        "BOHRIUM_USERNAME", "BOHRIUM_PASSWORD", "BOHRIUM_PROJECT_ID",
        "BOHRIUM_VASP_IMAGE", "BOHRIUM_VASP_MACHINE",
        "BOHRIUM_DEEPMD_IMAGE", "BOHRIUM_DEEPMD_MACHINE", "DEEPMD_MODEL_PATH",
        "KDG_EMBED_MODEL", "HF_HUB_OFFLINE",
    ]
    return {k: v for k in keys if (v := _runtime_env_value(k))}


def ensure_worker_running(user_id: str) -> str:
    """Ensure the worker container for *user_id* is running.

    Returns the base URL the worker's ADK server is reachable on.
    Creates the container if it doesn't exist.
    """
    import docker as _docker

    with _worker_registry_lock:
        _safe_user_dir_name(user_id)
        name = _worker_container_name(user_id)
        dc = _get_docker()

        # If we already know the target, verify the container is still running.
        if user_id in _worker_registry:
            target = _worker_registry[user_id]
            try:
                c = dc.containers.get(name)
                if c.status != "running":
                    c.start()
                _worker_last_used[user_id] = time.time()
                return target
            except _docker.errors.NotFound:
                del _worker_registry[user_id]
                _worker_ports.pop(user_id, None)

        # Container might exist from a previous server start.
        try:
            c = dc.containers.get(name)
            port = None
            if _WORKER_CONNECT_MODE == "host-port":
                bindings = c.ports.get(f"{get_adk_port()}/tcp") or []
                if not bindings:
                    c.remove(force=True)
                    c = None
                else:
                    port = int(bindings[0]["HostPort"])
                    _worker_ports[user_id] = port
            if c is not None:
                target = _worker_target_url(user_id, port)
                _worker_registry[user_id] = target
                if c.status != "running":
                    c.start()
                _worker_last_used[user_id] = time.time()
                return target
        except _docker.errors.NotFound:
            pass

        # Create a fresh worker container.
        port = _next_free_port() if _WORKER_CONNECT_MODE == "host-port" else None
        user_home_container = _user_matcreator_home(user_id)
        user_home_host = _user_matcreator_home(user_id, host=True)
        user_home_container.mkdir(parents=True, exist_ok=True)

        env_vars = _worker_env_vars()
        env_vars["MATCREATOR_MODE"] = "server"
        env_vars["MATCREATOR_USER_ID"] = user_id

        adk_port = get_adk_port()
        run_kwargs: dict = dict(
            image=_WORKER_IMAGE,
            command=["matcreator", "api-server", "--host", "0.0.0.0", "--port", str(adk_port)],
            name=name,
            environment=env_vars,
            volumes={
                str(user_home_host): {
                    "bind": str(_WORKER_MATCREATOR_HOME),
                    "mode": "rw",
                },
            },
            detach=True,
            restart_policy={"Name": "unless-stopped"},
        )
        if port is not None:
            run_kwargs["ports"] = {f"{adk_port}/tcp": port}
        if _WORKER_NETWORK:
            run_kwargs["network"] = _WORKER_NETWORK
        if _WORKER_MEM_LIMIT:
            run_kwargs["mem_limit"] = _WORKER_MEM_LIMIT
        if _WORKER_CPUS:
            run_kwargs["nano_cpus"] = int(float(_WORKER_CPUS) * 1_000_000_000)
        if _WORKER_PIDS_LIMIT:
            run_kwargs["pids_limit"] = int(_WORKER_PIDS_LIMIT)

        dc.containers.run(**run_kwargs)
        target = _worker_target_url(user_id, port)
        _worker_registry[user_id] = target
        if port is not None:
            _worker_ports[user_id] = port
        _worker_last_used[user_id] = time.time()
        return target


def stop_worker(user_id: str) -> None:
    """Stop (but keep) the worker container for *user_id*."""
    try:
        import docker as _docker
        dc = _get_docker()
        dc.containers.get(_worker_container_name(user_id)).stop(timeout=10)
    except Exception as exc:
        logger.warning("Failed to stop worker for user %s: %s", user_id, exc)


def remove_worker(user_id: str) -> None:
    """Stop and remove the worker container and clean up the registry."""
    try:
        import docker as _docker
        dc = _get_docker()
        dc.containers.get(_worker_container_name(user_id)).remove(force=True)
    except Exception as exc:
        logger.warning("Failed to remove worker for user %s: %s", user_id, exc)
    _worker_registry.pop(user_id, None)
    _worker_ports.pop(user_id, None)
    _worker_last_used.pop(user_id, None)


def _list_workers() -> list[dict]:
    """Return status info for all known worker containers."""
    try:
        import docker as _docker
        dc = _get_docker()
        results = []
        for user_id, target in list(_worker_registry.items()):
            name = _worker_container_name(user_id)
            try:
                c = dc.containers.get(name)
                status = c.status
            except _docker.errors.NotFound:
                status = "missing"
            results.append({"user_id": user_id, "container": name,
                             "target": target, "port": _worker_ports.get(user_id),
                             "status": status, "last_used": _worker_last_used.get(user_id)})
        return results
    except Exception as exc:
        return [{"error": str(exc)}]


async def _idle_worker_reaper() -> None:
    """Stop idle workers when MATCREATOR_WORKER_IDLE_TIMEOUT_SECONDS is set."""
    while True:
        await asyncio.sleep(min(max(_WORKER_IDLE_TIMEOUT_SECONDS // 2, 60), 600))
        cutoff = time.time() - _WORKER_IDLE_TIMEOUT_SECONDS
        with _worker_registry_lock:
            idle_users = [
                user_id
                for user_id, last_used in _worker_last_used.items()
                if last_used < cutoff
            ]
        for user_id in idle_users:
            await asyncio.to_thread(stop_worker, user_id)


def _extract_user_id_from_adk_path(path: str) -> str:
    match = re.search(r"(?:^|/)users/([^/]+)(?:/|$)", path)
    return unquote(match.group(1)) if match else ""


async def _adk_target_url(request: Request, adk_path: str = "") -> str:
    """Return the ADK base URL to proxy to, starting a worker if needed."""
    if _MATCREATOR_MODE != "server":
        return f"http://127.0.0.1:{get_adk_port()}"

    # Determine user_id from query string, ADK URL path, or request body.
    user_id = request.query_params.get("user_id", "") or _extract_user_id_from_adk_path(adk_path)
    if not user_id:
        try:
            body = await request.body()
            if body:
                payload = json.loads(body)
                user_id = payload.get("user_id", "")
        except Exception:
            pass

    if not user_id:
        raise HTTPException(status_code=400,
                            detail="user_id required to route to worker in server mode")

    return await asyncio.to_thread(ensure_worker_running, user_id)



def _is_port_open(host: str = "127.0.0.1", port: int | None = None) -> bool:
    if port is None:
        port = get_adk_port()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


# ---------------------------------------------------------------------------
# ADK proxy routes — forward /run_sse, /apps/*, /list-apps to the right worker
# ---------------------------------------------------------------------------

async def _proxy_request(request: Request, target_url: str, path: str) -> Response:
    url = f"{target_url.rstrip('/')}/{path.lstrip('/')}"
    params = dict(request.query_params)
    headers = {k: v for k, v in request.headers.items()
               if k.lower() not in ("host", "content-length", "origin")}
    body = await request.body()
    async with httpx.AsyncClient(timeout=None) as client:
        resp = await client.request(
            method=request.method,
            url=url,
            params=params,
            headers=headers,
            content=body,
        )
    return Response(
        content=resp.content,
        status_code=resp.status_code,
        headers=dict(resp.headers),
        media_type=resp.headers.get("content-type"),
    )


async def _proxy_sse(request: Request, target_url: str, path: str):
    url = f"{target_url.rstrip('/')}/{path.lstrip('/')}"
    params = dict(request.query_params)
    headers = {k: v for k, v in request.headers.items()
               if k.lower() not in ("host", "content-length", "origin")}
    body = await request.body()

    async def stream():
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream(
                method=request.method,
                url=url,
                params=params,
                headers=headers,
                content=body,
            ) as resp:
                async for chunk in resp.aiter_bytes():
                    yield chunk

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.api_route("/run_sse", methods=["GET", "POST"])
async def proxy_run_sse(request: Request):
    target_url = await _adk_target_url(request, "/run_sse")
    return await _proxy_sse(request, target_url, "/run_sse")


@app.api_route("/apps/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy_apps(path: str, request: Request):
    adk_path = f"/apps/{path}"
    target_url = await _adk_target_url(request, adk_path)
    return await _proxy_request(request, target_url, adk_path)


@app.api_route("/list-apps", methods=["GET"])
async def proxy_list_apps(request: Request):
    target_url = await _adk_target_url(request, "/list-apps")
    return await _proxy_request(request, target_url, "/list-apps")


# ---------------------------------------------------------------------------
# Worker management API
# ---------------------------------------------------------------------------

@app.get("/api/workers")
async def list_workers(user_id: str = Query(...)) -> JSONResponse:
    if not _is_admin(user_id):
        raise HTTPException(status_code=403, detail="Admin access required")
    return JSONResponse(_list_workers())


@app.post("/api/workers/{worker_user_id}/start")
async def start_worker_api(worker_user_id: str, user_id: str = Query(...)) -> JSONResponse:
    if not _is_admin(user_id):
        raise HTTPException(status_code=403, detail="Admin access required")
    if _MATCREATOR_MODE != "server":
        raise HTTPException(status_code=400, detail="Worker management only available in server mode")
    target = await asyncio.to_thread(ensure_worker_running, worker_user_id)
    return JSONResponse({"user_id": worker_user_id, "target": target, "status": "running"})


@app.post("/api/workers/{worker_user_id}/stop")
async def stop_worker_api(worker_user_id: str, user_id: str = Query(...)) -> JSONResponse:
    if not _is_admin(user_id):
        raise HTTPException(status_code=403, detail="Admin access required")
    if _MATCREATOR_MODE != "server":
        raise HTTPException(status_code=400, detail="Worker management only available in server mode")
    await asyncio.to_thread(stop_worker, worker_user_id)
    return JSONResponse({"user_id": worker_user_id, "status": "stopped"})


@app.delete("/api/workers/{worker_user_id}")
async def remove_worker_api(worker_user_id: str, user_id: str = Query(...)) -> JSONResponse:
    if not _is_admin(user_id):
        raise HTTPException(status_code=403, detail="Admin access required")
    if _MATCREATOR_MODE != "server":
        raise HTTPException(status_code=400, detail="Worker management only available in server mode")
    await asyncio.to_thread(remove_worker, worker_user_id)
    return JSONResponse({"user_id": worker_user_id, "status": "removed"})


def _kill_port(port: int) -> None:
    try:
        result = subprocess.run(
            ["lsof", "-ti", f":{port}"],
            capture_output=True, text=True, timeout=5,
        )
        for pid_str in result.stdout.strip().splitlines():
            try:
                os.kill(int(pid_str.strip()), signal.SIGTERM)
            except (ProcessLookupError, ValueError):
                pass
    except Exception:
        pass


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
    admin_names = _admin_users()
    if user_id in admin_names:
        return True  # legacy path: user_id is a display name (pre-UUID)
    user = users_db.get_by_id(user_id)
    return user is not None and user["display_name"] in admin_names


def _session_row_to_summary(row: sqlite3.Row, summaries: dict[str, dict] | None = None) -> dict:
    result = {
        "id": row["id"],
        "appName": row["app_name"],
        "userId": row["user_id"],
        "createTime": row["create_time"],
        "lastUpdateTime": row["update_time"],
    }
    if summaries is not None:
        entry = summaries.get(row["id"], "")
        if isinstance(entry, dict):
            result["summary"] = entry.get("summary", "")
        else:
            result["summary"] = entry
    return result


def _query_session_summaries(user_id: str | None = None) -> list[dict]:
    if _MATCREATOR_MODE == "server":
        return _query_session_summaries_server(user_id)

    if not SESSION_DB_PATH.exists():
        return []

    where_clause = "WHERE app_name = ?"
    params: tuple[str, ...] = (APP_NAME,)

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

    summaries = _load_summaries()
    return [_session_row_to_summary(row, summaries) for row in rows]


# ---------------------------------------------------------------------------
# Session summaries (experimental)
# ---------------------------------------------------------------------------

import hashlib
import tempfile


def _load_summaries() -> dict[str, dict]:
    """Load session summaries from the JSON file.

    Returns dict of {session_id: {"summary": str, "content_hash": str}}.
    For backward compatibility, plain string values are also accepted.
    """
    if not SUMMARIES_PATH.exists():
        return {}
    try:
        data = json.loads(SUMMARIES_PATH.read_text(encoding="utf-8"))
        # Normalize: support both old format (str) and new format (dict)
        result = {}
        for k, v in data.items():
            if isinstance(v, str):
                result[k] = {"summary": v, "content_hash": ""}
            else:
                result[k] = v
        return result
    except (json.JSONDecodeError, OSError):
        return {}


def _save_summaries(data: dict[str, dict]) -> None:
    """Persist session summaries atomically via temp file + rename."""
    SUMMARIES_PATH.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        dir=str(SUMMARIES_PATH.parent), suffix=".tmp", prefix="summaries_"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, str(SUMMARIES_PATH))
    except Exception:
        # Clean up temp file on failure
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _get_session_summary(session_id: str) -> str:
    """Get summary text for a single session, or empty string."""
    entry = _load_summaries().get(session_id)
    if isinstance(entry, dict):
        return entry.get("summary", "")
    return ""


def _fetch_first_user_message(session_id: str) -> str:
    """Read the first user message text from the session DB."""
    if not SESSION_DB_PATH.exists():
        return ""
    try:
        with sqlite3.connect(SESSION_DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT event_data FROM events
                WHERE app_name = ? AND session_id = ?
                ORDER BY timestamp ASC
                """,
                (APP_NAME, session_id),
            ).fetchall()
    except sqlite3.Error:
        return ""

    for row in rows:
        event = _load_json_field(row["event_data"], {})
        if event.get("author") != "user":
            continue
        parts = event.get("content", {}).get("parts", [])
        text = " ".join(p.get("text", "") for p in parts if isinstance(p, dict) and p.get("text"))
        if text.strip():
            return text.strip()
    return ""


_SUMMARIZE_PROMPT = (
    "请用一句简洁的中文（不超过30个字）总结以下对话的核心内容。"
    "只输出总结本身，不要任何前缀、解释或标点符号。不要使用句号、逗号、感叹号等任何标点。\n\n"
    "对话内容：\n{text}"
)

_SUMMARIZE_PROMPT_EN = (
    "Summarize the following conversation in a concise English phrase (no more than 6 words). "
    "Output only the summary itself, no prefixes, no explanations, no punctuation whatsoever "
    "(no periods, commas, exclamation marks, colons, semicolons, or any other punctuation).\n\n"
    "Conversation:\n{text}"
)


def _is_primarily_english(text: str) -> bool:
    """Return True if the majority of non-whitespace characters are ASCII letters."""
    alpha = [c for c in text.strip()[:500] if c.isalpha()]
    if not alpha:
        return False
    return sum(1 for c in alpha if c.isascii()) / len(alpha) > 0.5


def _llm_config() -> tuple[str, str | None, str | None]:
    """Resolve LLM config from env vars, with .env fallback.

    Returns None for api_key/base_url when not explicitly set, so litellm
    can use its built-in provider detection (e.g. MINIMAX_API_KEY for minimax/).
    """
    env = dotenv_values(ENV_PATH) if ENV_PATH.exists() else {}
    model = os.environ.get("LLM_MODEL") or env.get("LLM_MODEL", "")
    api_key = os.environ.get("LLM_API_KEY") or env.get("LLM_API_KEY") or None
    base_url = os.environ.get("LLM_BASE_URL") or env.get("LLM_BASE_URL") or None
    return model, api_key, base_url


@app.post("/api/sessions/{session_id}/summarize")
async def summarize_session(session_id: str) -> JSONResponse:
    """Generate a one-sentence summary from the session's first user message."""
    # Fetch canonical first message from DB
    first_msg = _fetch_first_user_message(session_id)
    if not first_msg:
        return JSONResponse({"summary": ""})

    content_hash = hashlib.md5(first_msg.encode()).hexdigest()[:12]

    # Return cached summary if content hash matches
    summaries = _load_summaries()
    cached = summaries.get(session_id)
    if cached and isinstance(cached, dict) and cached.get("content_hash") == content_hash:
        return JSONResponse({"summary": cached["summary"]})

    # Call LLM to generate summary
    summary = ""
    try:
        from litellm import acompletion

        model, api_key, base_url = _llm_config()
        prompt_template = _SUMMARIZE_PROMPT_EN if _is_primarily_english(first_msg) else _SUMMARIZE_PROMPT
        prompt = prompt_template.format(text=first_msg[:2000])

        response = await acompletion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            api_key=api_key or None,
            base_url=base_url or None,
            temperature=0.3,
            max_tokens=100,
        )
        raw = response.choices[0].message.content or ""
        import unicodedata
        summary = ''.join(c for c in raw if unicodedata.category(c)[0] != 'P').strip()
    except Exception as exc:
        import logging
        logging.getLogger(__name__).warning("Session summary LLM call failed: %s", exc)
        # Fallback: use truncated first message
        summary = first_msg[:40] + ("…" if len(first_msg) > 40 else "")

    # Persist with content hash (skip caching empty results so it retries next time)
    if summary:
        summaries[session_id] = {"summary": summary, "content_hash": content_hash}
        _save_summaries(summaries)

    return JSONResponse({"summary": summary})


class UpdateSummaryBody(BaseModel):
    summary: str


@app.put("/api/sessions/{session_id}/summary")
async def update_session_summary(session_id: str, body: UpdateSummaryBody) -> JSONResponse:
    """Manually set, override, or clear the summary for a session."""
    text = body.summary.strip()

    summaries = _load_summaries()
    if not text:
        # Clear summary
        summaries.pop(session_id, None)
        _save_summaries(summaries)
        return JSONResponse({"summary": ""})

    existing = summaries.get(session_id, {})
    content_hash = existing.get("content_hash", "") if isinstance(existing, dict) else ""
    summaries[session_id] = {"summary": text, "content_hash": content_hash}
    _save_summaries(summaries)
    return JSONResponse({"summary": text})


def _query_session_summaries_server(user_id: str | None = None) -> list[dict]:
    """In server mode, aggregate sessions from all per-user session DBs."""
    results: list[dict] = []
    # Each user's DB is at users/<user_id>/.matcreator/.adk/session.db.
    for owner_id, db_path in _iter_session_db_paths(user_id):
        try:
            with sqlite3.connect(db_path) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    """
                    SELECT app_name, user_id, id, create_time, update_time
                    FROM sessions
                    WHERE app_name = ?
                    ORDER BY update_time DESC
                    """,
                    (APP_NAME,),
                ).fetchall()
            results.extend(_session_row_to_summary(r) for r in rows)
        except sqlite3.Error:
            continue

    results.sort(key=lambda r: r.get("lastUpdateTime") or "", reverse=True)
    return results


def _load_json_field(raw_value: str | None, fallback):
    if not raw_value:
        return fallback
    try:
        return json.loads(raw_value)
    except json.JSONDecodeError:
        return fallback


def _ase_read_structure(path: Path):
    from ase.io import read as ase_read

    name = path.name.lower()
    if name in {"poscar", "contcar"} or path.suffix.lower() == ".vasp":
        return ase_read(str(path), format="vasp")
    return ase_read(str(path))


def _load_agent_graph_data(session_id: str) -> dict:
    graph_paths: list[Path]
    if _MATCREATOR_MODE == "server":
        graph_paths = [
            _user_adk_dir(owner_id) / "agent_graphs" / f"{session_id}.json"
            for owner_id, _ in _iter_session_db_paths()
        ]
    else:
        graph_paths = [_ADK_DIR / "agent_graphs" / f"{session_id}.json"]

    for graph_path in graph_paths:
        if not graph_path.exists():
            continue
        try:
            return json.loads(graph_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _map_worker_path_to_control_plane(user_id: str, path_str: str) -> Path | None:
    raw = (path_str or "").strip()
    if not raw:
        return None
    if raw.startswith("~/"):
        raw = str(_WORKER_MATCREATOR_HOME.parent / raw[2:])
    candidate = Path(raw)
    workspace_root = _user_workspace_root(user_id)
    user_home = _user_matcreator_home(user_id)

    if not candidate.is_absolute():
        return (workspace_root / candidate).resolve()

    try:
        rel = candidate.relative_to(_WORKER_WORKSPACE_ROOT)
        return (workspace_root / rel).resolve()
    except ValueError:
        pass

    try:
        rel = candidate.relative_to(_WORKER_MATCREATOR_HOME)
        return (user_home / rel).resolve()
    except ValueError:
        return candidate.resolve()


def _get_workdir_for_session(session_id: str) -> Path:
    """Resolve workdir for a session, preferring the value stored in session state.

    Priority: state["workdir"] → state["custom_workdir"] → computed default.
    In server mode, worker-container paths are mapped to the user's host-mounted
    .matcreator tree and paths outside that user's workspace are rejected.
    """
    owner_id, state = _load_session_state(session_id)
    path_str = state.get("workdir") or state.get("custom_workdir")
    if path_str:
        if _MATCREATOR_MODE == "server" and owner_id:
            candidate = _map_worker_path_to_control_plane(owner_id, path_str)
            ws_root = _user_workspace_root(owner_id).resolve()
            if candidate and candidate.is_relative_to(ws_root):
                return candidate
        elif _MATCREATOR_MODE != "server":
            return Path(path_str).expanduser().resolve()

    if _MATCREATOR_MODE == "server" and owner_id:
        return _user_workspace_root(owner_id)
    if _MATCREATOR_MODE == "server":
        return _USERS_DATA_ROOT / "_unknown" / ".matcreator" / "workspace"

    # Fall back to config.yaml default_workdir before using WORKSPACE_ROOT
    cfg_workdir = (load_config().get("workspace") or {}).get("default_workdir") or ""
    if cfg_workdir:
        candidate = Path(cfg_workdir).expanduser().resolve()
        if _MATCREATOR_MODE == "server":
            if candidate.is_relative_to(get_workspace_root().resolve()):
                return candidate
        else:
            return candidate
    return get_session_workdir(session_id)


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


def _control_plane_path_to_worker(user_id: str, path: Path) -> str:
    if _MATCREATOR_MODE != "server":
        return str(path)
    user_home = _user_matcreator_home(user_id).resolve()
    try:
        rel = path.resolve().relative_to(user_home)
    except ValueError:
        return str(path)
    return str(_WORKER_MATCREATOR_HOME / rel)


def _resolve_readable_file_path(path: str, session_id: str = "") -> Path:
    if _MATCREATOR_MODE == "server":
        owner_id = None
        if session_id:
            owner_id, _ = _load_session_state(session_id)
        if owner_id:
            resolved = _map_worker_path_to_control_plane(owner_id, path)
            if resolved is None:
                raise HTTPException(status_code=404, detail="File not found")
            allowed_roots = [
                _get_workdir_for_session(session_id).resolve(),
                _user_workspace_root(owner_id).resolve(),
            ]
        else:
            resolved = Path(path).expanduser().resolve()
            allowed_roots = [_USERS_DATA_ROOT.resolve()]
        if not any(resolved.is_relative_to(root) for root in allowed_roots):
            raise HTTPException(status_code=403, detail="Access denied: path is outside workspace")
        return resolved

    ws_root = get_workspace_root().resolve()
    p = Path(path)
    resolved = p.resolve() if p.is_absolute() else (ws_root / p).resolve()
    allowed = ws_root
    if session_id:
        allowed = _get_workdir_for_session(session_id).resolve()
    if not resolved.is_relative_to(allowed) and not resolved.is_relative_to(ws_root):
        raise HTTPException(status_code=403, detail="Access denied: path is outside workspace")
    return resolved


def _normalize_cli_cwd(cwd: str) -> Path:
    raw = (cwd or ".").strip()
    if raw in {"", "~"}:
        raw = "."
    if raw.startswith("~/"):
        raw = raw[2:]
    path = Path(raw)
    if path.is_absolute():
        raise HTTPException(status_code=403, detail="CLI cwd must be workspace-relative")

    root = Path("/")
    normalized = (root / path).resolve().relative_to(root)
    return normalized


def _cli_script(command: str) -> str:
    return (
        "exec 2>&1\n"
        "trap 's=$?; printf \"\\n__MATCREATOR_CWD__%s\\n__MATCREATOR_STATUS__%s\\n\" \"$PWD\" \"$s\"' EXIT\n"
        f"{command}\n"
    )


def _parse_cli_output(raw_output: str, workspace_root: Path | str) -> dict:
    cwd_marker = "\n__MATCREATOR_CWD__"
    status_marker = "\n__MATCREATOR_STATUS__"
    output = raw_output
    exit_code = 0
    next_cwd = "."

    cwd_idx = raw_output.rfind(cwd_marker)
    status_idx = raw_output.rfind(status_marker)
    if cwd_idx >= 0 and status_idx > cwd_idx:
        output = raw_output[:cwd_idx]
        cwd_abs = raw_output[cwd_idx + len(cwd_marker):status_idx].strip()
        status_raw = raw_output[status_idx + len(status_marker):].strip().splitlines()[0:1]
        if status_raw:
            try:
                exit_code = int(status_raw[0])
            except ValueError:
                exit_code = 1
        try:
            rel = Path(cwd_abs).resolve().relative_to(Path(workspace_root).resolve())
            next_cwd = rel.as_posix() or "."
        except (OSError, ValueError):
            next_cwd = "."

    if len(output) > _WORKSPACE_CLI_OUTPUT_LIMIT:
        output = output[:_WORKSPACE_CLI_OUTPUT_LIMIT] + "\n... [truncated]"
    return {"output": output, "exit_code": exit_code, "cwd": next_cwd}


async def _run_local_workspace_cli(command: str, cwd: str) -> dict:
    rel_cwd = _normalize_cli_cwd(cwd)
    workspace = get_workspace_root().resolve()
    run_cwd = (workspace / rel_cwd).resolve()
    if not run_cwd.is_relative_to(workspace):
        raise HTTPException(status_code=403, detail="CLI cwd is outside workspace")
    run_cwd.mkdir(parents=True, exist_ok=True)

    proc = await asyncio.create_subprocess_exec(
        "bash",
        "-lc",
        _cli_script(command),
        cwd=str(run_cwd),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    try:
        stdout, _ = await asyncio.wait_for(
            proc.communicate(), timeout=_WORKSPACE_CLI_TIMEOUT_SECONDS
        )
    except asyncio.TimeoutError:
        proc.kill()
        stdout, _ = await proc.communicate()
        output = stdout.decode("utf-8", errors="replace")
        return {
            "output": output + f"\n[Timeout after {_WORKSPACE_CLI_TIMEOUT_SECONDS}s]",
            "exit_code": 124,
            "cwd": rel_cwd.as_posix() or ".",
        }
    return _parse_cli_output(stdout.decode("utf-8", errors="replace"), workspace)


def _run_worker_workspace_cli(user_id: str, command: str, cwd: str) -> dict:
    _safe_user_dir_name(user_id)
    rel_cwd = _normalize_cli_cwd(cwd)
    ensure_worker_running(user_id)

    dc = _get_docker()
    container = dc.containers.get(_worker_container_name(user_id))
    workdir = (_WORKER_WORKSPACE_ROOT / rel_cwd).as_posix()
    container.exec_run(["mkdir", "-p", workdir])

    exec_result = container.exec_run(
        [
            "timeout",
            str(_WORKSPACE_CLI_TIMEOUT_SECONDS),
            "bash",
            "-lc",
            _cli_script(command),
        ],
        workdir=workdir,
        stdout=True,
        stderr=True,
        demux=False,
    )
    raw = exec_result.output.decode("utf-8", errors="replace")
    if exec_result.exit_code == 124:
        return {
            "output": raw + f"\n[Timeout after {_WORKSPACE_CLI_TIMEOUT_SECONDS}s]",
            "exit_code": 124,
            "cwd": rel_cwd.as_posix() or ".",
        }
    return _parse_cli_output(raw, _WORKER_WORKSPACE_ROOT)


def _complete_workspace_paths(workspace_root: Path, cwd: str, token: str) -> dict:
    workspace = workspace_root.resolve()
    rel_cwd = _normalize_cli_cwd(cwd)
    raw_token = (token or "").strip()
    if raw_token.startswith("~/"):
        raw_token = raw_token[2:]
    if Path(raw_token).is_absolute():
        raise HTTPException(status_code=403, detail="Completion path must be workspace-relative")

    if "/" in raw_token:
        parent_raw, prefix = raw_token.rsplit("/", 1)
        parent_rel = _normalize_cli_cwd(parent_raw or ".")
        replacement_prefix = "" if parent_raw in {"", "."} else f"{parent_raw}/"
    else:
        parent_rel = rel_cwd
        prefix = raw_token
        replacement_prefix = ""

    base_dir = (workspace / parent_rel).resolve()
    if not base_dir.is_relative_to(workspace):
        raise HTTPException(status_code=403, detail="Completion path is outside workspace")
    if not base_dir.is_dir():
        return {"matches": []}

    matches = []
    try:
        entries = sorted(base_dir.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Failed to list workspace path: {exc}")

    for entry in entries:
        if not entry.name.startswith(prefix):
            continue
        suffix = "/" if entry.is_dir() else ""
        matches.append({
            "name": entry.name + suffix,
            "replacement": replacement_prefix + entry.name + suffix,
            "type": "dir" if entry.is_dir() else "file",
        })

    return {"matches": matches[:200]}


def _set_pty_size(fd: int, rows: int, cols: int) -> None:
    rows = max(1, min(int(rows or 24), 200))
    cols = max(1, min(int(cols or 80), 400))
    fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))


async def _send_terminal_output(websocket: WebSocket, data: bytes) -> None:
    await websocket.send_text(json.dumps({
        "type": "output",
        "data": data.decode("utf-8", errors="replace"),
    }))


async def _local_terminal_session(websocket: WebSocket) -> None:
    workspace = get_workspace_root().resolve()
    workspace.mkdir(parents=True, exist_ok=True)

    master_fd, slave_fd = pty.openpty()
    _set_pty_size(master_fd, 24, 80)
    env = {
        **os.environ,
        "TERM": "xterm-256color",
        "COLORTERM": "truecolor",
        "MATCLAW_WORKSPACE": str(workspace),
        "MATCLAW_SESSION_DIR": str(workspace),
    }
    proc = subprocess.Popen(
        ["bash", "-l"],
        cwd=str(workspace),
        stdin=slave_fd,
        stdout=slave_fd,
        stderr=slave_fd,
        start_new_session=True,
        env=env,
    )
    os.close(slave_fd)
    loop = asyncio.get_running_loop()

    async def read_loop() -> None:
        while True:
            try:
                data = await loop.run_in_executor(None, os.read, master_fd, 4096)
            except OSError:
                break
            if not data:
                break
            try:
                await _send_terminal_output(websocket, data)
            except Exception:
                break

    reader = asyncio.create_task(read_loop())
    try:
        while True:
            message = json.loads(await websocket.receive_text())
            msg_type = message.get("type")
            if msg_type == "input":
                os.write(master_fd, str(message.get("data", "")).encode())
            elif msg_type == "resize":
                _set_pty_size(master_fd, int(message.get("rows", 24)), int(message.get("cols", 80)))
    except WebSocketDisconnect:
        pass
    finally:
        reader.cancel()
        try:
            os.close(master_fd)
        except OSError:
            pass
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass


def _docker_exec_socket(user_id: str):
    ensure_worker_running(user_id)
    dc = _get_docker()
    container = dc.containers.get(_worker_container_name(user_id))
    workspace = _WORKER_WORKSPACE_ROOT.as_posix()
    container.exec_run(["mkdir", "-p", workspace])
    exec_id = dc.api.exec_create(
        container.id,
        ["bash", "-l"],
        stdin=True,
        stdout=True,
        stderr=True,
        tty=True,
        workdir=workspace,
        environment={
            "TERM": "xterm-256color",
            "COLORTERM": "truecolor",
            "MATCLAW_WORKSPACE": workspace,
            "MATCLAW_SESSION_DIR": workspace,
        },
    )["Id"]
    sock = dc.api.exec_start(exec_id, tty=True, socket=True)
    raw_sock = getattr(sock, "_sock", sock)
    raw_sock.settimeout(None)
    return dc, exec_id, sock, raw_sock


async def _worker_terminal_session(websocket: WebSocket, user_id: str) -> None:
    _safe_user_dir_name(user_id)
    dc, exec_id, sock, raw_sock = await asyncio.to_thread(_docker_exec_socket, user_id)
    loop = asyncio.get_running_loop()

    async def read_loop() -> None:
        while True:
            try:
                data = await loop.run_in_executor(None, raw_sock.recv, 4096)
            except OSError:
                break
            if not data:
                break
            try:
                await _send_terminal_output(websocket, data)
            except Exception:
                break

    reader = asyncio.create_task(read_loop())
    try:
        while True:
            message = json.loads(await websocket.receive_text())
            msg_type = message.get("type")
            if msg_type == "input":
                await loop.run_in_executor(None, raw_sock.send, str(message.get("data", "")).encode())
            elif msg_type == "resize":
                rows = max(1, min(int(message.get("rows", 24)), 200))
                cols = max(1, min(int(message.get("cols", 80)), 400))
                await asyncio.to_thread(dc.api.exec_resize, exec_id, height=rows, width=cols)
    except WebSocketDisconnect:
        pass
    finally:
        reader.cancel()
        try:
            sock.close()
        except Exception:
            pass


@app.get("/api/health")
async def health_check():
    mode = os.environ.get("MATCREATOR_MODE", "local")
    return {"status": "ok", "mode": mode}
@app.on_event("startup")
async def _on_startup() -> None:
    users_db.init_db()
    if _MATCREATOR_MODE != "server":
        users_db.migrate_legacy_adk_sessions(SESSION_DB_PATH, APP_NAME)
    if _MATCREATOR_MODE == "server" and _WORKER_IDLE_TIMEOUT_SECONDS > 0:
        asyncio.create_task(_idle_worker_reaper())


class LoginBody(BaseModel):
    display_name: str
    password: str | None = None


class RegisterBody(BaseModel):
    display_name: str
    password: str


class SetPasswordBody(BaseModel):
    user_id: str
    old_password: str | None = None
    new_password: str


class LogoutBody(BaseModel):
    user_id: str


class WorkspaceCliBody(BaseModel):
    command: str
    cwd: str = "."
    user_id: str | None = None


class WorkspaceCompleteBody(BaseModel):
    token: str = ""
    cwd: str = "."
    user_id: str | None = None


class KnowledgeReviewBody(BaseModel):
    session_id: str


def _knowledge_review_snapshot() -> dict:
    with _knowledge_review_lock:
        return dict(_knowledge_review_state)


def _set_knowledge_review_state(**changes) -> None:
    with _knowledge_review_lock:
        _knowledge_review_state.update(changes)


def _review_model_config() -> tuple[str, str, str | None]:
    model = (
        _runtime_env_value("REVIEW_AGENT_MODEL")
        or _runtime_env_value("GRAPH_AGENT_MODEL")
        or _runtime_env_value("LLM_MODEL")
        or GRAPH_AGENT_MODEL
    )
    api_key = _runtime_env_value("LLM_API_KEY")
    base_url = _runtime_env_value("LLM_BASE_URL") or None
    if "/" in model:
        model = model.split("/", 1)[1]
    return model, api_key, base_url


async def _run_knowledge_review(session_id: str) -> None:
    try:
        model, api_key, base_url = _review_model_config()
        if not api_key:
            raise RuntimeError(
                "No review API key configured. Set LLM_API_KEY in Settings "
                "(stored in ~/.matcreator/config.yaml in local mode)."
            )
        if not model:
            raise RuntimeError(
                "No REVIEW_AGENT_MODEL, GRAPH_AGENT_MODEL, or LLM_MODEL configured."
            )

        def run_review() -> dict:
            graph = _get_kg()
            result = run_review_pipeline(
                graph,
                model=model,
                api_key=api_key,
                base_url=base_url,
                batch_size=20,
                strategy=os.environ.get("MATCREATOR_REVIEW_STRATEGY", "auto"),
                on_status=lambda phase, status: _set_knowledge_review_state(
                    **status,
                    phase=phase,
                    trigger_session_id=session_id,
                ),
            )
            return result

        result = await asyncio.to_thread(run_review)
        _set_knowledge_review_state(**result, trigger_session_id=session_id)
    except Exception as exc:
        _set_knowledge_review_state(
            status="failed",
            trigger_session_id=session_id,
            progress={"completed": 0, "total": 0, "percent": 0},
            results=[],
            errors=[str(exc)],
            summary="",
        )
    finally:
        global _knowledge_review_task
        _knowledge_review_task = None


@app.post("/api/knowledge-review/start")
async def start_knowledge_review(body: KnowledgeReviewBody) -> JSONResponse:
    global _knowledge_review_task
    if _knowledge_review_task is not None and not _knowledge_review_task.done():
        return JSONResponse(_knowledge_review_snapshot(), status_code=202)

    _set_knowledge_review_state(
        status="running",
        trigger_session_id=body.session_id,
        progress={"completed": 0, "total": 0, "percent": 0},
        results=[],
        errors=[],
        summary="",
    )
    _knowledge_review_task = asyncio.create_task(_run_knowledge_review(body.session_id))
    return JSONResponse(_knowledge_review_snapshot(), status_code=202)


@app.get("/api/knowledge-review/status")
async def get_knowledge_review_status() -> JSONResponse:
    return JSONResponse(_knowledge_review_snapshot())


@app.post("/api/auth/login")
async def auth_login(body: LoginBody) -> JSONResponse:
    name = body.display_name.strip()
    if not name:
        raise HTTPException(status_code=422, detail="display_name cannot be empty")

    # Special identity: "user" always allowed, no password required.
    if name == users_db.LEGACY_USER:
        if _MATCREATOR_MODE == "server":
            await asyncio.to_thread(ensure_worker_running, users_db.LEGACY_USER)
        return JSONResponse({
            "user_id": users_db.LEGACY_USER,
            "display_name": users_db.LEGACY_USER,
            "is_admin": _is_admin(users_db.LEGACY_USER),
        })

    user = users_db.get_by_display_name(name)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found. Please register first.")
    if user["password_hash"] is not None and not users_db.verify_password(user["password_hash"], body.password):
        raise HTTPException(status_code=401, detail="Invalid password.")

    if _MATCREATOR_MODE == "server":
        await asyncio.to_thread(ensure_worker_running, user["id"])

    return JSONResponse({
        "user_id": user["id"],
        "display_name": user["display_name"],
        "is_admin": _is_admin(user["id"]),
    })


@app.post("/api/auth/register")
async def auth_register(body: RegisterBody) -> JSONResponse:
    name = body.display_name.strip()
    if not name:
        raise HTTPException(status_code=422, detail="display_name cannot be empty")
    if name == users_db.LEGACY_USER:
        raise HTTPException(status_code=400, detail="'user' is a reserved username.")
    if not body.password:
        raise HTTPException(status_code=422, detail="Password is required for registration.")

    existing = users_db.get_by_display_name(name)
    if existing is not None:
        raise HTTPException(status_code=409, detail="Username already taken.")

    user = users_db.create_user(name, body.password)
    if _MATCREATOR_MODE == "server":
        await asyncio.to_thread(ensure_worker_running, user["id"])

    return JSONResponse({
        "user_id": user["id"],
        "display_name": user["display_name"],
        "is_admin": _is_admin(user["id"]),
    }, status_code=201)


@app.post("/api/auth/set-password")
async def auth_set_password(body: SetPasswordBody) -> JSONResponse:
    if body.user_id == users_db.LEGACY_USER:
        raise HTTPException(status_code=400, detail="Cannot set password for the 'user' account")
    user = users_db.get_by_id(body.user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    if not users_db.verify_password(user["password_hash"], body.old_password):
        raise HTTPException(status_code=401, detail="Current password is incorrect")
    users_db.set_password(body.user_id, body.new_password)
    return JSONResponse({"status": "ok"})


@app.post("/api/auth/logout")
async def auth_logout(body: LogoutBody) -> JSONResponse:
    """Log out the current user and stop their server-mode worker."""
    if _MATCREATOR_MODE == "server" and body.user_id:
        await asyncio.to_thread(stop_worker, body.user_id)
    return JSONResponse({"status": "ok"})


@app.get("/api/session-access/{user_id}")
async def get_session_access(user_id: str) -> JSONResponse:
    return JSONResponse({"user_id": user_id, "is_admin": _is_admin(user_id)})


@app.get("/api/users/{user_id}/sessions")
async def list_user_sessions(user_id: str) -> JSONResponse:
    return JSONResponse(_query_session_summaries(user_id))


@app.get("/api/users/{user_id}/sessions/{session_id}")
async def get_user_session(user_id: str, session_id: str) -> JSONResponse:
    session_db_path = next((db for _, db in _iter_session_db_paths(user_id)), None)
    if not session_db_path:
        raise HTTPException(status_code=404, detail="Session not found")

    try:
        with sqlite3.connect(session_db_path) as conn:
            conn.row_factory = sqlite3.Row
            if _MATCREATOR_MODE == "server":
                session = conn.execute(
                    """
                    SELECT app_name, user_id, id, state, create_time, update_time
                    FROM sessions
                    WHERE app_name = ? AND user_id = ? AND id = ?
                    """,
                    (APP_NAME, user_id, session_id),
                ).fetchone()
            else:
                session = conn.execute(
                    """
                    SELECT app_name, user_id, id, state, create_time, update_time
                    FROM sessions
                    WHERE app_name = ? AND id = ?
                    """,
                    (APP_NAME, session_id),
                ).fetchone()
            if session is None:
                raise HTTPException(status_code=404, detail="Session not found")

            if _MATCREATOR_MODE == "server":
                event_rows = conn.execute(
                    """
                    SELECT event_data
                    FROM events
                    WHERE app_name = ? AND user_id = ? AND session_id = ?
                    ORDER BY timestamp ASC
                    """,
                    (APP_NAME, user_id, session_id),
                ).fetchall()
            else:
                event_rows = conn.execute(
                    """
                    SELECT event_data
                    FROM events
                    WHERE app_name = ? AND session_id = ?
                    ORDER BY timestamp ASC
                    """,
                    (APP_NAME, session_id),
                ).fetchall()
    except sqlite3.Error as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read session: {exc}")

    summary = _session_row_to_summary(session)
    summary["summary"] = _get_session_summary(session_id)
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


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str) -> JSONResponse:
    """Delete a session and all associated metadata. Workspace files are not deleted."""
    # 1. Delete from session DB (events cascade-deleted via FK)
    deleted_owners: list[str] = []
    db_paths = list(_iter_session_db_paths()) if _MATCREATOR_MODE == "server" else [("", SESSION_DB_PATH)]
    for owner_id, db_path in db_paths:
        if not db_path.exists():
            continue
        try:
            with sqlite3.connect(db_path) as conn:
                conn.execute("PRAGMA foreign_keys = ON")
                cur = conn.execute(
                    "DELETE FROM sessions WHERE app_name = ? AND id = ?",
                    (APP_NAME, session_id),
                )
                if cur.rowcount:
                    deleted_owners.append(owner_id)
        except sqlite3.Error as exc:
            raise HTTPException(status_code=500, detail=f"Failed to delete session from DB: {exc}")

    # 2. Remove summary entry
    summaries = _load_summaries()
    if session_id in summaries:
        del summaries[session_id]
        _save_summaries(summaries)

    # 3. Delete session-scoped metadata artifacts (not workspace files)
    if _MATCREATOR_MODE == "server":
        targets = []
        for owner_id in deleted_owners:
            workspace = _user_workspace_root(owner_id)
            adk_dir = _user_adk_dir(owner_id)
            targets.extend([
                adk_dir / "agent_graphs" / f"{session_id}.json",
                workspace / "trajectories" / f"{session_id}.jsonl",
                workspace / "trajectories" / f"{session_id}_summary.json",
                workspace / "cancellation" / f"{session_id}.flag",
            ])
    else:
        workspace = get_workspace_root()
        targets = [
            _ADK_DIR / "agent_graphs" / f"{session_id}.json",
            workspace / "trajectories" / f"{session_id}.jsonl",
            workspace / "trajectories" / f"{session_id}_summary.json",
            workspace / "cancellation" / f"{session_id}.flag",
        ]
    for target in targets:
        try:
            if target.is_file():
                target.unlink()
        except OSError:
            pass  # best-effort cleanup

    return JSONResponse({"status": "ok", "deleted": session_id})


def _load_execution_graph(session_id: str) -> dict:
    """Read execution_graph from the SQLite session state."""
    for _, db_path in _iter_session_db_paths():
        try:
            with sqlite3.connect(db_path) as conn:
                conn.row_factory = sqlite3.Row
                row = conn.execute(
                    "SELECT state FROM sessions WHERE app_name = ? AND id = ?",
                    (APP_NAME, session_id),
                ).fetchone()
                if row is None:
                    continue
                state = _load_json_field(row["state"], {})
                raw = state.get("execution_graph")
                if isinstance(raw, str):
                    raw = _load_json_field(raw, None)
                if not isinstance(raw, dict):
                    return {"nodes": {}, "edges": []}
                return raw
        except sqlite3.Error:
            continue
    return {"nodes": {}, "edges": []}


def _load_session_log_export(session_id: str, user_id: str | None = None) -> dict:
    owner_id, state = _load_session_state(session_id, user_id)
    if not state:
        raise HTTPException(status_code=404, detail="Session not found")
    payload = build_session_log_export(session_id, state)
    payload["owner_id"] = owner_id
    return payload


@app.get("/api/execution-graph/{session_id}")
async def get_execution_graph(session_id: str) -> JSONResponse:
    """Return the execution graph (plan DAG) from session state for frontend visualization."""
    data = _load_execution_graph(session_id)
    return JSONResponse(data)


@app.get("/api/sessions/{session_id}/session-log")
async def download_session_log(
    session_id: str,
    user_id: str = Query(default="", description="Current user ID; required to scope server-mode sessions."),
) -> Response:
    payload = _load_session_log_export(session_id, user_id or None)
    content = json.dumps(payload, ensure_ascii=False, indent=2)
    response = Response(content=content, media_type="application/json")
    safe_session_id = re.sub(r"[^A-Za-z0-9_.-]", "_", session_id)
    response.headers["Content-Disposition"] = (
        f'attachment; filename="matcreator-session-log-{safe_session_id}.json"'
    )
    return response



@app.get("/api/agent-graph/{session_id}")
async def get_agent_graph(session_id: str) -> JSONResponse:
    data = _load_agent_graph_data(session_id)
    if not data:
        return JSONResponse({"session_id": session_id, "nodes": {}, "edges": [], "updated_at": None})
    return JSONResponse(data)


@app.post("/api/workspace/cli")
async def run_workspace_cli(body: WorkspaceCliBody) -> JSONResponse:
    command = body.command.strip()
    if not command:
        raise HTTPException(status_code=422, detail="command cannot be empty")

    if _MATCREATOR_MODE == "server":
        if not body.user_id:
            raise HTTPException(status_code=400, detail="user_id required in server mode")
        result = await asyncio.to_thread(
            _run_worker_workspace_cli,
            body.user_id,
            command,
            body.cwd,
        )
    else:
        result = await _run_local_workspace_cli(command, body.cwd)
    return JSONResponse(result)


@app.post("/api/workspace/complete")
async def complete_workspace_cli(body: WorkspaceCompleteBody) -> JSONResponse:
    if _MATCREATOR_MODE == "server":
        if not body.user_id:
            raise HTTPException(status_code=400, detail="user_id required in server mode")
        workspace = _user_workspace_root(body.user_id)
    else:
        workspace = get_workspace_root()
    workspace.mkdir(parents=True, exist_ok=True)
    return JSONResponse(_complete_workspace_paths(workspace, body.cwd, body.token))


@app.websocket("/api/workspace/terminal")
async def workspace_terminal(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        if _MATCREATOR_MODE == "server":
            user_id = websocket.query_params.get("user_id", "")
            if not user_id:
                await websocket.send_text(json.dumps({
                    "type": "output",
                    "data": "user_id required in server mode\r\n",
                }))
                await websocket.close(code=1008)
                return
            await _worker_terminal_session(websocket, user_id)
        else:
            await _local_terminal_session(websocket)
    except Exception as exc:
        logger.exception("Workspace terminal failed")
        try:
            await websocket.send_text(json.dumps({
                "type": "output",
                "data": f"\r\n[terminal error] {exc}\r\n",
            }))
        except Exception:
            pass


@app.get("/api/workspace/files")
async def serve_workspace_file(
    path: str = Query(..., description="Absolute or workspace-relative file path"),
    session_id: str = Query(default="", description="Session ID to resolve custom workdir boundaries"),
) -> FileResponse:
    resolved = _resolve_readable_file_path(path, session_id)
    if not resolved.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(resolved)


@app.get("/api/structure/view")
async def view_structure(
    path: str = Query(..., description="Absolute or workspace-relative structure file path"),
    session_id: str = Query(default="", description="Session ID to resolve custom workdir boundaries"),
) -> JSONResponse:
    from io import StringIO

    try:
        from ase.io import write as ase_write
    except ImportError:
        raise HTTPException(status_code=500, detail="ASE is not installed")

    resolved = _resolve_readable_file_path(path, session_id)
    if not resolved.is_file():
        raise HTTPException(status_code=404, detail="File not found")

    try:
        atoms = _ase_read_structure(resolved)
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
    owner_id, _ = _load_session_state(session_id)
    session_dir = _get_workdir_for_session(session_id)
    if not session_dir.exists():
        return JSONResponse({"files": []})
    files = [
        {
            "name": f.name,
            "path": _control_plane_path_to_worker(owner_id, f) if owner_id else str(f),
            "size": f.stat().st_size,
        }
        for f in sorted(session_dir.rglob("*"))
        if f.is_file()
    ]
    return JSONResponse({"files": files})


@app.post("/api/sessions/{session_id}/files")
async def upload_session_file(session_id: str, file: UploadFile = File(...)) -> JSONResponse:
    owner_id, _ = _load_session_state(session_id)
    session_dir = _get_workdir_for_session(session_id)
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
        "path": _control_plane_path_to_worker(owner_id, target) if owner_id else str(target),
        "size": size,
    })


@app.delete("/api/sessions/{session_id}/files")
async def delete_session_file(
    session_id: str,
    path: str = Query(..., description="Absolute or session-relative file path"),
) -> JSONResponse:
    owner_id, _ = _load_session_state(session_id)
    session_dir = _get_workdir_for_session(session_id).resolve()
    if _MATCREATOR_MODE == "server" and owner_id:
        resolved = _map_worker_path_to_control_plane(owner_id, path)
        if resolved is None:
            raise HTTPException(status_code=404, detail="File not found")
        resolved = resolved.resolve()
    else:
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
    await asyncio.to_thread(request_cancellation, session_id, reason)
    await asyncio.to_thread(
        AgentGraphLogger(session_id).mark_running_nodes_cancelled,
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
    await asyncio.to_thread(request_step_cancellation, session_id, step_number, reason)
    found = await asyncio.to_thread(
        AgentGraphLogger(session_id).cancel_step_node_by_number,
        step_number, f"Cancelled by user ({reason})"
    )
    return JSONResponse({
        "status": "ok",
        "session_id": session_id,
        "step_number": step_number,
        "graph_updated": found,
        "message": f"Step {step_number} cancellation requested.",
    })


@app.get("/api/skills")
async def list_skills() -> JSONResponse:
    """Return all loaded skills with their planning_enabled status and parent skill (if any)."""
    from matcreator.skill import _MODULE_SKILLS_ROOT, _discover_skill_dirs  # noqa: PLC0415

    parent_map: dict[str, str] = {}
    for root in [_MODULE_SKILLS_ROOT, workspace_skills_dir()]:
        for path in _discover_skill_dirs(root):
            parent_skill_md = path.parent / "SKILL.md"
            if parent_skill_md.is_file():
                parent_map[path.name] = path.parent.name

    default_skill_names = get_default_skill_names()
    disabled_skills = set(get_disabled_skills())
    skills = [
        {
            "name": s.name,
            "description": s.description or "",
            "planning_enabled": s.name in PLANNING_SKILL_NAMES,
            "enabled": s.name not in disabled_skills,
            "parent": parent_map.get(s.name),
            "is_custom": s.name not in default_skill_names,
        }
        for s in sorted(ALL_SKILLS, key=lambda s: s.name)
    ]
    return JSONResponse(skills)


@app.get("/api/skills/defaults")
async def list_default_skill_names() -> JSONResponse:
    """Return the names of all bundled default skills."""
    return JSONResponse({"names": sorted(get_default_skill_names())})


_SKILL_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


def _validate_skill_md_name(content: bytes, expected_name: str) -> None:
    """Raise ValueError if the SKILL.md frontmatter name field doesn't match expected_name."""
    text = content.decode("utf-8", errors="replace")
    parts = text.split("---")
    if len(parts) < 3:
        return  # No frontmatter; ADK will catch structural issues later
    try:
        frontmatter = yaml.safe_load(parts[1])
    except yaml.YAMLError as exc:
        raise ValueError(f"SKILL.md has invalid YAML frontmatter: {exc}")
    if not isinstance(frontmatter, dict):
        return
    md_name = frontmatter.get("name")
    if md_name is not None and md_name != expected_name:
        raise ValueError(
            f"SKILL.md 'name' field ('{md_name}') does not match the skill directory name ('{expected_name}'). "
            f"Update the 'name' field in SKILL.md to '{expected_name}'."
        )


@app.post("/api/skills/custom")
async def create_custom_skill(
    name: str = Form(...),
    skill_md: UploadFile = File(...),
    references: List[UploadFile] = File(default=[]),
    scripts: List[UploadFile] = File(default=[]),
) -> JSONResponse:
    """Upload a custom skill to the workspace skills directory."""
    name = name.strip()
    if not _SKILL_NAME_RE.match(name):
        raise HTTPException(
            status_code=422,
            detail="Skill name must be lowercase alphanumeric with hyphens/underscores, starting with a letter or digit.",
        )
    if name in get_default_skill_names():
        raise HTTPException(
            status_code=409,
            detail=f"'{name}' is a built-in default skill. Custom skills cannot use the same name.",
        )

    skill_dir = workspace_skills_dir() / name
    skill_dir.mkdir(parents=True, exist_ok=True)

    try:
        skill_md_content = await skill_md.read()
        try:
            _validate_skill_md_name(skill_md_content, name)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        (skill_dir / "SKILL.md").write_bytes(skill_md_content)

        ref_names = []
        non_empty_refs = [r for r in references if r.filename]
        if non_empty_refs:
            ref_dir = skill_dir / "references"
            ref_dir.mkdir(exist_ok=True)
            for ref_file in non_empty_refs:
                safe_name = _safe_upload_filename(ref_file.filename or "ref")
                (ref_dir / safe_name).write_bytes(await ref_file.read())
                ref_names.append(safe_name)

        script_names = []
        non_empty_scripts = [s for s in scripts if s.filename]
        if non_empty_scripts:
            scripts_dir = skill_dir / "scripts"
            scripts_dir.mkdir(exist_ok=True)
            for script_file in non_empty_scripts:
                safe_name = _safe_upload_filename(script_file.filename or "script")
                (scripts_dir / safe_name).write_bytes(await script_file.read())
                script_names.append(safe_name)
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Failed to write skill files: {exc}")
    finally:
        await skill_md.close()
        for r in references:
            await r.close()
        for s in scripts:
            await s.close()

    try:
        refresh_skills()
    except Exception as exc:
        shutil.rmtree(skill_dir, ignore_errors=True)
        raise HTTPException(status_code=422, detail=f"Skill files were written but failed to load: {exc}")
    return JSONResponse({"status": "ok", "name": name, "references": ref_names, "scripts": script_names})


@app.delete("/api/skills/custom/{skill_name}")
async def delete_custom_skill(skill_name: str) -> JSONResponse:
    """Delete a custom workspace skill. Default skills cannot be deleted."""
    if not _SKILL_NAME_RE.match(skill_name):
        raise HTTPException(status_code=400, detail=f"Invalid skill name: '{skill_name}'.")
    if skill_name in get_default_skill_names():
        raise HTTPException(
            status_code=400,
            detail=f"'{skill_name}' is a built-in default skill and cannot be deleted.",
        )
    root = workspace_skills_dir()
    skill_dir = root / skill_name
    if skill_dir.resolve() == root.resolve() or not skill_dir.resolve().is_relative_to(root.resolve()):
        raise HTTPException(status_code=400, detail="Invalid skill path.")
    if not skill_dir.exists():
        raise HTTPException(status_code=404, detail=f"Custom skill '{skill_name}' not found in workspace.")
    try:
        shutil.rmtree(skill_dir)
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Failed to delete skill: {exc}")

    try:
        refresh_skills()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Skill deleted but registry reload failed: {exc}")
    return JSONResponse({"status": "ok", "deleted": skill_name})


@app.get("/api/settings")
async def get_settings() -> JSONResponse:
    """Return the current user config."""
    return JSONResponse(load_config())


class SettingsBody(BaseModel):
    planning: dict | None = None
    user: dict | None = None
    skills: dict | None = None
    workspace: dict | None = None


@app.put("/api/settings")
async def update_settings(body: SettingsBody) -> JSONResponse:
    """Merge *body* into the config, persist it, and reload skills."""
    config = load_config()
    if body.planning is not None:
        config.setdefault("planning", {}).update(body.planning)
    if body.user is not None:
        config.setdefault("user", {}).update(body.user)
    if body.skills is not None:
        config.setdefault("skills", {}).update(body.skills)
    if body.workspace is not None:
        config.setdefault("workspace", {}).update(body.workspace)
    save_config(config)
    try:
        refresh_skills()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Settings saved but skill registry reload failed: {exc}")
    return JSONResponse({
        "status": "ok",
        "planning_skill_names": sorted(PLANNING_SKILL_NAMES),
    })


@app.get("/api/env-config")
async def get_env_config() -> JSONResponse:
    """Return current LLM/compute configuration, masking sensitive fields.

    In local mode, values come from ~/.matcreator/config.yaml (with .env as
    fallback). In server mode, values come from the .env file as before.
    """
    mode = os.environ.get("MATCREATOR_MODE", "local")
    result: dict[str, str] = {}

    if mode == "local":
        from matcreator.config import get_config_value
        for env_key in _ENV_FIELDS:
            yaml_key = ENV_TO_YAML.get(env_key)
            if yaml_key:
                val = get_config_value(yaml_key)
            else:
                val = dotenv_values(ENV_PATH).get(env_key, "") if ENV_PATH.exists() else ""
            sensitive = yaml_key in SENSITIVE_YAML_KEYS if yaml_key else env_key in _SENSITIVE_FIELDS
            result[env_key] = "***" if (sensitive and val) else val
    else:
        values = dotenv_values(ENV_PATH) if ENV_PATH.exists() else {}
        for field in _ENV_FIELDS:
            val = values.get(field, "")
            result[field] = "***" if (field in _SENSITIVE_FIELDS and val) else val

    return JSONResponse(result)


class EnvConfigBody(BaseModel):
    values: dict[str, str]


@app.put("/api/env-config")
async def update_env_config(body: EnvConfigBody) -> JSONResponse:
    """Write updated configuration fields.

    In local mode, writes to ~/.matcreator/config.yaml.
    In server mode, writes to agents/MatCreator/.env as before.
    """
    mode = os.environ.get("MATCREATOR_MODE", "local")

    if mode == "local":
        from matcreator.config import set_config_value
        for key, value in body.values.items():
            if key not in _ENV_FIELDS:
                continue
            yaml_key = ENV_TO_YAML.get(key)
            if yaml_key is None:
                continue
            sensitive = yaml_key in SENSITIVE_YAML_KEYS
            if sensitive and value == "***":
                continue
            set_config_value(yaml_key, value)
            if value:
                os.environ[key] = value
            else:
                os.environ.pop(key, None)
    else:
        ENV_PATH.parent.mkdir(parents=True, exist_ok=True)
        if not ENV_PATH.exists():
            ENV_PATH.touch()
        for key, value in body.values.items():
            if key not in _ENV_FIELDS:
                continue
            if key in _SENSITIVE_FIELDS and value == "***":
                continue
            dotenv_set_key(str(ENV_PATH), key, value)
            if value:
                os.environ[key] = value
            else:
                os.environ.pop(key, None)

    return JSONResponse({"status": "ok"})


@app.post("/api/restart-backend")
async def restart_backend(user_id: str = Query(default="")) -> JSONResponse:
    """Restart the ADK backend.

    In server mode: restart the user's worker container.
    In local mode: kill and relaunch the local ADK process on the configured port.
    """
    global _adk_process

    if _MATCREATOR_MODE == "server":
        if not user_id:
            raise HTTPException(status_code=400, detail="user_id required in server mode")
        try:
            import docker as _docker
            dc = _get_docker()
            name = _worker_container_name(user_id)
            try:
                c = dc.containers.get(name)
                c.restart(timeout=15)
            except _docker.errors.NotFound:
                target = await asyncio.to_thread(ensure_worker_running, user_id)
                return JSONResponse({"status": "created", "user_id": user_id, "target": target})
            target = _worker_registry.get(user_id)
            return JSONResponse({"status": "restarting", "user_id": user_id, "target": target})
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    _kill_port(get_adk_port())
    await asyncio.sleep(1.5)

    try:
        _adk_process = subprocess.Popen(
            get_local_adk_command(),
            cwd=str(ROOT),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="'matcreator' command not found in PATH")
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Failed to start backend: {exc}")

    return JSONResponse({"status": "restarting", "pid": _adk_process.pid})


@app.get("/api/backend-status")
async def get_backend_status(user_id: str = Query(default="")) -> JSONResponse:
    """Check whether the ADK backend is reachable.

    In server mode: checks the user's worker container port.
    In local mode: checks the configured ADK port on localhost.
    """
    if _MATCREATOR_MODE == "server" and user_id:
        with _worker_registry_lock:
            target = _worker_registry.get(user_id)
        if target is None:
            return JSONResponse({"ready": False})
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                resp = await client.get(f"{target.rstrip('/')}/list-apps")
            return JSONResponse({"ready": resp.status_code < 500})
        except httpx.HTTPError:
            return JSONResponse({"ready": False})

    return JSONResponse({"ready": _is_port_open(port=get_adk_port())})


# Serve built frontend in production
_dist = Path(__file__).parent / "vite-frontend" / "dist"
if _dist.exists():
    app.mount("/", StaticFiles(directory=str(_dist), html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    host = "0.0.0.0" if _MATCREATOR_MODE == "server" else "127.0.0.1"
    uvicorn.run(app, host=host, port=get_web_port())
