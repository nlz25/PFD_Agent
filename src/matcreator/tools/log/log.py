"""tools/workflow_log_tools.py

Environment-backed workflow log helpers.

- create_workflow_log always creates a new timestamped file and stores its
  absolute path in the environment variable LOG_PATH.
- update_workflow_log reads LOG_PATH by default and updates/appends steps.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Literal, Dict, Any, Iterable, Callable
from functools import wraps
import asyncio
from matcreator.constants import TASK_INSTRUCTIONS
# Default filename used to derive the timestamped target when creating logs.
LOG_PATH = "workflow_log.json"
# Environment variable name holding the active workflow log path.
ENV_LOG_PATH = "LOG_PATH"


def _iso_now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def _resolve_base_path() -> Path:
    """Get the base path to derive the new log file name.

    Prefers the current value of LOG_PATH env if set; otherwise uses the
    module default LOG_PATH in the current working directory.
    """

    p = os.environ.get(ENV_LOG_PATH, LOG_PATH)
    
    return Path(p).expanduser().resolve()


def _timestamped_path(base: Path) -> Path:
    """Create a unique timestamped filename based on base path."""

    root, ext = os.path.splitext(str(base))
    if not ext:
        ext = ".json"
    ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    candidate = Path(f"{root}_{ts}{ext}")
    i = 1
    while candidate.exists():
        candidate = Path(f"{root}_{ts}_{i}{ext}")
        i += 1
    candidate.parent.mkdir(parents=True, exist_ok=True)
    return candidate


def _load_log(path: Optional[str] = None) -> Dict[str, Any]:
    """Load log from path/env or return empty structure if missing."""

    p = Path(path) if path else Path(os.environ.get(ENV_LOG_PATH, LOG_PATH))
    p = p.expanduser().resolve()
    if p.exists():
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {"workflow_name": None, "version": None, "steps": []}


def _save_log(data: Dict[str, Any], path: Optional[str] = None) -> str:
    p = Path(path) if path else Path(os.environ.get(ENV_LOG_PATH, LOG_PATH))
    p = p.expanduser().resolve()
    if p.parent:
        p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return str(p)


def create_workflow_log(
    workflow_name: str,#Literal["pfd_finetune","pfd_distillation"],
    task_instructions: Optional[Dict[str, Any]] = {},
    additional_info: Optional[Dict[str,Any]] = None,
    
):
    """Initialize a workflow log and update the LOG_PATH environment variable.

    - Uses a timestamped file name to avoid clobbering.
    - Stores absolute path in env LOG_PATH for subsequent updates.
    """

    base = _resolve_base_path()
    target = _timestamped_path(base)
    data = {
        "workflow_name": workflow_name,
        "default_instructions": task_instructions.get(workflow_name, ""),
        "created_at": _iso_now(),
        "steps": [],
    }
    if additional_info:
        data.update(additional_info)
    saved_path = _save_log(data, str(target))
    os.environ[ENV_LOG_PATH] = saved_path
    return {
        "status": "ok",
        "message": f"Workflow log created: {saved_path}",
        "path": saved_path,
        "workflow_name": workflow_name,
        "default_instructions": task_instructions.get(workflow_name, ""),
        "env": {ENV_LOG_PATH: saved_path},
    }

def update_workflow_log_plan(
    planning_details: str="No details provided."    
):
    """
    Update workflow planning details in the log. 
    Always generates 'planning_details' based on default instruction and current context.
    Recursively invoke this tool until satisfactory planning is achieved.
    
    Args:
        planning_details: The detailed plan for the workflow.
    """
    if p := os.environ.get(ENV_LOG_PATH):
        p_abs = str(Path(p).expanduser().resolve())
    else:
        return {"message":"No workflow log path found in environment."}
    data = _load_log(p_abs)
    data["planning_details"] = planning_details
    saved_path = _save_log(data, p_abs)
    return {
        "message": f"Updated planning details. Successfully saved log to {saved_path}.",
        "planning_details": planning_details
    }

#@mcp.tool()
def resubmit_workflow_log(
    log_path:str,
):
    """Reload a previous workflow log. Using this tool when resubmitting a workflow.

    Args:
        log_path (str): path to the existing log file

    Returns:
        _type_: _description_
    """
    p = Path(log_path).expanduser().resolve()

    if not p.exists():
        return {
            "status": "error",
            "message": f"Log file not found: {p}",
            "path": str(p),
        }

    try:
        base = _resolve_base_path()
        target = _timestamped_path(base)
        # Simply load and return the entire log content
        data = _load_log(str(p))
        data.update({"resubmitted_at": _iso_now()})
        
        saved_path = _save_log(data, str(target))
        os.environ[ENV_LOG_PATH] = saved_path
        return {
            "status": "ok",
            "message": f"Workflow reloaded: {saved_path}",
            "path": saved_path,
            "env": {ENV_LOG_PATH: saved_path},
    }
       
    except Exception as exc:
        return {
            "status": "error",
            "message": f"Failed to read or parse log: {exc}",
            "path": str(p),
        }


def update_workflow_log(
    step_name: str,
    status: str = "completed",
    input: Optional[Dict[str,Any]] = None,
    output: Optional[Dict[str,Any]] = None,
    notes: Optional[str] = None,
):
    """Append a step information to the workflow log."""

    if p := os.environ.get(ENV_LOG_PATH):
        p_abs = str(Path(p).expanduser().resolve())
    else:
        return {"message":"No workflow log path found in environment."}
    data = _load_log(p_abs)
    timestamp = _iso_now()

    entry = {
        "id": len(data.get("steps", [])) + 1,
        "name": step_name,
        "status": status,
        "created_at": timestamp,
        "input": input,
        "output": output,
        "notes": notes or "",
    }
    data.setdefault("steps", []).append(entry)
    saved_path = _save_log(data, p_abs)
    return {
        "message": f"Added new step '{step_name}'. Successfully saved log to {saved_path}."
    }

#@mcp.tool()
def read_workflow_log(
    log_path: Optional[str] = None,
    last_n: int = 10,
):
    """Return the log header and the last N steps. Always invoke this tool after each major step.

    Args:
        log_path: Optional explicit path. Defaults to env LOG_PATH or the module default. Do not provide 'log_path' unless being explicitly requested.
        last_n: Number of recent steps to return (default 10).

    Returns:
        {
          "status": "ok"|"error",
          "path": str,
          "header": dict,          # all top-level keys except "steps"
          "steps": list[dict],     # at most last_n entries
          "total_steps": int,
          "returned_steps": int,
          "message": str (on error)
        }
    """

    # Resolve path precedence: explicit > env > default
    p = Path(log_path) if log_path else Path(os.environ.get(ENV_LOG_PATH, LOG_PATH))
    p = p.expanduser().resolve()

    if not p.exists():
        return {
            "status": "error",
            "message": f"Log file not found: {p}",
            "path": str(p),
        }

    try:
        data = _load_log(str(p))
        steps = data.get("steps", [])
        # Build header by excluding the steps list
        header = {k: v for k, v in data.items() if k != "steps"}

        # Clamp last_n to safe bounds
        n = max(0, min(int(last_n), 1000))
        tail = steps[-n:] if n > 0 else []
        return {
            "status": "ok",
            "path": str(p),
            "header": header,
            "steps": tail,
            "total_steps": len(steps),
            "returned_steps": len(tail),
        }
    except Exception as exc:
        return {
            "status": "error",
            "message": f"Failed to read or parse log: {exc}",
            "path": str(p),
        }


def _to_json_safe(x: Any) -> Any:
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, dict):
        return {k: _to_json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_json_safe(v) for v in x]
    return x

def _select(d: Dict[str, Any], keys: Optional[Iterable[str]]) -> Dict[str, Any]:
    if d is None:
        return {}
    if not keys:
        return d
    return {k: d.get(k) for k in keys if k in d}

def log_step(
    step_name: Optional[str] = None,
    include_input_keys: Optional[Iterable[str]] = None,
    include_output_keys: Optional[Iterable[str]] = None,
) -> Callable:
    """
    Decorator mode: log a workflow step using update_workflow_log (env-backed LOG_PATH).

    Use include_*_keys to limit logged fields.
    """
    def decorator(fn: Callable):
        name = step_name or fn.__name__

        @wraps(fn)
        def _wrapper(*args, **kwargs):
            input_payload = _select(_to_json_safe(kwargs), include_input_keys)
            result = fn(*args, **kwargs)
            output_payload = _to_json_safe(result)
            output_payload = _select(_to_json_safe(output_payload), include_output_keys)
            status = output_payload.pop("status", "completed") if isinstance(output_payload, dict) else "completed"
            update_workflow_log(step_name=name, status=status, input=input_payload, output=output_payload)
            return result

        @wraps(fn)
        async def _async_wrapper(*args, **kwargs):
            await _wrapper(*args, **kwargs)

        return _async_wrapper if asyncio.iscoroutinefunction(fn) else _wrapper
    return decorator


def after_tool_log_callback(tool, args, tool_context, tool_response,
                            *, include_input_keys: Optional[Iterable[str]] = None,
                            include_output_keys: Optional[Iterable[str]] = None,
                            step_name_map: Optional[Dict[str, str]] = None):
    """
    Agent-level after_tool_callback to append entries to the workflow log.

    Intended use:
        from matcreator.tools.log.log import after_tool_log_callback
        abacus_agent = LlmAgent(..., after_tool_callback=after_tool_log_callback)

    Behavior:
    - Derives step_name from tool.name (or step_name_map override).
    - Logs status from tool_response['status'] if present, else 'completed'.
    - Serializes args/tool_response to JSON-safe payloads; can restrict fields via include_*_keys.
    - Returns the original tool_response unmodified.
    """
    try:
        name = (step_name_map or {}).get(getattr(tool, 'name', ''), getattr(tool, 'name', 'tool'))
        input_payload = _select(_to_json_safe(args if isinstance(args, dict) else {}), include_input_keys)
        output_payload = _to_json_safe(tool_response)['structuredContent']['result']
        if isinstance(output_payload, dict):
            status = output_payload.pop('status', 'completed')
        else:
            status = 'completed'
        output_payload = _select(_to_json_safe(output_payload), include_output_keys)
        update_workflow_log(step_name=name, status=status, input=input_payload, output=output_payload)
    except Exception as exc:
        # Logging must never break tool execution; swallow errors but include minimal context
        try:
            update_workflow_log(step_name='after_tool_log_error', status='failed',
                                input={'tool': getattr(tool, 'name', 'tool')},
                                output={'message': str(exc)})
        except Exception:
            pass
    return tool_response