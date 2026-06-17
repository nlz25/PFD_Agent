#!/usr/bin/env python3
"""MatCreator CLI.

Usage:
    matcreator [OPTIONS] COMMAND [ARGS]

Commands:
    web         Launch the ADK web UI.
    api-server  Launch the ADK API server (used by the Streamlit app).
    run         Run the agent non-interactively on a single prompt.
    chat        Run the agent interactively in the terminal (workspace = CWD by default).

Examples:
    matcreator web
    matcreator web --reload-agents --port 8080
    matcreator api-server
    matcreator run -p "Build a silicon FCC structure"
    matcreator run -f prompt.txt --output-format json -o result.json
    matcreator chat
    matcreator chat --workspace ~/my-project
    MATCLAW_WORKSPACE=/data/ws matcreator run -p "hello"
"""

import asyncio
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
import warnings
from pathlib import Path
from typing import Union
import yaml

import click


_CONFIG_PATH = Path("~/.matcreator/config.yaml").expanduser()
_DEFAULT_ADK_DIR = Path("~/.matcreator/.adk").expanduser()
PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _make_agent_loader():
    """Return a custom AgentLoader that serves matcreator.agent.app directly.

    This avoids ADK's file-based discovery (which is case-sensitive and tightly
    coupled to a specific directory structure) and ensures session data lands in
    ~/.matcreator/.adk/ regardless of where the CLI is invoked from.
    """
    from google.adk.cli.utils.base_agent_loader import BaseAgentLoader

    class _MatCreatorLoader(BaseAgentLoader):
        def load_agent(self, agent_name: str):
            from matcreator.agent import app
            return app

        def list_agents(self) -> list[str]:
            return ["MatCreator"]

    return _MatCreatorLoader()


def _start_adk_server(
    host: str,
    port: int,
    log_level: str,
    web_ui: bool,
    reload_agents: bool = False,
    reload: bool = False,
) -> None:
    """Start the ADK server programmatically with controlled session storage."""
    import uvicorn
    from google.adk.cli.fast_api import get_fast_api_app

    session_db = _DEFAULT_ADK_DIR / "session.db"
    session_db.parent.mkdir(parents=True, exist_ok=True)

    fast_api = get_fast_api_app(
        agents_dir=str(_DEFAULT_ADK_DIR),   # unused for discovery; loader overrides it
        agent_loader=_make_agent_loader(),
        session_service_uri=f"sqlite:///{session_db}",
        web=web_ui,
        host=host,
        port=port,
        reload_agents=reload_agents,
    )

    config = uvicorn.Config(
        fast_api,
        host=host,
        port=port,
        log_level=log_level,
        reload=reload,
    )
    server = uvicorn.Server(config)
    try:
        server.run()
    except KeyboardInterrupt:
        pass


# Shared options applied to web/api-server subcommands
_shared_options = [
    click.option("--host", default="127.0.0.1", show_default=True,
                 help="Binding host for the ADK server."),
    click.option("--port", default=8000, show_default=True, type=int,
                 help="Port for the ADK server."),
    click.option("--workspace", default=None, metavar="DIR",
                 help="Override the workspace root directory (also settable via MATCLAW_WORKSPACE)."),
    click.option("--log-level",
                 type=click.Choice(["debug", "info", "warning", "error", "critical"],
                                   case_sensitive=False),
                 default="info", show_default=True,
                 help="Logging level for the ADK server."),
    click.option("-v", "--verbose", is_flag=True, default=False,
                 help="Shortcut for --log-level debug."),
]


def add_shared_options(fn):
    for opt in reversed(_shared_options):
        fn = opt(fn)
    return fn


def _default_workspace() -> Path:
    """Resolve the default workspace using the same logic as workspace.py."""
    env_val = os.environ.get("MATCLAW_WORKSPACE", "")
    if env_val:
        return Path(env_val).expanduser().resolve()
    return Path.home() / ".matcreator" / "workspace"


def _setup_workspace(workspace: str | None) -> Path:
    """Set MATCLAW_WORKSPACE env var and return the resolved path."""
    if workspace:
        ws_root = Path(workspace).expanduser().resolve()
    else:
        ws_root = _default_workspace()
    os.environ["MATCLAW_WORKSPACE"] = str(ws_root)
    ws_root.mkdir(parents=True, exist_ok=True)
    click.echo(f"Workspace: {ws_root}")
    return ws_root



def _run_with_env(cmd: list[str], env: dict[str, str]) -> None:
    click.echo("Starting: " + " ".join(cmd))
    try:
        subprocess.run(cmd, check=True, env=env)
    except KeyboardInterrupt:
        pass
    except subprocess.CalledProcessError as exc:
        sys.exit(exc.returncode)


def _ensure_project_imports() -> None:
    """Add project root and agents dir to sys.path for imports."""
    for p in (str(PROJECT_ROOT), str(PROJECT_ROOT / "agents")):
        if p not in sys.path:
            sys.path.insert(0, p)


def _resolve_kdg_cli() -> str:
    """Resolve the know-do-graph executable from the current environment."""
    local_bin = Path(sys.executable).resolve().with_name("know-do-graph")
    if local_bin.is_file():
        return str(local_bin)

    found = shutil.which("know-do-graph")
    if found:
        return found

    raise click.ClickException(
        "Could not find the `know-do-graph` executable in the current environment."
    )


def _matcreator_kdg_env() -> dict[str, str]:
    """Return an environment that pins KDG CLI calls to MatCreator's database."""
    _ensure_project_imports()
    from matcreator.constants import KNOW_DO_GRAPH_DB

    env = os.environ.copy()
    env["KDG_DB_PATH"] = str(KNOW_DO_GRAPH_DB)
    return env


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def main():
    """MatCreator CLI — manage and run the MatCreator agent."""


@main.command("web")
@add_shared_options
@click.option("--reload-agents", is_flag=True, default=False,
              help="Enable live reload when agent files change.")
@click.option("--reload", is_flag=True, default=False,
              help="Enable auto-reload for the FastAPI server.")
def web(host, port, workspace, log_level, verbose, reload_agents, reload):
    """Launch the ADK web UI (browser-based chat interface)."""
    _setup_workspace(workspace)
    _start_adk_server(
        host=host,
        port=port,
        log_level="debug" if verbose else log_level,
        web_ui=True,
        reload_agents=reload_agents,
        reload=reload,
    )


@main.command("api-server")
@add_shared_options
@click.option("--reload-agents", is_flag=True, default=False,
              help="Enable live reload when agent files change.")
@click.option("--reload", is_flag=True, default=False,
              help="Enable auto-reload for the FastAPI server.")
def api_server(host, port, workspace, log_level, reload_agents, verbose, reload):
    """Launch the ADK API server (used by the Streamlit app)."""
    _setup_workspace(workspace)
    _start_adk_server(
        host=host,
        port=port,
        log_level="debug" if verbose else log_level,
        web_ui=False,
        reload_agents=reload_agents,
        reload=reload,
    )


@main.group("graph", context_settings={"ignore_unknown_options": True})
def graph_group():
    """Run Know-Do Graph inspection commands against MatCreator's active database."""


@graph_group.command(
    "serve",
    context_settings={"ignore_unknown_options": True, "allow_extra_args": True},
)
@click.pass_context
def graph_serve(ctx: click.Context) -> None:
    """Launch the Know-Do Graph HTTP UI/server using MatCreator's active database."""
    env = _matcreator_kdg_env()
    cmd = [_resolve_kdg_cli(), "serve", *ctx.args]
    _run_with_env(cmd, env)


@graph_group.command(
    "stats",
    context_settings={"ignore_unknown_options": True, "allow_extra_args": True},
)
@click.pass_context
def graph_stats(ctx: click.Context) -> None:
    """Show graph statistics using MatCreator's configured Know-Do Graph database."""
    env = _matcreator_kdg_env()
    cmd = [_resolve_kdg_cli(), "graph", "stats", *ctx.args]
    _run_with_env(cmd, env)


@graph_group.command(
    "neighbors",
    context_settings={"ignore_unknown_options": True, "allow_extra_args": True},
)
@click.argument("entry_id")
@click.pass_context
def graph_neighbors(ctx: click.Context, entry_id: str) -> None:
    """Show graph neighbors for an entry using MatCreator's active database."""
    env = _matcreator_kdg_env()
    cmd = [_resolve_kdg_cli(), "graph", "neighbors", entry_id, *ctx.args]
    _run_with_env(cmd, env)


@graph_group.command(
    "export",
    context_settings={"ignore_unknown_options": True, "allow_extra_args": True},
)
@click.pass_context
def graph_export(ctx: click.Context) -> None:
    """Export graph entries using MatCreator's configured Know-Do Graph database."""
    env = _matcreator_kdg_env()
    cmd = [_resolve_kdg_cli(), "graph", "export", *ctx.args]
    _run_with_env(cmd, env)


# ---------------------------------------------------------------------------
# Non-interactive runner core (used by `run` and benchmarks)
# ---------------------------------------------------------------------------

async def run_agent_async(
    workspace_dir: str,
    prompt: str,
    max_turns: int = 50,
    session_id: str | None = None,
    flash: bool = False,
) -> dict:
    """Run the agent non-interactively on a single prompt.

    Returns a dict with keys: answer, model_name, num_turns, is_error,
    duration_ms, num_events.
    """
    os.environ["MATCLAW_WORKSPACE"] = str(Path(workspace_dir).resolve())

    from google.adk.runners import Runner
    from google.adk.sessions.sqlite_session_service import SqliteSessionService
    from google.genai import types

    from matcreator.agent import app

    _adk_db = _DEFAULT_ADK_DIR / "session.db"
    _adk_db.parent.mkdir(parents=True, exist_ok=True)
    session_service = SqliteSessionService(db_path=str(_adk_db))
    runner = Runner(app=app, session_service=session_service)

    session = await runner.session_service.create_session(
        app_name=app.name,
        user_id="user",
        state={"agent_mode": "flash" if flash else "bench"},
        session_id=session_id,
    )

    user_message = types.Content(
        role="user",
        parts=[types.Part(text=prompt)],
    )

    start_ms = time.monotonic_ns() // 1_000_000
    events = []
    turn_count = 0
    is_error = False
    error_msg = ""

    try:
        async for event in runner.run_async(
            user_id="user",
            session_id=session.id,
            new_message=user_message,
        ):
            events.append(event)
            if hasattr(event, "is_final_response") and event.is_final_response():
                turn_count += 1
            if turn_count >= max_turns:
                break
    except Exception as exc:
        is_error = True
        error_msg = str(exc)

    duration_ms = (time.monotonic_ns() // 1_000_000) - start_ms

    answer = ""
    for event in reversed(events):
        if event.content and event.content.parts:
            for part in event.content.parts:
                if hasattr(part, "text") and part.text:
                    answer = part.text
                    break
            if answer:
                break

    if is_error and not answer:
        answer = error_msg

    return {
        "answer": answer,
        "model_name": os.environ.get("LLM_MODEL", "matcreator"),
        "num_turns": turn_count,
        "is_error": is_error,
        "duration_ms": duration_ms,
        "num_events": len(events),
    }


# ---------------------------------------------------------------------------
# `matcreator run` subcommand
# ---------------------------------------------------------------------------

@main.command("run")
@click.option("--workspace", default=None, metavar="DIR",
              help="Override the workspace root directory (also settable via MATCLAW_WORKSPACE).")
@click.option("-p", "--prompt", "prompt_text", default=None,
              help="Inline prompt text to send to the agent.")
@click.option("-f", "--prompt-file", default=None, type=click.Path(exists=True),
              help="Read prompt from a file (mutually exclusive with -p).")
@click.option("--output-format", "output_format",
              type=click.Choice(["text", "json"], case_sensitive=False),
              default="text", show_default=True,
              help="Output format: 'text' prints the answer, 'json' prints full structured result.")
@click.option("-o", "--output", "output_file", default=None, type=click.Path(),
              help="Write output to a file instead of stdout.")
@click.option("--max-turns", default=50, show_default=True, type=int,
              help="Maximum agent turns before stopping.")
@click.option("--session-id", default=None, metavar="ID",
              help="Custom session ID (default: auto-generated by ADK).")
@click.option("--flash", is_flag=True, default=False,
              help="Run in Flash mode: thinking agent executes directly, no DAG or approval required.")
def run_cmd(workspace, prompt_text, prompt_file, output_format, output_file, max_turns, session_id, flash):
    """Run the agent non-interactively on a single prompt."""
    if prompt_text and prompt_file:
        raise click.UsageError("Use either -p/--prompt or -f/--prompt-file, not both.")
    if not prompt_text and not prompt_file:
        raise click.UsageError("Provide a prompt via -p/--prompt or -f/--prompt-file.")

    if prompt_file:
        prompt_text = Path(prompt_file).read_text().strip()

    ws_root = Path(workspace).expanduser().resolve() if workspace else _default_workspace()
    ws_root.mkdir(parents=True, exist_ok=True)

    result = asyncio.run(run_agent_async(str(ws_root), prompt_text, max_turns, session_id, flash))

    if output_format == "json":
        content = json.dumps(result, ensure_ascii=False, indent=2)
    else:
        content = result["answer"]

    if output_file:
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        Path(output_file).write_text(content)
    else:
        click.echo(content)


# ---------------------------------------------------------------------------
# `matcreator chat` subcommand — interactive terminal REPL
# ---------------------------------------------------------------------------

async def _chat_loop(workspace_dir: Path, session_id: str, agent_mode: str = "flash") -> None:
    """Run the interactive agent REPL."""
    from google.adk.runners import Runner
    from google.adk.sessions.sqlite_session_service import SqliteSessionService
    from google.genai import types

    from matcreator.agent import app
    from matcreator.workspace import init_workspace

    init_workspace()

    _adk_db = _DEFAULT_ADK_DIR / "session.db"
    _adk_db.parent.mkdir(parents=True, exist_ok=True)
    session_service = SqliteSessionService(db_path=str(_adk_db))
    runner = Runner(app=app, session_service=session_service)

    session = await runner.session_service.create_session(
        app_name=app.name,
        user_id="user",
        state={"agent_mode": agent_mode},
        session_id=session_id,
    )
    actual_id = session.id

    mode_label = "Flash" if agent_mode == "flash" else "Plan"
    click.echo(f"\nMatCreator chat  |  workspace: {workspace_dir}  |  mode: {mode_label}")
    click.echo(f"Session: {actual_id}")
    click.echo("Type /quit or /exit to end.  /session shows the session ID.  /workspace shows the path.")
    click.echo("─" * 60)

    while True:
        try:
            user_input = input("\nYou> ").strip()
        except (EOFError, KeyboardInterrupt):
            click.echo(f"\n\nSession saved.  Resume with: matcreator chat --session {actual_id}")
            break

        if not user_input:
            continue

        if user_input in ("/quit", "/exit", "quit", "exit"):
            click.echo(f"\nSession saved.  Resume with: matcreator chat --session {actual_id}")
            break
        if user_input == "/session":
            click.echo(f"Session ID: {actual_id}")
            continue
        if user_input == "/workspace":
            click.echo(f"Workspace: {workspace_dir}")
            continue

        user_message = types.Content(
            role="user",
            parts=[types.Part(text=user_input)],
        )

        click.echo("\nMatCreator> ", nl=False)
        answer_parts: list[str] = []
        try:
            async for event in runner.run_async(
                user_id="user",
                session_id=actual_id,
                new_message=user_message,
            ):
                if event.content and event.content.parts:
                    for part in event.content.parts:
                        if hasattr(part, "text") and part.text:
                            click.echo(part.text, nl=False)
                            answer_parts.append(part.text)
        except KeyboardInterrupt:
            click.echo("\n[interrupted]")
            continue
        except Exception as exc:
            click.echo(f"\n[error: {exc}]")
            continue

        if answer_parts:
            click.echo()  # newline after streamed answer


@main.command("chat")
@click.option("--workspace", default=None, metavar="DIR",
              help="Workspace directory (defaults to current working directory).")
@click.option("--session", "session_id", default=None, metavar="SESSION_ID",
              help="Resume an existing session by ID.")
@click.option("--plan", "use_plan", is_flag=True, default=False,
              help="Use standard plan mode (thinking + DAG execution) instead of the default Flash mode.")
def chat_cmd(workspace, session_id, use_plan):
    """Run the agent interactively in the terminal.

    Defaults to Flash mode for fast, direct responses.  Pass --plan to use
    the full planning + execution pipeline instead.

    The workspace defaults to the current working directory, making it easy
    to use MatCreator directly from any project folder.  All session state
    is stored in ~/.matcreator/.adk/.  Resume a previous session with
    --session <id>.
    """
    if workspace:
        ws_root = Path(workspace).expanduser().resolve()
    else:
        ws_root = Path.cwd()

    ws_root.mkdir(parents=True, exist_ok=True)
    os.environ["MATCLAW_WORKSPACE"] = str(ws_root)
    warnings.filterwarnings("ignore", category=UserWarning)

    agent_mode = "normal" if use_plan else "flash"
    sid = session_id or str(uuid.uuid4())
    asyncio.run(_chat_loop(ws_root, sid, agent_mode))


# ---------------------------------------------------------------------------
# `matcreator knowledge` subcommand group
# ---------------------------------------------------------------------------

@main.group("knowledge")
def knowledge_group():
    """Inspect and manage the knowledge graph."""


def _ensure_workspace(workspace: str | None) -> None:
    if workspace:
        ws_root = Path(workspace).expanduser().resolve()
        os.environ["MATCLAW_WORKSPACE"] = str(ws_root)
        ws_root.mkdir(parents=True, exist_ok=True)


@knowledge_group.command("query")
@click.argument("text")
@click.option("--top-k", default=15, show_default=True, type=int,
              help="Maximum number of nodes to return.")
@click.option("--depth", default=2, show_default=True, type=int,
              help="BFS expansion depth from seed nodes.")
@click.option("--workspace", default=None, metavar="DIR",
              help="Override the workspace root directory.")
def knowledge_query(text, top_k, depth, workspace):
    """Query the memory knowledge graph for nodes matching TEXT."""
    _ensure_workspace(workspace)
    from matcreator.knowledge.query import query_knowledge_graph
    result = query_knowledge_graph(text, depth=depth, top_k=top_k)
    click.echo(result)


@knowledge_group.command("search-skills")
@click.argument("text")
@click.option("--top-k", default=5, show_default=True, type=int,
              help="Maximum number of skills to return.")
@click.option("--workspace", default=None, metavar="DIR",
              help="Override the workspace root directory.")
def knowledge_search_skills(text, top_k, workspace):
    """Search for skill nodes semantically matching TEXT."""
    _ensure_workspace(workspace)
    from matcreator.knowledge.query import search_skills
    result = search_skills(text, top_k=top_k)
    click.echo(result)


@knowledge_group.command("related-skills")
@click.argument("start_node")
@click.option("--top-k", default=5, show_default=True, type=int,
              help="Maximum number of related skills to return.")
@click.option("--depth", default=2, show_default=True, type=int,
              help="BFS depth limit.")
@click.option("--workspace", default=None, metavar="DIR",
              help="Override the workspace root directory.")
def knowledge_related_skills(start_node, top_k, depth, workspace):
    """Traverse the dependency graph from a known skill START_NODE."""
    _ensure_workspace(workspace)
    from matcreator.knowledge.query import get_related_skills
    result = get_related_skills(start_node, top_k=top_k, depth=depth)
    click.echo(result)


@knowledge_group.command("stats")
@click.option("--workspace", default=None, metavar="DIR",
              help="Override the workspace root directory.")
def knowledge_stats(workspace):
    """Print durable Know-Do and writable MemGraph counts."""
    _ensure_workspace(workspace)
    from matcreator.knowledge.kdg_memory import iter_memory
    from matcreator.knowledge.query import _get_kg
    graph = _get_kg()
    stats = graph.stats()
    memories = list(iter_memory(graph))
    click.echo("Know-Do graph (all native nodes):")
    click.echo(f"  Nodes: {stats['nodes']}  Edges: {stats['edges']}")
    click.echo("MemGraph nodes:")
    click.echo(
        f"  Entries: {len(memories)}  "
        f"Unpromoted: {sum(not entry.promoted for entry in memories)}"
    )


@knowledge_group.command("seed")
@click.option("--workspace", default=None, metavar="DIR",
              help="Override the workspace root directory.")
def knowledge_seed(workspace):
    """Seed Know-Do Graph with all SKILL.md and guide entries."""
    _ensure_workspace(workspace)
    from matcreator.skill import seed_skills_to_graph
    result = seed_skills_to_graph()
    click.echo(
        f"Created {result['seeded']} entries and "
        f"{result['edges_created']} dependency edges."
    )


@knowledge_group.command("migrate")
@click.option("--workspace", default=None, metavar="DIR",
              help="Override the workspace root directory.")
@click.option("--memory-md", default=None, metavar="FILE",
              help="Also import a legacy MEMORY.md file.")
def knowledge_migrate(workspace, memory_md):
    """Migrate legacy skill, memory, and optional MEMORY.md data."""
    _ensure_workspace(workspace)
    from matcreator.constants import KNOW_DO_GRAPH_DB
    from matcreator.knowledge.kdg_memory import iter_memory
    from matcreator.knowledge.migrate import migrate_memory_md
    from matcreator.knowledge.query import _get_kg, get_migration_result

    result = get_migration_result()
    click.echo(
        "Newly migrated from legacy sources: "
        f"{result.get('know_do_nodes', 0)} durable entries, "
        f"{result.get('memory_entries', 0)} memories, "
        f"{result.get('edges', 0)} edges."
    )
    if memory_md:
        md_result = migrate_memory_md(memory_md)
        click.echo(md_result["message"])

    graph = _get_kg()
    stats = graph.stats()
    memory_count = sum(1 for _ in iter_memory(graph))
    click.echo(f"Database: {KNOW_DO_GRAPH_DB}")
    click.echo(
        f"Current graph: {stats['nodes']} entries "
        f"({memory_count} memories), {stats['edges']} edges."
    )


@knowledge_group.command("distill")
@click.option("--min-evidence", default=3, show_default=True, type=int,
              help="Minimum similar memories required for promotion.")
@click.option("--stale-days", default=30, show_default=True, type=int,
              help="Age threshold for pruning unsuccessful memory.")
@click.option("--workspace", default=None, metavar="DIR",
              help="Override the workspace root directory.")
def knowledge_distill(min_evidence, stale_days, workspace):
    """Promote repeated successful memory into durable Know-Do entries."""
    _ensure_workspace(workspace)
    from matcreator.knowledge.synthesizer import run_knowledge_synthesizer
    result = run_knowledge_synthesizer(
        stale_days=stale_days,
        min_insights_for_workflow=min_evidence,
    )
    click.echo(result["message"])


# ---------------------------------------------------------------------------
# `matcreator config` subcommand group
# ---------------------------------------------------------------------------

_SENSITIVE_CONFIG_KEYS = frozenset({"llm.api_key", "bohrium.password"})


def _mask(key: str, value: str) -> str:
    return "***" if key in _SENSITIVE_CONFIG_KEYS and value else value


@main.group("config")
def config_group():
    """Read and write ~/.matcreator/config.yaml settings."""


@config_group.command("set")
@click.argument("assignment", metavar="KEY=VALUE")
def config_set(assignment: str):
    """Set a config value.

    KEY uses dotted notation: section.field (e.g. llm.api_key, bohrium.email).
    """
    if "=" not in assignment:
        raise click.UsageError("Expected KEY=VALUE format (e.g. llm.api_key=sk-xxx)")
    key, _, value = assignment.partition("=")
    key = key.strip()
    value = value.strip()
    if not key:
        raise click.UsageError("Key must not be empty.")
    from matcreator.config import set_config_value
    set_config_value(key, value)
    display = _mask(key, value)
    click.echo(f"Set {key} = {display}")


@config_group.command("get")
@click.argument("key")
@click.option("--reveal", is_flag=True, default=False, help="Show secret values in plain text.")
def config_get(key: str, reveal: bool):
    """Print the value of KEY from ~/.matcreator/config.yaml."""
    from matcreator.config import get_config_value
    value = get_config_value(key)
    display = value if reveal else _mask(key, value)
    if display:
        click.echo(display)
    else:
        click.echo("(not set)", err=True)
        raise SystemExit(1)


@config_group.command("show")
@click.option("--reveal-secrets", is_flag=True, default=False,
              help="Show API keys and passwords in plain text.")
def config_show(reveal_secrets: bool):
    """Print all settings from ~/.matcreator/config.yaml."""
    from matcreator.config import load_config, SENSITIVE_YAML_KEYS
    cfg = load_config()
    if not cfg:
        click.echo("No config file found at ~/.matcreator/config.yaml")
        return

    def _mask_cfg(section: str, key: str, value: object) -> object:
        dotted = f"{section}.{key}"
        if not reveal_secrets and dotted in SENSITIVE_YAML_KEYS and value:
            return "***"
        return value

    for section, content in cfg.items():
        if isinstance(content, dict):
            click.echo(f"{section}:")
            for k, v in content.items():
                click.echo(f"  {k}: {_mask_cfg(section, k, v)}")
        else:
            click.echo(f"{section}: {content}")


if __name__ == "__main__":
    main()
