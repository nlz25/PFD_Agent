# PFD-Agent

A toolkit that exposes materials workflow tools (database, exploration/MD, DFT/ABACUS, training) over an MCP server and a CLI entry point.

## Requirements

- Linux
- Python 3.12+
- Recommended: a virtualenv or Conda environment

## Installation

```bash
# Create and activate an environment (optional but recommended)
conda create -n pfd python=3.12 -y
conda activate pfd

# From the project root
pip install -U pip
pip install -e .
```

This installs the `pfd-agent` CLI entry point.

## Quick start

Run the MCP server locally over SSE with default settings:

```bash
export PFD_AGENT_HOST=127.0.0.1
export PFD_AGENT_PORT=50001
export PFD_AGENT_TRANSPORT=sse

pfd-agent
```

You should see logs like:
- “Loading tools…”
- “✅ Successfully loaded: pfd_agent_tool.modules.db” (and others)
- “Address: 127.0.0.1:50001/sse”

Stop the server with Ctrl+C.

## CLI usage

```bash
pfd-agent --help
```

Options:
- `--transport {sse,streamable-http}` Transport protocol (default: `sse`)
- `--model {fastmcp,dp}` Backend model runner (default is determined by env)
- `--port <int>` Server port (default: 50001)
- `--host <str>` Server host (default: localhost)
- `--screen-modules <names...>` Exclude specific modules from loading (see Modules)

Example: run on a custom port and host
```bash
pfd-agent --host 0.0.0.0 --port 51000 --transport sse
```

You can also set these via env vars (the CLI flags override env):
```bash
export PFD_AGENT_HOST=0.0.0.0
export PFD_AGENT_PORT=51000
export PFD_AGENT_TRANSPORT=sse
pfd-agent
```

## Modules

By default, the server loads all available modules:
- db
- expl
- dft
- train

To exclude modules, use `--screen-modules`:
```bash
# Load everything except 'expl' and 'train'
pfd-agent --screen-modules expl train
```

## Connecting a client agent (SSE)

Point your client’s MCP toolset at the server’s SSE endpoint:

```python
from google.adk.tools.mcp_tool.mcp_session_manager import SseServerParams
from google.adk.tools.mcp_tool import McpToolset

toolset = McpToolset(
    connection_params=SseServerParams(
        url="http://127.0.0.1:50001/sse"
    )
)
```

For Streamable HTTP:
```python
from google.adk.tools.mcp_tool.mcp_session_manager import StreamableHTTPServerParams

toolset = McpToolset(
    connection_params=StreamableHTTPServerParams(
        url="http://127.0.0.1:50001/mcp"
    )
)
```

## Useful environment variables

- `PFD_AGENT_HOST` Host to bind (e.g., `127.0.0.1`)
- `PFD_AGENT_PORT` Port to bind (e.g., `50001`)
- `PFD_AGENT_TRANSPORT` `sse` or `streamable-http`
- Module-specific:
  - `ASE_DB_PATH` Path to your ASE database file(s) for db tools
  - `BASE_MODEL_PATH` Default model path for exploration tools (used when not provided explicitly)

Example:
```bash
export ASE_DB_PATH=/data/ase/structures.db
export BASE_MODEL_PATH=/models/dpa/model.pt
```

## Configuration file (~/.pfd_agent/env.json)

You can also configure the server and tools via a JSON file at `~/.pfd_agent/env.json`.

- If the file does not exist, it will be created on first run with sensible defaults.
- You can edit this file to set the same variables shown above (and more) in one place.
- Precedence: CLI flags > process environment variables > values in `~/.pfd_agent/env.json`.
- After editing the file, restart the server for changes to take effect.

Example `~/.pfd_agent/env.json` snippet:

```json
{
  "PFD_AGENT_WORK_PATH": "/tmp/pfd_agent/",
  "PFD_AGENT_TRANSPORT": "sse",
  "PFD_AGENT_HOST": "127.0.0.1",
  "PFD_AGENT_PORT": "50001",
  "PFD_AGENT_MODEL": "fastmcp",

  "ASE_DB_PATH": "/data/ase/structures.db",
  "BASE_MODEL_PATH": "/models/dpa/model.pt",

  "ABACUS_COMMAND": "abacus",
  "ABACUS_PP_PATH": "/opt/abacus/pp",
  "ABACUS_ORB_PATH": "/opt/abacus/orb"
}
```

Notes:
- Do not commit secrets (e.g., API keys) to version control. Prefer environment variables in CI.
- CLI flags always take priority if you need a one-off override.

## Verifying tool load

On startup you should see:
- “✅ Successfully loaded: pfd_agent_tool.modules.db”
- “✅ Successfully loaded: pfd_agent_tool.modules.expl”
- “✅ Successfully loaded: pfd_agent_tool.modules.dft”
- “✅ Successfully loaded: pfd_agent_tool.modules.train”

If a module fails to load, you’ll see:
- “⚠️ Failed to load pfd_agent_tool.modules.<name>: <error>”

## Troubleshooting

- Port already in use:
  - Pick a new port: `pfd-agent --port 51001`
- No tools visible in client:
  - Ensure the server is running and the client URL matches `/sse` (or `/mcp` for streamable-http)
  - Confirm modules loaded successfully in server logs
- DB tools cannot find your database:
  - Set `ASE_DB_PATH` to a valid `.db` file
- Exploration tools complain about model path:
  - Pass a `model_path` explicitly in calls or set `BASE_MODEL_PATH`

## Uninstall / upgrade

```bash
pip uninstall pfd-agent -y
pip install -e .
```

## Development tips

- To see fewer modules while iterating:
  ```bash
  pfd-agent --screen-modules dft train
  ```
- Use a dedicated terminal window for server logs while driving the agent from an IDE or notebook.
