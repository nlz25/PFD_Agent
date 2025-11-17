# MatCreator

A Multi-agent network combines AI-oriented database and complex computational materials workflow.


## Installation

```bash
# Create and activate an environment (optional but recommended)
conda create -n pfd python=3.12 -y
conda activate matcreator

# From the project root
pip install -U pip
pip install -e .
```

## Quick start
### Set up MCP servers
`MatCreator` takes a modular design principle: tools are standalone mini-packages that can be instantiated in a containerized enviroments. For example, to set up a `mcp` server for `ABACUS` DFT software in an seperated environment, `uv run` the script: 

```bash
cd tools/abacus
uv sync 

nv run server.py --port 50001
```
Repeat this for all the tools.


### Running agent networks

```bash
cd agents
adk web
```
This would set up the `MatCreator` agent network. You can tune the LLM model and communication settings for the agents.


#### Connecting a client agent (SSE)

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

#### Configuration file

You may need to configure the server and tools via a JSON file. For example, when running `ABACUS` server, environment file at `~/.abacus_server/env.json` would be accessed:

- If the file does not exist, it will be created on first run with sensible defaults.
- You can edit this file to set the same variables shown above (and more) in one place.
- Precedence: CLI flags > process environment variables > values in `~/.pfd_agent/env.json`.
- After editing the file, restart the server for changes to take effect.

Example `~/.pfd_agent/env.json` snippet:

```json
{
  "ABACUS_SERVER_WORK_PATH": "/tmp/abacus_server",
  "BOHRIUM_USERNAME": "",
  "BOHRIUM_PASSWORD": "",
  "BOHRIUM_PROJECT_ID": "",
  "BOHRIUM_ABACUS_IMAGE": "registry.dp.tech/dptech/abacus-stable:LTSv3.10",
  "BOHRIUM_ABACUS_MACHINE": "c32_m64_cpu",
  "BOHRIUM_ABACUS_COMMAND": "OMP_NUM_THREADS=1 mpirun -np 16 abacus",
  "ABACUSAGENT_SUBMIT_TYPE": "bohrium",
  "ABACUS_COMMAND": "abacus",
  "ABACUS_PP_PATH": "/home/ruoyu/dev/SG15_ONCV_v1.0_upf",
  "ABACUS_ORB_PATH": "/home/ruoyu/dev/SG15-Version1p0__AllOrbitals-Version2p0",
  "ABACUS_SOC_PP_PATH": "",
  "ABACUS_SOC_ORB_PATH": "",
  "PYATB_COMMAND": "OMP_NUM_THREADS=1 pyatb",
}
```

Notes:
- Do not commit secrets (e.g., API keys) to version control. Prefer environment variables in CI.
- CLI flags always take priority if you need a one-off override.

