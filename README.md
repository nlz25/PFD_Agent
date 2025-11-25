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
`MatCreator` takes a modular design principle: tools are standalone mini-packages that can be instantiated in a isolated enviroments. For example, to set up a `mcp` server for `ABACUS` DFT software, `uv run` the script: 

```bash
cd tools/abacus
uv sync 

nv run server.py --port 50001 --model dp
```
If you prefer `bohr-agent-sdk` wrapper which supports submitting tool job to `Bohrium` platform, set `--model dp`; otherwise you get standard `FastMCP` experience. You may need to set environment variables specific to the mcp server at `tools/$TOOLNAME/.env`, which can be referenced in `tools/$TOOLNAME/README.md`

Repeat this for all the servers in the `tools` directory. If required dependencies have been already installed, you can also directly run `python server.py --port 50001 --model dp`.  


### Running agent networks
#### Setting constants
Populate `agents/MatCreator/input.json` with your model and Bohrium credentials (or point the
`MATCREATOR_CONFIG_PATH` environment variable to another JSON file). Environment variables of the
same name always override what is stored in the JSON file.

```json
{
    "LLM_MODEL": "",
    "LLM_API_KEY": "",
    "LLM_BASE_URL": "",
    "BOHRIUM_USERNAME": "",
    "BOHRIUM_PASSWORD": "",
    "BOHRIUM_PROJECT_ID": 11111
}
```

#### Connecting a client agent (SSE)

Point your client’s MCP toolset at the server’s SSE endpoint:
```python
## standard FastMCP server
from google.adk.tools.mcp_tool.mcp_session_manager import SseServerParams
from google.adk.tools.mcp_tool import McpToolset
from ..constants import LLM_MODEL, LLM_API_KEY, LLM_BASE_URL, 
model_name = os.environ.get("LLM_MODEL", LLM_MODEL)
model_api_key = os.environ.get("LLM_API_KEY", LLM_API_KEY)
model_base_url = os.environ.get("LLM_BASE_URL", LLM_BASE_URL)

toolset = McpToolset(
    connection_params=SseServerParams(
        url="$HOST:$PORT/$TRANSPORT" # example:"http://127.0.0.1:50001/sse"
    )
)
```

```python
## bohr-compatible, recommended for heavy calculation tasks
from google.adk.tools.mcp_tool.mcp_session_manager import SseServerParams
from dp.agent.adapter.adk import CalculationMCPToolset
from ..constants import LLM_MODEL, LLM_API_KEY, LLM_BASE_URL, BOHRIUM_USERNAME, BOHRIUM_PASSWORD, BOHRIUM_PROJECT_ID

# Set the secret key in ~/.abacusagent/env.json or as an environment variable, or modify the code t
model_name = os.environ.get("LLM_MODEL", LLM_MODEL)
model_api_key = os.environ.get("LLM_API_KEY", LLM_API_KEY)
model_base_url = os.environ.get("LLM_BASE_URL", LLM_BASE_URL)
bohrium_username = os.environ.get("BOHRIUM_USERNAME", BOHRIUM_USERNAME)
bohrium_password = os.environ.get("BOHRIUM_PASSWORD", BOHRIUM_PASSWORD)
bohrium_project_id = int(os.environ.get("BOHRIUM_PROJECT_ID", BOHRIUM_PROJECT_ID))

executor = {
    "bohr": {
        "type": "dispatcher",
        "machine": {
            "batch_type": "Bohrium",
            "context_type": "Bohrium",
            "remote_profile": {
                "email": bohrium_username,
                "password": bohrium_password,
                "program_id": bohrium_project_id,
                "input_data": {
                    "image_name": "registry.dp.tech/dptech/dp/native/prod-26745/matcreator:0.0.1",
                    "job_type": "container",
                    "platform": "ali",
                    "scass_type": "1 * NVIDIA V100_16g",
                },
            },
        }
    },
    "local": {"type": "local",}
}


EXECUTOR_MAP = {
    "run_molecular_dynamics": executor["bohr"],
    "optimize_structure": executor["bohr"],
    "training": executor["bohr"],
    "ase_calculation": executor["bohr"],
}

STORAGE = {
    "type": "https",
    "plugin":{
        "type": "bohrium",
        "username": bohrium_username,
        "password": bohrium_password,
        "project_id": bohrium_project_id,
    }
}

toolset = CalculationMCPToolset(
    connection_params=SseServerParams(
        url="$HOST:$PORT/$TRANSPORT"
    ),
    executor=executor["local"], # default executor type
    executor_map = EXECUTOR_MAP, # executor type override, these tools would be submitted to bohrium
    storage=STORAGE, # access artifact stored in bohrium storage.
)

```

#### Starting agent

```bash
cd agents
adk web
```
This would set up the `MatCreator` agent network. You can tune the LLM model and communication settings for the agents.


