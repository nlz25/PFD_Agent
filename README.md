
# MatCreator

MatCreator is a **skill-based, agentic platform** for computational material science tasks, with a focus on Machine Learning Force Field (MLFF) generation and application. It would evolve with users by experience accumulation and creation of new skills. 

## Quick start
### Installation
```bash
# Create and activate an environment (optional but recommended)
conda create -n pfd python=3.12 -y
conda activate pfd

# From the project root
pip install -U pip
pip install -e .
```

### Set up MCP servers for tools
MatCreator follows a modular design principle: skills are text files that define metadata, procedures and workflows. Some skills may require specialized tools (configured by `$PROJECT/agents/MatCreator/tools.py`), and some of them, e.g. tools for DFT calculations, may be hosted on MCP servers. 

#### Manual server setup
For example, to set up a `mcp` server for `ABACUS` DFT software, `uv run` the script: 

```bash
cd tools/abacus
uv sync 

uv run server.py --port 50001
```
If you prefer `bohr-agent-sdk` wrapper which supports submitting tool job to `Bohrium` platform, set `--model dp`; otherwise you get standard `FastMCP` experience. You may need to set environment variables specific to the mcp server at `tools/$TOOLNAME/.env`, which can be referenced in `tools/$TOOLNAME/README.md`

#### Automated Server Management

For convenience, you can use the provided startup script to manage all MCP servers at once:

```bash
# Start all MCP servers
python script/start_mcp_servers.py start

# Check server status
python script/start_mcp_servers.py status

# Stop all servers
python script/start_mcp_servers.py stop

# Start specific servers only
python script/start_mcp_servers.py start database dpa vasp

# Start servers within current python environment
python script/start_mcp_servers.py start database dpa vasp --no-uv

# Stop specific servers only
python script/start_mcp_servers.py stop database dpa
```

The script automatically handles port assignments and logging for each server, making it easier to manage the entire MCP server ecosystem in your local development environment. The server log would be sync at `logs/mcp_servers` under project root.

### Running agent networks
#### Setting constants
Populate `agents/MatCreator/.env` with your model API credentials.

```env
LLM_MODEL= "MODEL_TYPE"
LLM_API_KEY="API_KEYS",
LLM_BASE_URL="BASE_URL",
```

If you prefer different LLM models for sub-agents, you can override the default setting at the `.env` file within sub-agents directories. 

#### Starting agent

```bash
python script/start_agent.py web
```
This would set up the MatCreator agent network through the default `adk web` server. You can tune the LLM model and communication settings for the agents.

The default agent workspace is located at `agents/MatCreator/.workspace`, where skills, memory, etc., are stored. 

#### Web UI
A simple web UI that supports artifact upload/download, structure visualization and scientific plotting. The web UI server can be started with the following command:
```bash
python script/start_agent.py api-server
streamlit run web/streamlit_app.py 
```
![The web UI for MatCreator](docs/images/agent_plot.png)

### Customize skills

Skills are Markdown files with a YAML frontmatter block (declaring `name`, `description`, `tools`, and `dependent_skills`) followed by a plain-text instruction body. MatCreator loads skills from two locations in order:

1. **Built-in skills** — shipped with the package under `agents/MatCreator/knowledge/skills/`. Skills can be placed as flat `<name>.md` files or in a subdirectory `<name>/<name>.md`; the subdirectory form takes precedence.
2. **Workspace overlay** — your personal skills under `$MATCLAW_WORKSPACE/skills/` (defaults to `.workspace/` in the project root). Any skill here with the same name overrides the built-in version.

To customize a skill manually, copy its `.md` file into your workspace `skills/` directory and edit it. To add a new skill, create a new `skills/<name>/<name>.md` file following the same frontmatter format.

The agent can also create and update skills on its own. During a session, the thinking agent can call built-in tools to scaffold a new skill file, write updated content to an existing one, or list what skills are currently available — letting the system accumulate knowledge automatically over time.
