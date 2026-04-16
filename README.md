
# MatCreator

MatCreator is a **skill-based, agentic platform** for computational material science tasks, with a focus on Machine Learning Force Field (MLFF) generation and application. It would evolve with users by experience accumulation and creation of new skills. 

## Quick start
### Installation
```bash
# Create and activate an environment with uv (optional but recommended)

pip install uv
uv venv .venv --python 3.12

source .venv/bin/activate
uv pip install -e .
```

### Recommended Workflow: WSL + uv for Virtual Environment Deployment

We highly recommend using WSL (Windows Subsystem for Linux) with uv to deploy local development virtual environments. WSL provides a native Linux environment seamlessly integrated with Windows, enabling access to Linux tools. As a fast, lightweight Python package manager, uv creates isolated environments to avoid dependency conflicts, ideal for Python applications like Streamlit.

#### Why Use pipx for uv in WSL

In WSL, system Python is managed by apt and is a core system component. PEP 668 prohibits direct pip installation to avoid breaking system dependencies. Pipx is ideal for tools like uv: it creates isolated virtual environments for global access without polluting system Python.

#### Solution: Install uv Using pipx

As a command-line tool, uv can be installed via pipx, which creates an independent virtual environment for global use. Run these commands in the WSL terminal:

```bash
# 1. Install pipx
sudo apt update && sudo apt install pipx -y

# 2. Initialize pipx (add to system PATH)
pipx ensurepath

# 3. Restart terminal, then install uv
pipx install uv
```

### Running agent networks
#### Setting environments
Before the first run, create the `agents/MatCreator/.env` file and configure your model API credentials (additional environment variables may be required for some functionalities).

```bash
touch agents/MatCreator/.env
```

An example content of `.env`:

```env
LLM_MODEL= "MODEL_TYPE"
LLM_API_KEY="API_KEYS"
LLM_BASE_URL="BASE_URL"

# SKILL_RELATED_ENV
# ...
```

If you prefer different LLM models for sub-agents, you can override the default setting at the `.env` file within sub-agents directories. 

#### Web UI (Recommended)
A simple web UI that supports artifact upload/download, structure visualization and scientific plotting. The web UI server can be started with the following command:
```bash
#We strongly recommend running the following script in the background and saving the runtime logs.
 
nohup matcreator api-server >> server.log &
nohup streamlit run web/streamlit_app.py >> web.log &
```
![The web UI for MatCreator](docs/images/agent_plot.png)

#### Non-interactive CLI mode

Run the agent on a single prompt without starting any server:

```bash
# Inline prompt
matcreator run -p "Build a silicon FCC structure"

# Prompt from a file
matcreator run -f prompt.txt

# Save the answer to a file
matcreator run -p "Build a silicon FCC structure" -o result.txt

# Full structured JSON output (includes turn count, duration, etc.)
matcreator run -p "Build a silicon FCC structure" --output-format json -o result.json

# Override the workspace directory
matcreator run --workspace /data/my_workspace -p "Build a silicon FCC structure"
# or via environment variable
MATCLAW_WORKSPACE=/data/my_workspace matcreator run -p "Build a silicon FCC structure"
```

Each run creates a session directory under `<workspace>/sessions/<session-id>/` where any files produced by the agent are saved.

#### Default adk web server (old style)

```bash
matcreator run web
```
This would set up the MatCreator agent network through the default `adk web` server. You can tune the LLM model and communication settings for the agents.

The default agent workspace is located at `agents/MatCreator/.workspace`, where skills, memory, etc., are stored.

## Skills
MatCreator follows a modular design principle: skills are text files that define metadata, procedures and workflows. Some skills may require specialized tools (configured by `$PROJECT/agents/MatCreator/tools.py`), and some of them, e.g. tools for DFT calculations, may be hosted on MCP servers.

> The default domain-based computational materials datasets is located at `database/domain_datasets.tar.gz`, which should be extracted for database skill usage. (See `tools/database/README.md`)

> Check the `README.md` in `skills/$SKILL` if you really wanna use them. 


> **Note — transitioning from MCP servers to skills:** MatCreator is progressively moving tool logic out of dedicated MCP servers and into self-contained skills. A skill bundles its own workflow instructions, helper scripts, and configuration alongside the `.md` file, so it can be run with only a general-purpose shell/Python tool rather than a running server process. If a capability you previously used via an MCP server is no longer listed under `tools/`, check `agents/MatCreator/knowledge/skills/` — it may have been migrated to a skill. MCP servers are retained only for tools that genuinely require a persistent service (e.g. a remote job scheduler or a database backend).


###  Server setup (Optional)
For example, to set up a `mcp` server for `ABACUS` DFT software, `uv run` the script: 

```bash
cd tools/abacus
uv sync 

uv run server.py --port 50001
```
 You may need to set environment variables specific to the mcp server at `tools/$TOOLNAME/.env`, which can be referenced in `tools/$TOOLNAME/README.md`

### Customize skills

Skills are Markdown files with a YAML frontmatter block (declaring `name`, `description`, `tools`, and `dependent_skills`) followed by a plain-text instruction body. The active loader discovers any workspace directory that contains a `SKILL.md` file, including nested directories such as `skills/mattergen/mattergen_generation/SKILL.md`. MatCreator loads skills from two locations in order:

1. **Built-in skills** — shipped with the package under `agents/MatCreator/knowledge/skills/`. Skills can be placed as flat `<name>.md` files or in a subdirectory `<name>/<name>.md`; the subdirectory form takes precedence.
2. **Workspace overlay** — your personal skills under `$MATCLAW_WORKSPACE/skills/` (defaults to `.workspace/` in the project root). Any skill here with the same name overrides the built-in version.

To customize a skill manually, copy its skill directory into your workspace `skills/` directory and edit the contained `SKILL.md`. To add a new skill, create a new `skills/<name>/SKILL.md` file following the same frontmatter format.

The agent can also create and update skills on its own. During a session, the thinking agent can call built-in tools to scaffold a new skill file, write updated content to an existing one, or list what skills are currently available — letting the system accumulate knowledge automatically over time.
