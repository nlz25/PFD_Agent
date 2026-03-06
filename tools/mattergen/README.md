## Environment variables
Please set following variables
```env
MATTERGEN_MODEL_ROOT = "/opt/mattergen/checkpoints/"
MATTERGEN_SERVER_WORK_PATH = "/tmp/mattergen_server"
MATTERGEN_SUBMIT_TYPE="bohrium"  # "local" or "bohrium" or "debug"
BOHRIUM_USERNAME="xxxx"
BOHRIUM_PASSWORD="xxxxx"
BOHRIUM_PROJECT_ID="xxxxx"
BOHRIUM_MATTERGEN_IMAGE="registry.dp.tech/dptech/dp/native/prod-25997/mattergen:1.0.0"
BOHRIUM_MATTERGEN_MACHINE="1 * NVIDIA V100_32g"
BOHRIUM_PYTHON_PACKAGES="pymatgen,ase,pydflow,python-dotenv,mattergen_tool"
```

## Virutual environment installation
To run locally, please install mattergen via venv.
Follow
```bash
cd /opt

# Download from github.
git clone https://github.com/microsoft/mattergen.git
cd mattergen

# Install via uv.
pip install uv
uv venv .venv --python 3.10 
source .venv/bin/activate
uv pip install -e .

# Install git-lfs.
sudo apt install git-lfs
git lfs install

# Download desired conditional model using git-lfs, for example:
git lfs pull -I checkpoints/chemical_system_energy_above_hull --exclude="" 
```
Then set environment variable `MATTERGEN_MODEL_ROOT` to the path of the model checkpoints,
such as /opt/mattergen/checkpoints/,
and `MATTERGEN_VENV_ROOT` to the path you installed mattergen venv, such as /opt/mattergen/.venv.

`MATTERGEN_VENV_ROOT` only required when `MATTERGEN_SUBMIT_TYPE` is `local` or `debug`.

Conditional models not downloaded yet cannot be used for training and generation.