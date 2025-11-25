Modified from [ABACUS agent tools](https://github.com/deepmodeling/ABACUS-agent-tools.git)  for batch execution. 

#### Environment variables
You may need to set following variables in `.env`
```bash
ABACUS_SERVER_WORK_PATH=/tmp/abacus_server
BOHRIUM_USERNAME=name
BOHRIUM_PASSWORD=password
BOHRIUM_PROJECT_ID=11111
BOHRIUM_ABACUS_IMAGE=registry.dp.tech/dptech/abacus-stable:LTSv3.10
BOHRIUM_ABACUS_MACHINE=c32_m64_cpu
BOHRIUM_ABACUS_COMMAND="OMP_NUM_THREADS=1 mpirun -np 16 abacus"
ABACUSAGENT_SUBMIT_TYPE=bohrium
ABACUS_COMMAND=abacus
ABACUS_PP_PATH="/home/ruoyu/dev/SG15_ONCV_v1.0_upf"
ABACUS_ORB_PATH="/home/ruoyu/dev/SG15-Version1p0__AllOrbitals-Version2p0"
```