Modified from [ABACUS agent tools](https://github.com/deepmodeling/ABACUS-agent-tools.git)  for batch execution. It also incorporates tools implemented in the original projects.

You can download the official pseudopotential and LCAO orbital file for ABACUS agent by
```bash
wget https://store.aissquare.com/datasets/af21b5d9-19e6-462f-ada1-532f47f165f2/ABACUS-APNS-PPORBs-v1.zip&& unzip -u ABACUS-APNS-PPORBs-v1.zip
```

#### Environment variables
You may need to set following variables in `.env`
```bash
ABACUS_SERVER_WORK_PATH=/tmp/abacus_server
BOHRIUM_USERNAME=name
BOHRIUM_PASSWORD=password
BOHRIUM_PROJECT_ID=11111
BOHRIUM_ABACUS_IMAGE=registry.dp.tech/dptech/abacus-stable:LTSv3.10
BOHRIUM_ABACUS_MACHINE=c16_m32_cpu
BOHRIUM_ABACUS_COMMAND="OMP_NUM_THREADS=1 mpirun -np 16 abacus"
ABACUSAGENT_SUBMIT_TYPE=bohrium
ABACUS_COMMAND=abacus
ABACUS_PP_PATH="$PATH_TO_SERVER/apns-pseudopotentials-v1"
ABACUS_ORB_PATH="$PATH_TO_SERVER/apns-orbitals-efficiency-v1"
```