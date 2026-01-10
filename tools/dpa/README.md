## Environment variables
Please set following variables
```env
DPA_MODEL_PATH = "/home/ruoyu/dev/PFD-Agent/.tests/dpa/DPA2_medium_28_10M_rc0.pt"
DPA_SERVER_WORK_PATH = "/tmp/dpa_server"
DPA_SUBMIT_TYPE="bohrium"  # "local" or "bohrium"
BOHRIUM_USERNAME="xxxx"
BOHRIUM_PASSWORD="xxxxx"
BOHRIUM_PROJECT_ID="xxxxx"
BOHRIUM_DPA_IMAGE="registry.dp.tech/dptech/deepmd-kit:3.1.0-cuda12.1"
BOHRIUM_DPA_MACHINE="1 * NVIDIA V100_32g"
```