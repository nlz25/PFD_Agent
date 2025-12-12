For vasp tool
1. 安装Bohrium CLI，可以参考链接https://docs.bohrium.com/docs/bohrctl/install#%E5%87%86%E5%A4%87%E5%B7%A5%E4%BD%9C。

2.export PMG_VASP_PSP_DIR=/path/to/your/POTCARS/

例如/home/user/pbe/PMG_VASP_PSP_DIR/Cu/POTCAR,则export PMG_VASP_PSP_DIR=/home/user/pbe

3.在tool/vasp文件夹下配置.env文件，


VASP_SERVER_WORK_PATH= /tmp/vasp_server

BOHRIUM_USERNAME=

BOHRIUM_PASSWORD= 

BOHRIUM_PROJECT_ID= 

BOHRIUM_VASP_IMAGE= registry.dp.tech/dptech/prod-15454/vasp:5.4.4

BOHRIUM_VASP_MACHINE= c32_m64_cpu

BOHRIUM_VASP_COMMAND= source /opt/intel/oneapi/setvars.sh && mpirun -n 16 vasp_std

VASPAGENT_SUBMIT_TYPE= bohrium


4.uv run server.py --port 50004
