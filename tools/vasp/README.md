## set bohrium parameters (tool/vasp/.env)

BOHRIUM_USERNAME=

BOHRIUM_PASSWORD=

BOHRIUM_PROJECT_ID= 

BOHRIUM_VASP_IMAGE= 

BOHRIUM_VASP_MACHINE= c64_m256_cpu

BOHRIUM_VASP_COMMAND= source /opt/intel/oneapi/setvars.sh && mpirun -n 16 vasp_std

VASPAGENT_SUBMIT_TYPE= bohrium

VASP_SERVER_WORK_PATH= /tmp/vasp_server

## set POTCAR

export PMG_VASP_PSP_DIR=/path/to/your/POTCARS/

## set INCAR(tool/vasp/config.yaml)

work_dir: "/tmp/vasp_server"

VASP_default_INCAR:

  relaxation:



  scf_nsoc:




  nscf_nsoc: