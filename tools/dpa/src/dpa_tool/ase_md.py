import json
import time
import logging
import os
import shutil
import glob
from pathlib import Path
from typing import Optional, Dict, Any, Union, Sequence, List, Tuple, Literal
from dataclasses import dataclass, asdict, field

import ase
import numpy as np
from ase import Atoms, units
from ase.io import read,write
from ase.io.trajectory import Trajectory
from ase.md.langevin import Langevin
from ase.md.nose_hoover_chain import NoseHooverChainNVT
from ase.md.nptberendsen import NPTBerendsen
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.analysis import DiffusionCoefficient
from ase.optimize import LBFGS
from ase.filters import UnitCellFilter, ExpCellFilter

from deepmd.calculator import DP
import dflow
from dflow import (
    InputParameter,
    InputArtifact,
    Inputs,
    OutputParameter,
    OutputArtifact,
    Outputs,
    Step,
    Steps,
    Workflow,
    upload_artifact,
    download_artifact,
    argo_len,
    argo_sequence,
)
from dflow.python import (
    OP,
    OPIO,
    Parameter,
    OPIOSign,
    PythonOPTemplate,
    Artifact,
    TransientError,
    Slices
)

from .utils import set_directory, generate_work_path


ase_conf_name = "structure.extxyz"
ase_input_name = "ase.json"
ase_log_name = "ase.log"
ase_traj_name = "traj.traj"

import logging
logger = logging.getLogger(__name__)



def _log_progress(atoms, dyn):
    """Log simulation progress"""
    epot = atoms.get_potential_energy()
    ekin = atoms.get_kinetic_energy()
    temp = ekin / (1.5 * len(atoms) * units.kB)
    logger.info(f"Step: {dyn.nsteps:6d}, E_pot: {epot:.3f} eV, T: {temp:.2f} K")


def _run_md_stage(atoms, stage, save_interval_steps, traj_file, seed, stage_id):
    """Run a single MD simulation stage"""
    temperature_K = stage.get('temperature_K', None)
    pressure = stage.get('pressure', None)
    mode = stage['mode']
    runtime_ps = stage['runtime_ps']
    timestep_ps = stage.get('timestep', 0.0005)  # default: 0.5 fs
    tau_t_ps = stage.get('tau_t', 0.01)         # default: 10 fs
    tau_p_ps = stage.get('tau_p', 0.1)          # default: 100 fs

    timestep_fs = timestep_ps * 1000  # convert to fs
    total_steps = int(runtime_ps * 1000 / timestep_fs)

    # Initialize velocities if first stage with temperature
    if stage_id == 1 and temperature_K is not None:
        MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_K, 
                                rng=np.random.RandomState(seed))
        from ase.md.velocitydistribution import Stationary, ZeroRotation
        Stationary(atoms)
        ZeroRotation(atoms)

    # Choose ensemble
    if mode == 'NVT' or mode == 'NVT-NH':
        # Use NoseHooverChain for NVT by default
        dyn = NoseHooverChainNVT(
            atoms,
            timestep=timestep_fs * units.fs,
            temperature_K=temperature_K,
            tdamp=tau_t_ps * 1000 * units.fs
        )
    elif mode == 'NVT-Berendsen':
        from ase.md.nvtberendsen import NVTBerendsen
        dyn = NVTBerendsen(
            atoms,
            timestep=timestep_fs * units.fs,
            temperature_K=temperature_K,
            taut=tau_t_ps * 1000 * units.fs
        )
    elif mode == 'NVT-Andersen':
        from ase.md.andersen import Andersen
        dyn = Andersen(
            atoms,
            timestep=timestep_fs * units.fs,
            temperature_K=temperature_K,
            friction=1.0 / (tau_t_ps * 1000 * units.fs),
            rng=np.random.RandomState(seed)
        )
    elif mode == 'NVT-Langevin' or mode == 'Langevin':
        dyn = Langevin(
            atoms,
            timestep=timestep_fs * units.fs,
            temperature_K=temperature_K,
            friction=1.0 / (tau_t_ps * 1000 * units.fs),
            rng=np.random.RandomState(seed)
        )
    elif mode == 'NPT-aniso' or mode == 'NPT-tri':
        from ase.md.npt import NPT
        if mode == 'NPT-aniso':
            mask = np.eye(3, dtype=bool)
        elif mode == 'NPT-tri':
            mask = None
        else:
            raise ValueError(f"Unknown NPT mode: {mode}")

        if pressure is None:
            raise ValueError("Pressure must be specified for NPT simulations")

        dyn = NPT(
            atoms,
            timestep=timestep_fs * units.fs,
            temperature_K=temperature_K,
            externalstress=pressure * units.GPa,
            ttime=tau_t_ps * 1000 * units.fs,
            pfactor=tau_p_ps * 1000 * units.fs,
            mask=mask
        )
    elif mode == 'NVE':
        from ase.md.verlet import VelocityVerlet
        dyn = VelocityVerlet(
            atoms,
            timestep=timestep_fs * units.fs
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Prepare trajectory file
    traj_path = Path(traj_file)
    os.makedirs(traj_path.parent, exist_ok=True)
    if traj_path.exists():
        os.remove(traj_file)

    def _write_frame():
        """Write current frame to trajectory"""
        results = atoms.calc.results
        energy = results.get("energy", atoms.get_potential_energy())
        forces = results.get("forces", atoms.get_forces())
        stress = results.get("stress", atoms.get_stress(voigt=False))

        if np.isnan(energy).any() or np.isnan(forces).any() or np.isnan(stress).any():
            raise ValueError("NaN detected in simulation outputs. Aborting trajectory write.")

        new_atoms = atoms.copy()
        new_atoms.info["energy"] = energy
        new_atoms.arrays["force"] = forces
        if "occupancy" in atoms.info:
            del atoms.info["occupancy"]
        if "spacegroup" in atoms.info:
            del atoms.info["spacegroup"] 

        write(traj_file, new_atoms, format="extxyz", append=True)

    # Attach callbacks
    dyn.attach(_write_frame, interval=save_interval_steps)
    dyn.attach(lambda: _log_progress(atoms, dyn), interval=10)

    logger.info(f"[Stage {stage_id}] Starting {mode} simulation: T={temperature_K} K"
                + (f", P={pressure} GPa" if pressure is not None else "")
                + f", steps={total_steps}, dt={timestep_ps} ps")

    # Run simulation
    dyn.run(total_steps)
    logger.info(f"[Stage {stage_id}] Finished simulation. Trajectory saved to: {traj_file}\n")

    return atoms


class PrepareMDTaskGroupOP(OP):
    """OP that wraps ``_prepare_task_group_impl`` for DFlow."""

    @classmethod
    def get_input_sign(cls):
        return OPIOSign({"structure_paths": Artifact(List[Path])})

    @classmethod
    def get_output_sign(cls):
        return OPIOSign({"task_paths": Artifact(List[Path]),"task_idx":Parameter(List[str])})

    @OP.exec_sign_check
    def execute(self, ip: OPIO) -> OPIO:
        structure_paths=ip["structure_paths"]
        atoms_ls=[]
        if isinstance(structure_paths,str):
            structure_paths = [structure_paths]
        
        try:
            for structure_path in structure_paths:
                atoms_ls.extend(read(structure_path,index=':'))
        except Exception as e:
            logger.exception("Failed to read structures: %s",e)
            raise
    
        task_dir_ls=[]
        task_idx=[]
        for i, atoms in enumerate(atoms_ls):
            task_dir = Path(f"task.{i:05d}") 
            try:
                task_dir.mkdir(parents=True, exist_ok=True)
                struct_file = task_dir / ase_conf_name
                write(str(struct_file),atoms)
            except Exception:
                logger.exception("Failed to write structure for %s", task_dir)
                continue
            task_dir_ls.append(task_dir)
            task_idx.append(f"{i:05d}")
            #task_idx.append(i)
        return OPIO({"task_paths": task_dir_ls, "task_idx": task_idx})


class RunMDTask(OP):
    r"""Execute a ASE MD task.

    A working directory named `task_name` is created. All input files
    are copied or symbol linked to directory `task_name`. The LAMMPS
    command is exectuted from directory `task_name`. The trajectory
    and the model deviation will be stored in files `op["traj"]` and
    `op["model_devi"]`, respectively.

    """

    @classmethod
    def get_input_sign(cls):
        return OPIOSign(
            {
                "config": Parameter(Dict[str, Any]),
                "task_path": Artifact(Path),
                "model_path": Artifact(Path),
            }
        )

    @classmethod
    def get_output_sign(cls):
        return OPIOSign(
            {
                "traj": Artifact(Path),
                "log": Artifact(Path),
                "status": Artifact(Path),
            }
        )

    @OP.exec_sign_check
    def execute(
        self,
        ip: OPIO,
    ) -> OPIO:
        r"""Execute the OP.

        Parameters
        ----------
        ip : dict
            Input dict with components:

            - `config`: (`dict`) The config of MD task with stages, save_interval_steps, seed, etc.
            - `task_path`: (`Artifact(Path)`) The path that contains all input files prepared by `PrepMDTaskGroupOP`.
            - `model_path`: (`Artifact(Path)`) The model file path for DP calculator.

        Returns
        -------
        Any
            Output dict with components:
            - `traj`: (`Artifact(Path)`) The last trajectory file from the simulation stages.
            - `log`: (`Artifact(Path)`) The detailed log file containing simulation output.
            - `status`: (`Artifact(Path)`) A JSON file containing status and key messages.

        Raises
        ------
        TransientError
            On the failure of MD execution.
        """
        config = ip["config"]
        stages = config.get("stages", [])
        head = config.get("head")
        save_interval_steps = config.get("save_interval_steps", 100)
        traj_prefix = config.get("traj_prefix", "traj")
        seed = config.get("seed", 42)
        
        task_path = ip["task_path"]
        model_path = ip["model_path"]
        input_files = [ii.resolve() for ii in Path(task_path).iterdir()]
        work_dir = Path(Path(task_path).name)

        with set_directory(work_dir):
            # link input files
            for ii in input_files:
                iname = ii.name
                Path(iname).symlink_to(ii)
            
            # Setup logging
            log_file = Path("md_simulation.log")
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            
            # instantiate calculator
            calc = DP(model=model_path, head=head)
            
            # read initial structure
            atoms = read(ase_conf_name, index=0)
            atoms.calc = calc
            
            # create trajectory directory
            traj_dir = Path("trajs_files")
            traj_dir.mkdir(parents=True, exist_ok=True)
            
            try:
                logger.info("Starting molecular dynamics simulation")
                logger.info(f"Number of atoms: {len(atoms)}")
                logger.info(f"Number of stages: {len(stages)}")
                
                # run MD stages using _run_md_pipeline pattern
                
                for i, stage in enumerate(stages):
                    mode = stage.get('mode', 'NVT')
                    T = stage.get('temperature_K', 'NA')
                    P = stage.get('pressure', 'NA')
                    runtime_ps = stage.get('runtime_ps', 0.5)
                    
                    tag = f"stage{i+1}_{mode}_{T}K"
                    if P != 'NA':
                        tag += f"_{P}GPa"
                    traj_file = traj_dir / f"{traj_prefix}_{tag}.extxyz"
                    
                    logger.info(f"Starting stage {i+1}: {mode} ensemble, T={T}K, runtime={runtime_ps}ps")
                    
                    atoms = _run_md_stage(
                        atoms=atoms,
                        stage=stage,
                        save_interval_steps=save_interval_steps,
                        traj_file=str(traj_file),
                        seed=seed,
                        stage_id=i + 1
                    )
                
                logger.info("Molecular dynamics simulation completed successfully")
                
                # Create status file
                status_info = {
                    "status": "success",
                    "message": "Molecular dynamics simulation completed successfully",
                    "total_stages": len(stages),
                    #"trajectory_files": [str(p.name) for p in trajectory_list],
                    #"last_trajectory": str(last_traj.name) if last_traj else None,
                    "simulation_details": {
                        "num_atoms": len(atoms),
                        "stages_completed": len(stages),
                        "save_interval_steps": save_interval_steps,
                        "seed": seed
                    }
                }
                
                with open("status.json", 'w') as fp:
                    json.dump(status_info, fp, indent=2)
                    
            except Exception as e:
                logger.error(f"MD simulation failed: {str(e)}", exc_info=True)
                
                # Create error status file
                status_info = {
                    "status": "error",
                    "message": f"Molecular dynamics simulation failed: {str(e)}",
                    "total_stages": len(stages),
                    "stages_completed": i if 'i' in locals() else 0,
                    "error_details": str(e)
                }
                
                with open("status.json", 'w') as fp:
                    json.dump(status_info, fp, indent=2)
                    
                raise TransientError(f"ASE MD failed: {e}")
            finally:
                # Remove file handler to avoid accumulation
                logger.removeHandler(file_handler)
                file_handler.close()
                
        return OPIO({
            "traj": work_dir / str(traj_dir),
            "log": work_dir / "md_simulation.log",
            "status": work_dir / "status.json"
        })


class PrepRunMDTasks(Steps):
    """DFlow Steps wiring preparation, submission, and collection for MD tasks."""

    def __init__(
        self,
        *,
        name: str = "prep-run-md",
        prep_md_config: Dict[str, Any] = {},
        run_md_config: Dict[str, Any] = {},
        
    ):
        self._input_parameters = {
            "config": InputParameter(),
        }
        self._input_artifacts = {
            "structure_paths": InputArtifact(),
            "model_path": InputArtifact(),
        }
        self._output_parameters = {
        }
        self._output_artifacts = {
            "traj": OutputArtifact(),
            "log": OutputArtifact(),
            "status": OutputArtifact(),
        }

        super().__init__(
            name=name,
            inputs=Inputs(parameters=self._input_parameters, artifacts=self._input_artifacts),
            outputs=Outputs(parameters=self._output_parameters, artifacts=self._output_artifacts),
        )
        
        prep_md_executor = prep_md_config.pop("executor",None)
        prep_md_template_config = prep_md_config.pop("template_config", {})
        prep_step = Step(
            name="prep-md-tasks",
            template=PythonOPTemplate(
                PrepareMDTaskGroupOP,
                **prep_md_template_config,
            ),
            artifacts={
                "structure_paths": self.inputs.artifacts["structure_paths"]
                },
            **prep_md_config,
            executor=prep_md_executor,
        )
        self.add(prep_step)

        run_md_slice_config = run_md_config.pop("template_slice_config", {})
        run_md_executor = run_md_config.pop("executor",None)
        run_md_template_config = run_md_config.pop("template_config", {})
        run_md_step = Step(
            name="execute-md-tasks",
            template=PythonOPTemplate(
                RunMDTask,
                slices=Slices(
                '{{item}}',
                input_artifact=["task_path"],
                output_artifact=[
                    "traj",
                    "log",
                    "status",
                ],
                **run_md_slice_config,
            ),
            **run_md_template_config
            ),
            parameters={
                "config": self.inputs.parameters["config"],
            },
            artifacts={
                "task_path": prep_step.outputs.artifacts["task_paths"],
                "model_path": self.inputs.artifacts["model_path"],
            },
            key="--".join(["md-run","{{item}}"]),
            with_sequence=argo_sequence(
                argo_len(prep_step.outputs.parameters["task_idx"]),
            ),
            **run_md_config,
            executor=run_md_executor,
            
        )
        self.add(run_md_step)

        self.outputs.artifacts["traj"]._from = (
            run_md_step.outputs.artifacts["traj"]
        )
        self.outputs.artifacts["log"]._from = (
            run_md_step.outputs.artifacts["log"]
        )
        self.outputs.artifacts["status"]._from = (
            run_md_step.outputs.artifacts["status"]
        )


def _run_molecular_dynamics_batch(
    structure_paths: Union[Path, List[Path]],
    model_path: Path,
    config: Dict[str, Any],
    workflow_name: str = "molecular-dynamics-batch",
    debug: bool = True,
    prep_md_config: Optional[Dict[str, Any]] = None,
    run_md_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Run molecular dynamics simulations on multiple structures using DFlow workflow.
    
    Args:
        structure_paths: Path(s) to structure file(s) containing atomic configurations
        model_path: Path to the Deep Potential model file
        config: Configuration dictionary containing:
            - stages: List of MD simulation stages
            - head: Optional head name for multi-task models
            - save_interval_steps: Interval for saving trajectory frames
            - seed: Random seed for simulation
            - traj_prefix: Prefix for trajectory file names
        workflow_name: Name for the DFlow workflow
        mode: Workflow execution mode ("debug" for local, "default" for cluster)
        prep_md_config: Configuration for preparation step
        run_md_config: Configuration for MD execution step
        
    Returns:
        dict: Workflow results containing trajectory files, logs, and status
    """
    # Set DFlow mode
    if debug:
        dflow.config["mode"] = "debug"    
    #elif mode == "default":
    #    dflow.config["mode"] = "default"
    #else:
    #    raise ValueError(f"Invalid mode: {mode}. Must be 'debug' or 'default'")
    
    # Ensure structure_paths is a list
    if isinstance(structure_paths, (str, Path)):
        structure_paths = [Path(structure_paths)]
    else:
        structure_paths = [Path(p) for p in structure_paths]
    
    # Validate inputs
    for path in structure_paths:
        if not path.exists():
            raise FileNotFoundError(f"Structure file not found: {path}")
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Set default configurations
    if prep_md_config is None:
        prep_md_config = {
            "template_config": {},
            "executor": None
        }
    
    if run_md_config is None:
        run_md_config = {
            "template_config": {},
            "template_slice_config": {"group_size": 1, "pool_size": 1},
            "executor": None
        }
    
    # Create workflow artifacts
    artifacts = {
        "structure_paths": upload_artifact(structure_paths[0] if len(structure_paths) == 1 else structure_paths),
        "model_path": upload_artifact(model_path),
    }
    
    # Create MD batch operation
    run_md_batch_op = PrepRunMDTasks(
        prep_md_config=prep_md_config,
        run_md_config=run_md_config
    )
    
    # Create and submit workflow
    try:
        wf = Workflow(name=workflow_name)
        super_steps = Step(
            name="md-batch-execution",
            template=run_md_batch_op,
            parameters={
                "config": config,
            },
            artifacts=artifacts
        )
        wf.add(super_steps)
        
        logger.info(f"Submitting workflow '{workflow_name}' in {'debug' if debug else 'default'} mode")
        wf.submit()
        
        # Wait for completion and return results
        while wf.query_status() in ["Pending", "Running"]:
            time.sleep(5)
        
        status = wf.query_status()
        if status == "Succeeded":
            logger.info(f"Workflow '{workflow_name}' completed successfully")
            
            # Query step information to get artifacts
            step_info = wf.query()
            try:
                md_step = step_info.get_step(name="md-batch-execution")[0]
            except (IndexError, KeyError):
                logger.warning("Could not find 'md-batch-execution' step for artifact download")
                return {
                    "status": "error",
                    "workflow_id": wf.id,
                    "message": f"Molecular dynamics batch simulation completed successfully (artifacts not downloaded)",
                    "results": None
                }
            
            # Create download directory
            download_path = Path("./md_results")
            download_path.mkdir(exist_ok=True)
            logger.info(f"Downloading artifacts to: {download_path.resolve()}")
            
            # Download artifacts
            try:
                download_artifact(
                    artifact=md_step.outputs.artifacts["traj"],
                    path=download_path,
                )
                download_artifact(
                    artifact=md_step.outputs.artifacts["log"],
                    path=download_path,
                )
                dw=download_artifact(
                    artifact=md_step.outputs.artifacts["status"],
                    path=download_path,
                )
                print(dw)
                logger.info("Successfully downloaded all artifacts")
                
                return {
                    "status": "success",
                    "workflow_id": wf.id,
                    "message": f"Molecular dynamics batch simulation completed successfully",
                    "download_path": str(download_path.resolve())
                }
                
            except Exception as download_error:
                logger.warning(f"Failed to download artifacts: {str(download_error)}")
                return {
                    "status": "error",
                    "workflow_id": wf.id,
                    "message": f"Molecular dynamics batch simulation completed successfully (download failed: {str(download_error)})",
                }
        else:
            logger.error(f"Workflow '{workflow_name}' failed with status: {status}")
            return {
                "status": "error",
                "workflow_id": wf.id,
                "message": f"Workflow failed with status: {status}",
            }
            
    except Exception as e:
        logger.error(f"Failed to submit or execute workflow: {str(e)}", exc_info=True)
        return {
            "status": "error",
            "workflow_id": None,
            "message": f"Failed to submit or execute workflow: {str(e)}",
        }
        
        
        
def run_molecular_dynamics_batch(
    structure_paths: Union[Path, List[Path]],
    model_path: Path,
    config: Dict[str, Any],
    workflow_name: str = "molecular-dynamics-batch",
    mode: Literal["debug", "default"] = "debug",
    prep_md_config: Optional[Dict[str, Any]] = None,
    run_md_config: Optional[Dict[str, Any]] = None,
):
    
    work_path=Path(generate_work_path())
    work_path = work_path.resolve()
    work_path.mkdir(parents=True, exist_ok=True)
        
    with set_directory(work_path):
        result=_run_molecular_dynamics_batch(
                structure_paths=structure_paths,
                model_path=model_path,
                config=config,
                workflow_name=workflow_name,
                mode=mode,
                prep_md_config=prep_md_config,
                run_md_config=run_md_config
            )
        
        # If successful, scan for result files and organize them
        if result.get("status") == "success" and "download_path" in result:
            download_path = Path(result["download_path"])
            
            # Scan for task directories and their files
            task_results = {}
            log_files = []
            status_files = []
            traj_files = []
            
            # Find all task directories
            task_dirs = list(download_path.glob("task.[0-9]*"))
            
            for task_dir in sorted(task_dirs):
                task_name = task_dir.name
                task_results[task_name] = {}
                
                # Find log files
                log_pattern = task_dir / "*.log"
                task_logs = list(task_dir.glob("*.log"))
                if task_logs:
                    task_results[task_name]["log"] = task_logs[0].resolve()
                    log_files.extend([str(f.resolve()) for f in task_logs])
                
                # Find status/json files  
                json_pattern = task_dir / "*.json"
                task_jsons = list(task_dir.glob("*.json"))
                if task_jsons:
                    task_results[task_name]["status"] = task_jsons[0].resolve()
                    status_files.extend([str(f.resolve()) for f in task_jsons])
                
                # Find trajectory files
                traj_pattern = task_dir / "trajs_files" / "*.extxyz"
                task_trajs = list(task_dir.glob("trajs_files/*.extxyz"))
                if task_trajs:
                    task_results[task_name]["trajectories"] = [f.resolve() for f in task_trajs]
                    traj_files.extend([str(f.resolve()) for f in task_trajs])
            
            # Update result with organized file information
            result.update({
                "task_results": task_results,
                "all_trajectory_files": traj_files,
                "all_log_files": log_files,
                "all_status_files": status_files,
                "num_tasks": len(task_dirs)
            })
            
            logger.info(f"Found {len(task_dirs)} tasks with {len(traj_files)} trajectory files")
        
    return result