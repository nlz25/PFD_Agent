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
from ase.optimize import BFGS
from ase.filters import UnitCellFilter, ExpCellFilter

from deepmd.calculator import DP


from .utils import dflow_remote_execution,dflow_batch_execution


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


@dflow_batch_execution(
    batch_input_key=["structure_path"],
    artifact_inputs={
        "structure_path": Path,
        "model_path": Path,
    },
    artifact_outputs={
        "traj": Path,
        "log": Path,
    },
    parameter_inputs={
        "stages": list,
        "head": str,
        "save_interval_steps": int,
        "traj_prefix": str,
        "seed": int,
    },
    parameter_outputs={
        "message": str,
    },
    slice_config={
        "group_size": 1, 
        "pool_size": 1
    }
)
def _run_md(
    model_path: Path,
    stages: List[Dict[str, Any]],
    structure_path: Path,
    head: Optional[str]=None,
    save_interval_steps: int=100,
    traj_prefix: str="traj",
    seed: int=42,
) -> Dict[str, Any]:
    try:
        log_file = Path("md_simulation.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
            
        traj_dir = Path("trajectories")
        traj_dir.mkdir(parents=True, exist_ok=True)
        # instantiate calculator
        logger.info("Setting calculator")
        calc = DP(model=model_path, head=head)
            
        # read initial structure
        atoms = read(structure_path, index=0)
        atoms.calc = calc
            
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
            
        results = {
            "traj": traj_dir,
            "log": Path("md_simulation.log"),
            "status": Path("status.json"),
            "message": "MD simulation completed successfully"
        }
                    
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
                
                # Ensure log file exists for return
        if not log_file.exists():
            log_file.write_text(f"MD simulation failed: {str(e)}\n")
    
        if not traj_dir.exists():
            traj_dir.mkdir(parents=True, exist_ok=True)
            
        results = {
            "traj": traj_dir,
            "log": Path("md_simulation.log"),
            "status": Path("status.json"),
            "message": f"MD simulation failed: {e}"
        }
    finally:
        logger.removeHandler(file_handler)
        file_handler.close()
    return results


@dflow_remote_execution(
    artifact_inputs={
        "input_structure": Path,
        "model_path": Path,
    },
    artifact_outputs={
        "optimized_structure": Path,
        "optimization_traj": Path,
    },
    parameter_inputs={
        "head": str,
        "force_tolerance": float,
        "max_iterations": int,
        "relax_cell": bool,
    },
    parameter_outputs={
        "final_energy": float,
        "message": str,
    },
    op_name="DPAOptimizeStructureOP"
)
def optimize_structure( 
    input_structure: Path,
    model_path: Path,
    head: Optional[str]= None,
    force_tolerance: float = 0.01, 
    max_iterations: int = 100, 
    relax_cell: bool = False,
) -> Dict[str, Any]:
    try:
        base_name = input_structure.stem
        logging.info(f"Reading structure from: {input_structure}")
        atoms = read(str(input_structure))
       
        # Setup calculator
        calc=DP(model=model_path, head=head)  
        atoms.calc = calc

        traj_file = f"{base_name}_optimization_traj.extxyz"  
        if Path(traj_file).exists():
            logging.warning(f"Overwriting existing trajectory file: {traj_file}")
            Path(traj_file).unlink()

        logging.info("Starting structure optimization...")

        if relax_cell:
            logging.info("Using cell relaxation (ExpCellFilter)...")
            ecf = ExpCellFilter(atoms)
            optimizer = BFGS(ecf, trajectory=traj_file)
            optimizer.run(fmax=force_tolerance, steps=max_iterations)
        else:
            optimizer = BFGS(atoms, trajectory=traj_file)
            optimizer.run(fmax=force_tolerance, steps=max_iterations)

        output_file = Path(f"{base_name}_optimized.cif")
        write(output_file, atoms)
        final_energy = float(atoms.get_potential_energy())

        logging.info(
            f"Optimization completed in {optimizer.nsteps} steps. "
            f"Final energy: {final_energy:.4f} eV"
        )

        return {
            "optimized_structure": Path(output_file),
            "optimization_traj": Path(traj_file),
            "final_energy": final_energy,
            "message": f"Successfully completed in {optimizer.nsteps} steps",
        }

    except Exception as e:
        logging.error(f"Optimization failed: {str(e)}", exc_info=True)
        return {
            "optimized_structure": Path(""),
            "optimization_traj": None, 
            "final_energy": -1.0,
            "message": f"Optimization failed: {str(e)}",
        }