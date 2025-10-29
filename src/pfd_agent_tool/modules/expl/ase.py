from typing import (
    Optional, 
    List, 
    Dict, 
    TypedDict,
    Any,
)
import os
from pathlib import Path
import numpy as np
from ase.io import read, write
from ase.md.andersen import Andersen
from ase.md.langevin import Langevin
from ase.md.nose_hoover_chain import NoseHooverChainNVT
from ase.md.npt import NPT
from ase.md.nvtberendsen import NVTBerendsen
from ase.md.velocitydistribution import (MaxwellBoltzmannDistribution,
                                         Stationary, ZeroRotation)
from ase.md.verlet import VelocityVerlet
from ase.optimize import BFGS
from ase.constraints import ExpCellFilter
from ase import units
from .calculator import CalculatorWrapper
from pfd_agent_tool.init_mcp import mcp
from pfd_agent_tool.modules.util.common import generate_work_path
from pfd_agent_tool.modules.log.log import log_step
import logging

class OptimizationResult(TypedDict):
    """Result structure for structure optimization"""
    optimized_structure: Path
    optimization_traj: Optional[Path]
    final_energy: float
    message: str


@mcp.tool()
def list_calculators() -> List[Dict[str, Any]]:
    """List available calculators and their MD input requirements.

    This reflects the concrete implementations in `calculator.py` and is intended to
    guide correct usage of `optimize_structure` and `run_molecular_dynamics`. Please follow the examples provided.

    Returns:
      A list of dicts, one per registered calculator, with:
        - name: calculator key to use as model_style
        - description: short summary (aligned to wrapper implementation)
        - requires_model_path: whether `model_path` must be provided
        - optional_calc_args: optional kwargs for calculator initialization
        - md_supported: whether suitable for MD in this toolkit
        - notes: extra hints
        - example: minimal example for run_molecular_dynamics
    """
    # Specs derived from src/pfd_agent_tool/modules/expl/calculator.py wrappers
    specs: Dict[str, Dict[str, Any]] = {
        "dpa": {
            "description": "DeepMD (deepmd-kit) ASE calculator wrapper.",
            "requires_model_path": True,
            "optional_calc_args": ["head"],
            "md_supported": True,
            "notes": "Always specify `head` for multi-head models to select the desired potential. Default to `MP_traj_v024_alldata_mixu` if not specified.",
            "example": "run_molecular_dynamics(initial_structure=Path('in.xyz'), stages=stages, model_style='dpa', model_path=Path('.tests/dpa/DPA2_medium_28_10M_rc0.pt'),head='MP_traj_v024_alldata_mixu')",
        },
        "emt": {
            "description": "EMT toy calculator (no model file).",
            "requires_model_path": False,
            "optional_calc_args": [],
            "md_supported": True,
            "notes": "Good for smoke tests; no model_path needed.",
            "example": "run_molecular_dynamics(initial_structure=Path('in.xyz'), stages=stages, model_style='emt')",
        }
    }

    #available = CalculatorWrapper.get_all_calculator()
    result: List[Dict[Any]] = []
    for name in specs.keys():
        spec = specs.get(name, {})
        result.append({
            "name": name,
            "description": spec.get("description", ""),
            "requires_model_path": spec.get("requires_model_path", False),
            "required_init_params": spec.get("required_init_params", []),
            "optional_init_params": spec.get("optional_init_params", []),
            "md_supported": spec.get("md_supported", True),
            "notes": spec.get("notes", ""),
            "example": spec.get("example", ""),
        })
    return result

@mcp.tool()
def get_base_model_path(
    model_style: str= "dpa",
    model_path: Optional[Path]=None
    ) -> Dict[str,Any]:
    """Resolve a usable base model path before using `run_molecular_dynamics` tool.

    Behavior:
    1) Prefer the explicitly provided `model_path` if given.
    2) Otherwise, read from environment variable `BASE_MODEL_PATH`.
    3) Normalize local paths (expanduser + resolve). If a directory is provided,
       try to find a model file inside with common suffixes (.pt, .pth, .pb).
    4) If an HTTP(S) URL is provided, return it as-is.

    Returns: A dictionary contains:
        - base_model_path: normalized local Path or an HTTP(S) URI string (framework will serialize Paths),
        or None if nothing can be determined.
    """

    def _is_url(s: str) -> bool:
        return s.startswith("http://") or s.startswith("https://")

    def _as_path_or_str(p: Optional[Path | str]) -> Optional[Path | str]:
        if p is None:
            return None
        # Accept strings (including URLs) or Path-like
        if isinstance(p, Path):
            return p
        if isinstance(p, str) and _is_url(p):
            return p  # return URL as-is
        # Otherwise, treat as filesystem path
        try:
            return Path(p).expanduser().resolve()
        except Exception:
            return Path(p)

    def _pick_model_in_dir(d: Path) -> Optional[Path]:
        if not d.is_dir():
            return None
        # Preference order
        candidates = []
        for suf in ("*.pt", "*.pth", "*.pb"):
            candidates.extend(sorted(d.glob(suf)))
        return candidates[0] if candidates else None

    # 1) Prefer explicit argument
    source = model_path if model_path not in (None, "") else None
    # 2) Else environment
    if source is None:
        if model_style == "dpa":
            env_val = os.getenv("DPA_MODEL_PATH", "").strip()
            source = env_val if env_val else None

    if source is None:
        logging.error("No model path could be determined from arguments or environment.")
        return {"base_model_path": None}

    resolved = _as_path_or_str(source)

    # If URL, return as-is
    if isinstance(resolved, str) and _is_url(resolved):
        return {"base_model_path": resolved}  # type: ignore[return-value]

    # Local path handling
    assert isinstance(resolved, Path)

    # If it's a file with a known suffix, return it
    if resolved.suffix.lower() in {".pt", ".pth", ".pb"}:
        return {"base_model_path": resolved}

    # If it's a directory, try to pick a model file inside
    if resolved.is_dir():
        picked = _pick_model_in_dir(resolved)
        if picked is not None:
            return {"base_model_path": picked}
        # Fall back to directory itself if no files are found
        return {"base_model_path": resolved}

    # For unknown suffixes or non-existing paths: if it looks like a file with a known model suffix
    # in the name, just return it; otherwise return the parent as a best-effort "base" path.
    if any(str(resolved).lower().endswith(s) for s in (".pt", ".pth", ".pb")):
        return {"base_model_path": resolved}

    
    return {"base_model_path": resolved.parent if resolved.parent != resolved else resolved}

@mcp.tool()
def optimize_structure( 
    input_structure: Path,
    model_style: str = "dpa",
    model_path: Optional[Path]= None,
    force_tolerance: float = 0.01, 
    max_iterations: int = 100, 
    relax_cell: bool = False,
    **kwargs
) -> OptimizationResult:
    """Optimize crystal structure using a Deep Potential (DP) model.

    Args:
        input_structure (Path): Path to the input structure file (e.g., CIF, POSCAR).
        model_path (Path): Path to the model file
            If not provided, using the `get_base_model_path` tool to obtain the default model path.
        force_tolerance (float, optional): Convergence threshold for atomic forces in eV/Å.
            Default is 0.01 eV/Å.
        max_iterations (int, optional): Maximum number of geometry optimization steps.
            Default is 100 steps.
        relax_cell (bool, optional): Whether to relax the unit cell shape and volume in addition to atomic positions.
            Default is False.
        **kwargs: Additional keyword arguments passed to the calculator initialization. For example, for DeepMD calculator, you might specify `head` to select the model head for multi-head models.


    Returns:
        dict: A dictionary containing optimization results:
            - optimized_structure (Path): Path to the final optimized structure file.
            - optimization_traj (Optional[Path]): Path to the optimization trajectory file, if available.
            - final_energy (float): Final potential energy after optimization in eV.
            - message (str): Status or error message describing the outcome.
    """
    try:
        base_name = input_structure.stem
        
        logging.info(f"Reading structure from: {input_structure}")
        atoms = read(str(input_structure))
       
        # Setup calculator
        calc=CalculatorWrapper.get_calculator(model_style)
        calc=calc().create(model_path=model_path, **kwargs)
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
            
        work_path=Path(generate_work_path())
        work_path = work_path.expanduser().resolve()
        work_path.mkdir(parents=True, exist_ok=True)

        output_file = work_path / f"{base_name}_optimized.cif"
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

def _log_progress(atoms, dyn):
    """Log simulation progress"""
    epot = atoms.get_potential_energy()
    ekin = atoms.get_kinetic_energy()
    temp = ekin / (1.5 * len(atoms) * units.kB)
    logging.info(f"Step: {dyn.nsteps:6d}, E_pot: {epot:.3f} eV, T: {temp:.2f} K")

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
        dyn = NVTBerendsen(
            atoms,
            timestep=timestep_fs * units.fs,
            temperature_K=temperature_K,
            taut=tau_t_ps * 1000 * units.fs
        )
    elif mode == 'NVT-Andersen':
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
        dyn = VelocityVerlet(
            atoms,
            timestep=timestep_fs * units.fs
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Prepare trajectory file
    os.makedirs(os.path.dirname(traj_file), exist_ok=True)
    if os.path.exists(traj_file):
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
    dyn.attach(lambda: _log_progress(atoms, dyn), interval=100)

    logging.info(f"[Stage {stage_id}] Starting {mode} simulation: T={temperature_K} K"
                + (f", P={pressure} GPa" if pressure is not None else "")
                + f", steps={total_steps}, dt={timestep_ps} ps")

    # Run simulation
    dyn.run(total_steps)
    logging.info(f"[Stage {stage_id}] Finished simulation. Trajectory saved to: {traj_file}\n")
    #traj_file=Path(traj_file).absolute()

    return atoms#, traj_file

def _run_md_pipeline(atoms, stages, save_interval_steps=100, traj_prefix='traj', traj_dir='trajs_files', seed=42):
    """Run multiple MD stages sequentially"""
    for i, stage in enumerate(stages):
        mode = stage['mode']
        T = stage.get('temperature_K', 'NA')
        P = stage.get('pressure', 'NA')

        tag = f"stage{i+1}_{mode}_{T}K"
        if P != 'NA':
            tag += f"_{P}GPa"
        traj_file = os.path.join(traj_dir, f"{traj_prefix}_{tag}.extxyz")

        atoms = _run_md_stage(
            atoms=atoms,
            stage=stage,
            save_interval_steps=save_interval_steps,
            traj_file=traj_file,
            seed=seed,
            stage_id=i + 1
        )

    return atoms

@mcp.tool()
@log_step(
    step_name="explore_md"
)
def run_molecular_dynamics(
    initial_structure: Path,
    stages: List[Dict],
    model_style: str = "dpa",
    model_path: Optional[Path]= None,
    save_interval_steps: int = 100,
    traj_prefix: str = 'traj',
    seed: Optional[int] = 42,
    calc_args: Optional[Dict[str,str]]=None
) -> Dict:
    """
    [Modified from AI4S-agent-tools/servers/DPACalculator] Run a multi-stage molecular dynamics simulation using Deep Potential. 

    This tool performs molecular dynamics simulations with different ensembles (NVT, NPT, NVE)
    in sequence, using the ASE framework with the Deep Potential calculator.

    Args:
        initial_structure (Path): Input atomic structure file (supports .xyz, .cif, etc.)
        model_path (Path): Path to the model file
            If not provided, using the `get_base_model_path` tool to obtain the default model path.
        stages (List[Dict]): List of simulation stages. Each dictionary can contain:
            - mode (str): Simulation ensemble type. One of:
                * "NVT" or "NVT-NH"- NVT ensemble (constant Particle Number, Volume, Temperature), with Nosé-Hoover (NH) chain thermostat
                * "NVT-Berendsen"- NVT ensemble with Berendsen thermostat. For quick thermalization
                * 'NVT-Andersen- NVT ensemble with Andersen thermostat. For quick thermalization (not rigorous NVT)
                * "NVT-Langevin" or "Langevin"- Langevin dynamics. For biomolecules or implicit solvent systems.
                * "NPT-aniso" - constant Number, Pressure (anisotropic), Temperature
                * "NPT-tri" - constant Number, Pressure (tri-axial), Temperature
                * "NVE" - constant Number, Volume, Energy (no thermostat/barostat, or microcanonical)
            - runtime_ps (float): Simulation duration in picoseconds.
            - temperature_K (float, optional): Temperature in Kelvin (required for NVT/NPT).
            - pressure (float, optional): Pressure in GPa (required for NPT).
            - timestep_ps (float, optional): Time step in picoseconds (default: 0.0005 ps = 0.5 fs).
            - tau_t_ps (float, optional): Temperature coupling time in picoseconds (default: 0.01 ps).
            - tau_p_ps (float, optional): Pressure coupling time in picoseconds (default: 0.1 ps).
        save_interval_steps (int): Interval (in MD steps) to save trajectory frames (default: 100).
        traj_prefix (str): Prefix for trajectory output files (default: 'traj').
        seed (int, optional): Random seed for initializing velocities (default: 42).
        calc_args (Dict[str, str], optional): Optional calculator initialization parameters passed directly to the
            calculator wrapper. For pretrained DPA multi-head models, an available head should be provided. 
            The head is defaulted to "MP_traj_v024_alldata_mixu" if not specified.

    Returns: A dictionary containing:
            - trajectory_list (List[Path]): The paths of output trajectory files generated.
            - log_file (Path): Path to the log file containing simulation output.

    Examples:
        >>> stages = [
        ...     {
        ...         "mode": "NVT",
        ...         "temperature_K": 300,
        ...         "runtime_ps": 5,
        ...         "timestep_ps": 0.0005,
        ...         "tau_t_ps": 0.01
        ...     },
        ...     {
        ...         "mode": "NPT-aniso",
        ...         "temperature_K": 300,
        ...         "pressure": 1.0,
        ...         "runtime_ps": 5,
        ...         "timestep_ps": 0.0005,
        ...         "tau_t_ps": 0.01,
        ...         "tau_p_ps": 0.1
        ...     },
        ...     {
        ...         "mode": "NVE",
        ...         "runtime_ps": 5,
        ...         "timestep_ps": 0.0005
        ...     }
        ... ]

        >>> result = run_molecular_dynamics(
        ...     initial_structure=Path("input.xyz"),
        ...     model_path=Path("model.pb"),
        ...     stages=stages,
        ...     save_interval_steps=50,
        ...     traj_prefix="cu_relax",
        ...     seed=42
        ... )
    """
    # Create output directories
    try:
        work_path=Path(generate_work_path())
        work_path = work_path.expanduser().resolve()
        work_path.mkdir(parents=True, exist_ok=True)
    
        traj_dir = work_path / "trajs_files"
        traj_dir.mkdir(parents=True, exist_ok=True)
        log_file = work_path / "md_simulation.log"
    
        # Read initial structure
        atoms_ls = read(initial_structure,index=':')
    
        # Setup calculator
        # Flatten an accidentally nested 'kwargs' dict sent by clients; keep robust and log
        if calc_args is None:
            calc_args={}

        calc=CalculatorWrapper.get_calculator(model_style)
        calc=calc().create(model_path=model_path, **calc_args)
    
        #final_structures_ls=[]
        for idx,atoms in enumerate(atoms_ls):
            atoms.calc = calc
        # Run MD pipeline
            _ = _run_md_pipeline(
                atoms=atoms,
                stages=stages,
                save_interval_steps=save_interval_steps,
                traj_prefix=traj_prefix+("_%03d"%idx),
                traj_dir=str(traj_dir),
                seed=seed
            )
    
        traj_list = sorted(
            p.resolve()  # use resolve(strict=True) if you want to fail on broken links
            for p in Path(traj_dir).glob("*.extxyz")
            )
        result = {
            "status": "success",
            "message": "Molecular dynamics simulation completed successfully.",
            "trajectory_list": traj_list,
            "log_file": log_file,
        }

    except Exception as e:
        logging.error(f"Molecular dynamics simulation failed: {str(e)}", exc_info=True)
        result = {
            "status": "error",
            "message": f"Molecular dynamics simulation failed: {str(e)}",
            "trajectory_list": [],
            "log_file": Path(""),
        }
    return result

