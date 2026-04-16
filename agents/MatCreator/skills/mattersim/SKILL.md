---
name: mattersim
description: Run batch structure relaxation and batch molecular dynamics with the MatterSim CLI, using extxyz as the default structure and trajectory format.
tags: [MatterSim, relaxation, molecular dynamics, MSD, Mean Squared Displacement]
tools: [run_bash]
dependent_skills: []
---
# MatterSim Skill


Default format
- Use `extxyz` as the default input and output format.
- For batch jobs, prefer a multi-frame `extxyz` file as input.


## Mattersim on local

### Batch relaxation

Use the MatterSim CLI `relax` subcommand.

Example:

```bash
ts=$(date +"%Y%m%d%H%M%S")
workdir="matsim/${ts}.mattersim_relax"
mkdir -p "$workdir"
cd "$workdir"
source ${MATTERGEN_ENV}/bin/activate

python -m mattersim.cli.mattersim_app relax \
  --structure-file /abs/path/to/structures.extxyz \
  --device cuda \
  --work-dir results \
  --filter EXPCELLFILTER \
  --fmax 0.01 \
  --steps 500
```

Important options
- `--structure-file`: one or more structure files; prefer one multi-frame `extxyz`
- `--mattersim-model`: optional:["mattersim-v1.0.0-1m", "mattersim-v1.0.0-5m"], default:"mattersim-v1.0.0-1m"
- `--device`: `cpu` or `cuda`
- `--work-dir`: working directory for relaxation outputs
- `--save-csv`: output table for relaxation results
- `--filter`: optional cell filter
- `--constrain-symmetry`: optional enable symmetry constraints when needed
- `--pressure-in-GPa`: optional target pressure
- `--fmax`: force convergence threshold
- `--steps`: maximum relaxation steps

After relaxation, extract the structures from results.csv.gz and save them as an extxyz file.


### Batch molecular dynamics

Use the MatterSim CLI `moldyn` subcommand.

Recommended workflow:

1. Create a timestamped `workdir`:

```bash
ts=$(date +"%Y%m%d%H%M%S")
workdir="matsim/${ts}.mattersim_batch_md"
mkdir -p "$workdir"
```

2. Prepare the input structures. If the input is a multi-frame `extxyz`, split it into single-frame structure files, create numerically indexed subdirectories under `workdir`, and place one `extxyz` file in each directory. Also determine the starting and ending indices, `num_start` and `num_end`.
3. Run `mattersim moldyn` once for each structure directory.


Example:

```bash
source "${MATTERGEN_ENV}/bin/activate"
for i in $(seq "$num_start" "$num_end"); do
  python -m mattersim.cli.mattersim_app moldyn \
    --structure-file "$workdir/$i/structure.extxyz" \
    --device cuda \
    --work-dir "$workdir/$i/results" \
    --temperature 300 \
    --timestep 1 \
    --steps 1000 \
    --loginterval 10 \
    --taut 100
done
```

Important options
- `--structure-file`: a single structure file for each MD run
- `--mattersim-model`: optional:["mattersim-v1.0.0-1m", "mattersim-v1.0.0-5m"], default:"mattersim-v1.0.0-1m"
- `--work-dir`: working directory for MD outputs
- `--save-csv`: output table for MD results
- `--temperature`: temperature in Kelvin
- `--timestep`: timestep in femtoseconds
- `--steps`: number of MD steps
- `--loginterval`: logging interval
- `--trajectory`: trajectory output path
- `--taut`: optional thermostat parameter

Notes
- Molecular dynamics should not be run on multiple structures in a single call.
- If the input is a multi-frame `extxyz`, split it into multiple single-structure `extxyz` files and run them one by one.
- Use separate output directories and trajectory files for different structures.
- Prefer batch relaxation for structure optimization.
- Prefer batch MD for finite-temperature dynamics, diffusion, or structural evolution.








## Mattersim on Bohrium

When submitting MatterGen jobs to Bohrium through `dpdisp`, Bohrium-specific submission settings can be read from environment variables such as `BOHRIUM_MAT_IMAGE` and `BOHRIUM_MAT_MACHINE`. For the `dpdisp` submission procedure, refer to the `dpdisp` skill documentation.

### Batch relaxation

```bash
ts=$(date +"%Y%m%d%H%M%S")
workdir="matsim/${ts}.mattersim_relax"
mkdir -p "$workdir"
cd "$workdir"
```

Then copy the structure to be relaxed into the `workdir`.

In `submission.json`:

- set `command` to a generation command such as:

```bash
/opt/mattergen/.venv/bin/python -m mattersim.cli.mattersim_app relax \
  --structure-file structures.extxyz \
  --device cuda \
  --work-dir results \
  --filter EXPCELLFILTER \
  --fmax 0.01 \
  --steps 500
```

- `forward_files` should include the structures, such as `generated.extxyz`.
- `backward_files` can be `results`.

After relaxation, extract the structures from results.csv.gz and save them as an extxyz file.


### Batch molecular dynamics

1. Create a timestamped `workdir`:

```bash
ts=$(date +"%Y%m%d%H%M%S")
workdir="matsim/${ts}.mattersim_batch_md"
mkdir -p "$workdir"
cd "$workdir"
```

2. Prepare the input structures. If the input is a multi-frame `extxyz`, split it into single-frame structure files, create numerically indexed subdirectories under `workdir`, and place one `extxyz` file in each directory. Also determine the starting and ending indices, `num_start` and `num_end`.


3. In `submission.json`:

- set `command` to a shell loop such as:

```bash
for i in $(seq num_start num_end); do
  cd $i
  /opt/mattergen/.venv/bin/python -m mattersim.cli.mattersim_app moldyn \
    --structure-file structure.extxyz \
    --device cuda \
    --work-dir results \
    --temperature 300 \
    --timestep 1 \
    --steps 1000 \
    --loginterval 10 \
    --taut 100
  cd ..
done
```

- because `submission.json` is not executed in the same shell as step 1, use the actual numeric values for `num_start` and `num_end` in `command`.
- `forward_files` should include all indexed structure directories, for example `["1", "2", "3"]`.
- `backward_files` can use the same directory list as `forward_files`, for example `["1", "2", "3"]`.




## Compute Ionic Conductivity

After molecular dynamics, estimate the ionic conductivity of the target mobile ion species such as `Li` or other ions from the MD trajectory.

Recommended workflow:
1. Read the MD trajectory file, for example `md.traj`.
2. Select the target mobile ion species whose transport behavior should be analyzed.
3. Compute the mean squared displacement (MSD) of the selected ions relative to the initial frame.
4. Fit the long-time linear region of the MSD curve to estimate the diffusion coefficient `D` using the Einstein relation for three-dimensional diffusion:

```text
MSD(t) ~= 6 D t
```

5. Compute the number density `n` of the mobile ions from the simulation cell volume and the number of selected ions.
6. Convert the diffusion coefficient to ionic conductivity `sigma` with the Nernst-Einstein relation:

```text
sigma = n q^2 D / (k_B T)
```

where:
- `n` is the number density of the mobile ions
- `q` is the ion charge in coulombs
- `D` is the diffusion coefficient in m^2/s
- `k_B` is the Boltzmann constant
- `T` is the temperature in kelvin

7. Save the MSD curve and the estimated transport quantities to files such as `msd.csv` and `conductivity.json`.

Example:

```python
import json
import numpy as np
from ase.io import read

trajectory_path = "/abs/path/to/results/md_000001/md.traj"
frames = read(trajectory_path, index=":")

target_species = "Li"
ion_charge_number = 1
temperature_K = 300.0
timestep_fs = 1.0
loginterval = 10

indices = [i for i, s in enumerate(frames[0].get_chemical_symbols()) if s == target_species]
if not indices:
    raise ValueError(f"No atoms with species {target_species!r} were found.")

r0 = frames[0].get_positions()[indices]
times_ps = []
msd = []

for step, atoms in enumerate(frames):
    rt = atoms.get_positions()[indices]
    dr2 = np.sum((rt - r0) ** 2, axis=1)
    msd.append(float(np.mean(dr2)))
    times_ps.append(step * timestep_fs * loginterval / 1000.0)

times_ps = np.array(times_ps)
msd = np.array(msd)

# Example fitting window: use the middle 60% of the trajectory.
n_points = len(times_ps)
fit_start = int(n_points * 0.2)
fit_end = int(n_points * 0.8)
fit_times = times_ps[fit_start:fit_end]
fit_msd = msd[fit_start:fit_end]

if len(fit_times) < 2:
    raise ValueError("Not enough points in the fitting window to estimate conductivity.")

slope_ang2_per_ps, intercept_ang2 = np.polyfit(fit_times, fit_msd, 1)

# Einstein relation for 3D diffusion.
d_ang2_per_ps = slope_ang2_per_ps / 6.0
d_m2_per_s = d_ang2_per_ps * 1e-8

# Number density from the initial cell volume.
volume_ang3 = frames[0].get_volume()
volume_m3 = volume_ang3 * 1e-30
number_density_m3 = len(indices) / volume_m3

# Nernst-Einstein relation.
elementary_charge_C = 1.602176634e-19
boltzmann_constant_J_per_K = 1.380649e-23
charge_C = ion_charge_number * elementary_charge_C

conductivity_S_per_m = (
    number_density_m3 * charge_C**2 * d_m2_per_s / (boltzmann_constant_J_per_K * temperature_K)
)
conductivity_mS_per_cm = conductivity_S_per_m * 10.0

np.savetxt(
    "msd.csv",
    np.column_stack([times_ps, msd]),
    delimiter=",",
    header="time_ps,msd_angstrom2",
    comments="",
)

with open("conductivity.json", "w", encoding="ascii") as f:
    json.dump(
        {
            "target_species": target_species,
            "temperature_K": temperature_K,
            "ion_charge_number": ion_charge_number,
            "timestep_fs": timestep_fs,
            "loginterval": loginterval,
            "fit_start_index": fit_start,
            "fit_end_index": fit_end - 1,
            "fit_start_time_ps": float(fit_times[0]),
            "fit_end_time_ps": float(fit_times[-1]),
            "slope_angstrom2_per_ps": float(slope_ang2_per_ps),
            "diffusion_coefficient_m2_per_s": float(d_m2_per_s),
            "number_density_m3": float(number_density_m3),
            "conductivity_S_per_m": float(conductivity_S_per_m),
            "conductivity_mS_per_cm": float(conductivity_mS_per_cm),
        },
        f,
        indent=2,
    )
```

Notes
- The conductivity obtained this way is the Nernst-Einstein conductivity derived from self-diffusion.
- MSD is usually computed separately for each trajectory because molecular dynamics is run one structure at a time.
- Use the same target ion species, ion charge, and temperature consistently across all trajectories.
- If periodic boundary conditions are important for long trajectories, use an unwrapped trajectory or another PBC-aware displacement reconstruction method.
- The diffusion coefficient should be fitted from the diffusive linear regime, not from the initial short-time ballistic regime.
- If the trajectory is too short or too noisy, report that the estimated conductivity has limited confidence.
- In correlated ionic systems, the Nernst-Einstein relation can overestimate the true ionic conductivity.



## What to report

Report at minimum:
- for batch relaxation: the number of input structures, `optimizer`, `filter`, `fmax`, `steps`, and the absolute path to the relaxation work directory and CSV results
- for batch MD: the number of input structures, whether the input was split before running, temperature, timestep, number of MD steps, ensemble, and the absolute path to each MD work directory, trajectory file
- for ionic conductivity analysis: the target ion species, ion charge, temperature, the trajectory used, the fitting time window, the fitted MSD slope, the estimated diffusion coefficient `D`, the conductivity `sigma`, their units, and the absolute paths to the exported `msd.csv` and `conductivity.json`

Source basis
- local MatterSim CLI code provided in the workspace conversation, including subcommands `relax` and `moldyn`
