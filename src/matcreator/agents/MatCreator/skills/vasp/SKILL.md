---
name: vasp
description: Skills for VASP DFT calculations — input preparation, remote submission to Bohrium, and result collection.
metadata:
  tools:
    - run_bash
    - run_skill_script
  dependent_skills:
    - dpdisp
  tags:
    - vasp
    - dft
    - relaxation
    - scf
    - band-structure
    - dos
    - bohrium
---

# VASP DFT Skill


One script handles VASP-specific work; job submission is now delegated to the `dpdisp-submit` skill (DPDispatcher), which supports both Bohrium and standard Slurm/HPC clusters:

| Script | Role |
|---|---|
| `vasp_tools.py` | Prepare input files; collect / read results |
| `skills/dpdisp/` | Submit prepared directories as jobs via DPDispatcher (see `dpdisp-submit` skill) |

> **Important:** Never call `python vasp_tools.py` directly. Always invoke it via the `run_skill_script` tool:
> ```
> run_skill_script(skill_name="vasp", script_name="vasp_tools.py", args="<subcommand and flags>")
> ```
> This ensures the script is found regardless of the current working directory.

Every command prints JSON to stdout and exits 0 on success, 1 on error.

> **Note:** Submission is now handled by the `dpdisp-submit` skill, which supports both Bohrium and standard Slurm clusters. The previous `bohr` skill and `bohrium_submit.py` are deprecated for new workflows.

---


## Mandatory workflow sequence

1. **Obtain a structure** — if the user has not supplied one, generate it first.
2. **Prepare inputs** — run the appropriate `vasp_tools.py prepare_*` command.
3. **Submit jobs** — pass the returned `calc_dir_list` to the `dpdisp-submit` skill by generating a `submission.json` (see [Submission — dpdisp-submit skill](#submission-dpdisp-skill) below).
4. **Collect / read results** — after the job finishes, run `collect_results` or `read_results`.

Run exactly **one property step at a time**. Do not chain relaxation + SCF in a single step.

---

## vasp_tools.py — Command reference

### Common flags (all prepare_* commands)

| Flag | Default | Description |
|---|---|---|
| `--config PATH` | `config.yaml` next to the script | Path to config.yaml |
| `--incar_tags JSON` | `{}` | Extra INCAR overrides, e.g. `'{"ENCUT": 600}'` |
| `--potcar_map JSON` | `{}` | Element → POTCAR label, e.g. `'{"Bi": "Bi_d"}'` |

All `prepare_*` commands return:
```json
{ "status": "success", "calc_dir_list": ["<abs_path>", ...] }
```

---

### prepare_relaxation

Prepare structural relaxation (IBRION=2, ISIF=3, NSW=200).

```
run_skill_script(skill_name="vasp", script_name="vasp_tools.py", args="prepare_relaxation \
    --structure <path.extxyz|path.vasp> \
    [--frames 0 1 2] \
    [--kpoints NX NY NZ] \
    [--incar_tags '{\"ENCUT\": 600}'] \
    [--potcar_map '{\"Bi\": \"Bi_d\"}']")
```

- `--structure`: structure file to read. Supported formats:
  - `*.extxyz` — may contain multiple frames; all frames are processed by default.
  - `*.vasp` (POSCAR format) — always contains exactly one frame.
- `--frames`: integer indices to process (extxyz only); default is all frames.
- `--kpoints`: explicit Gamma-centred mesh. Default: auto KPPRA density 40.

---

### prepare_scf

Prepare a self-consistent field calculation (NSW=0, IBRION=-1).

```
run_skill_script(skill_name="vasp", script_name="vasp_tools.py", args="prepare_scf \
    --structure <path.extxyz|path.vasp> \
    [--frames 0 1 2] \
    [--kpoints NX NY NZ] \
    [--soc] \
    [--incar_tags '{\"ENCUT\": 600}']")
```

- `--structure`: structure file to read. Supported formats:
  - `*.extxyz` — may contain multiple frames; all frames are processed by default.
  - `*.vasp` (POSCAR format) — always contains exactly one frame.
- `--frames`: integer indices to process (extxyz only); default is all frames.
- `--soc`: use `scf_soc` INCAR preset (ISPIN=2, LSORBIT=.TRUE.); default: `scf_nsoc`.
- Default k-mesh: auto KPPRA density 40.
- SCF outputs CHGCAR and WAVECAR when `LCHARG=true` / `LWAVE=true` (SOC preset only by default).

---

### prepare_nscf_kpath

Prepare a non-self-consistent band-structure calculation along a k-path.
Requires completed SCF directories that contain `CONTCAR` and `CHGCAR`.

```
run_skill_script(skill_name="vasp", script_name="vasp_tools.py", args="prepare_nscf_kpath \
    --scf_dirs <scf_dir1> [<scf_dir2> ...] \
    [--kpath GMKG] \
    [--n_kpoints 16] \
    [--soc] \
    [--incar_tags '{\"NBANDS\": 48}']")
```

- `--kpath`: explicit path string (e.g. `GMKG`). Default: auto from pymatgen `HighSymmKpath`.
- `--n_kpoints`: points per segment (default 16).
- Copies `CHGCAR` (and `WAVECAR` if present) from the SCF directory.

---

### prepare_nscf_uniform

Prepare a non-self-consistent uniform-mesh calculation (for DOS).
Requires completed SCF directories.

```
run_skill_script(skill_name="vasp", script_name="vasp_tools.py", args="prepare_nscf_uniform \
    --scf_dirs <scf_dir1> [<scf_dir2> ...] \
    [--kpoints NX NY NZ] \
    [--soc] \
    [--incar_tags '{\"NEDOS\": 2000}']")
```

- Default k-mesh: auto KPPRA density 100.
- Copies `CHGCAR` (and `WAVECAR` if present) from the SCF directory.

---

### collect_results

Parse `OUTCAR` files and write all frames into a single extxyz (via dpdata).

```
run_skill_script(skill_name="vasp", script_name="vasp_tools.py", args="collect_results \
    --dirs <calc_dir1> [<calc_dir2> ...]")
```

Returns:
```json
{ "status": "success", "scf_result": "<abs_path_to_scf_result.extxyz>" }
```

---

### read_results

Read `vasprun.xml` / `OUTCAR` and return key scalar results as JSON.

```
run_skill_script(skill_name="vasp", script_name="vasp_tools.py", args="read_results \
    --calc_type <relaxation|scf|nscf> \
    --calc_dir  <calc_dir>")
```

| calc_type | Returned fields |
|---|---|
| `relaxation` | `structure`, `total_energy`, `max_force`, `stress`, `ionic_steps` |
| `scf` | `structure`, `total_energy`, `efermi`, `band_gap`, `is_metal` |
| `nscf` | `structure`, `efermi`, `is_metal`, `band_gap`, `cbm`, `vbm` |

---


## Submission — `dpdisp-submit` skill {#submission-dpdisp-skill}

Submission is handled by the `dpdisp-submit` skill (DPDispatcher), which supports both Bohrium and standard Slurm/HPC clusters. See the `dpdisp-submit` skill documentation for full details and schema.

For the full VASP-specific Bohrium `submission.json` template, environment variables, and submission flow, load the reference:

```
load_skill_resource(skill_name="vasp", path="references/bohrium-submission.md")
```

---


## End-to-end example: relaxation → SCF → band structure

```
# 1. Prepare relaxation
run_skill_script(skill_name="vasp", script_name="vasp_tools.py",
    args="prepare_relaxation --structure Al.extxyz")
# → returns {"status": "success", "calc_dir_list": [...]}

# 2. Generate submission.template.json for relaxation (see above for schema)
#    (Repeat for each calc_dir as a task in task_list)

# 3. Substitute environment variables
envsubst '${BOHRIUM_USERNAME} ${BOHRIUM_PASSWORD} ${BOHRIUM_PROJECT_ID} ${BOHRIUM_VASP_MACHINE} ${BOHRIUM_VASP_IMAGE}' < submission.template.json > submission.json

# 4. Validate and submit
uv run -m json.tool submission.json >/dev/null
uvx --with dpdispatcher dargs check -f dpdispatcher.entrypoints.submit.submission_args submission.json
uvx --from dpdispatcher --with oss2 dpdisp submit submission.json

# 5. Read relaxation results
run_skill_script(skill_name="vasp", script_name="vasp_tools.py",
    args="read_results --calc_type relaxation --calc_dir <relax_dir>")

# 6. Prepare SCF from relaxed structure (CONTCAR → extxyz conversion needed, or pass CONTCAR directly)
run_skill_script(skill_name="vasp", script_name="vasp_tools.py",
    args="prepare_scf --structure Al_relaxed.extxyz")

# 7. Repeat submission steps for SCF (adjust forward/backward files as needed)

# 8. Prepare NSCF k-path from SCF output
run_skill_script(skill_name="vasp", script_name="vasp_tools.py",
    args="prepare_nscf_kpath --scf_dirs <scf_dir1> [<scf_dir2> ...]")

# 9. Repeat submission steps for NSCF

# 10. Read NSCF results (band gap, CBM/VBM)
run_skill_script(skill_name="vasp", script_name="vasp_tools.py",
    args="read_results --calc_type nscf --calc_dir <nscf_dir>")
```

---

## Configuration file (config.yaml)

`config.yaml` lives in the same directory as `vasp_tools.py`. It controls:

- `work_dir` — where all calculation subdirectories are created (default `/tmp/vasp_server`).
- `VASP_default_INCAR` — one sub-key per preset (`relaxation`, `scf_nsoc`, `scf_soc`, `nscf_nsoc`, `nscf_soc`).

**When asked about default INCAR settings, read `config.yaml` directly** — do not guess from memory, as the user may have edited it.

Use `run_bash` to inspect it:
```bash
cat "$(python -c "import importlib.util, pathlib; print(pathlib.Path(importlib.util.find_spec('vasp_tools').origin).parent if importlib.util.find_spec('vasp_tools') else '.')")/config.yaml"
```
Or simply ask the agent to read it via `read_workspace_file("skills/vasp/config.yaml")`.

To override individual tags for a single run without editing the file, use `--incar_tags`:
```bash
--incar_tags '{"ENCUT": 600, "EDIFF": 1e-6}'
```

To permanently change a default, edit the relevant preset in `config.yaml` directly.
