---
name: dpa4
description: DPA4 (SeZM) finetuning skill — finetuning only, remote on Bohrium. All training labels and benchmarks must come from DFT; the pretrained model is for MD exploration only.
metadata:
  tools:
    - run_bash
  dependent_skills:
    - dpdisp
    - vasp
    - abacus
    - atomic-structure
  tags:
    - deepmd
    - dpa4
    - sezm
    - finetuning
    - machine-learning-potential
    - dft-labeling
---

> ⚠️ **MANDATORY — READ BEFORE ACTING**
>
> 1. **This skill is the SINGLE SOURCE OF TRUTH** for DPA4 workflows. It **overrides**
>    any existing DAG, execution plan, or pre-existing directory configuration.
> 2. **Always use `dpa4_prepare.py`** to generate the workdir. **Never reuse** old
>    `ft-*/`, `train_*/`, or any pre-existing work directories — their input.json,
>    job.json, and data layouts may be from a different workflow and are NOT valid.
> 3. **Read this skill IN FULL** before executing. Do not act on headline lines alone.
>    Key decision points (dataset check, system complexity, frame budget) are in the
>    workflow sections below.
> 4. If a DAG or orchestrator plan contradicts this skill, **follow this skill** and
>    discard the conflicting plan.

# DPA4 Skill

DPA4 (SeZM-type descriptor) **finetuning only** skill, targeting the **neo** version.
Future versions (air, plus, pro) will ship their own model and parameters — do not mix
across versions.

**Two phases:**

| Phase | Tool | Where |
|---|---|---|
| **Prepare** | `dpa4_prepare.py` | always local |
| **Execute** | `dp` CLI | **remote only** via Bohrium (`dpdisp` skill) |

Run the prepare script via `run_skill_script(skill_name="dpa4", script_name="dpa4_prepare.py", args="...")`.

---

## Recommended Workflow — Generate a force field with DPA4

When a user asks to generate or finetune a DPA4 force field, follow this decision tree.

> **Core principle:** The pretrained model is a **tool for exploration** (MD to discover
> candidate structures), **not a source of truth**. All training labels and evaluation
> benchmarks must come from **DFT calculations**. This is DFT-based fine-tuning,
> not distillation.

### Step 0 — Ask the user: Do you have a DFT-labelled dataset?

A "DFT-labelled dataset" means structures whose energy, forces, and (optionally) virial
were computed by DFT (VASP, ABACUS, etc.), **not** by a pretrained ML model.

- **Bench mode** (`agent_mode == "bench"`): skip this question — assume NO dataset and
  proceed directly to the "NO dataset" path below.

**If the user HAS a DFT-labelled dataset:**

1. Use at most **100 frames** for finetuning. If the dataset has more than 100 frames,
   the excess automatically becomes the test set.

2. Finetune DPA4, then run `dp test` on the test set with **both** the pretrained model
   and the finetuned model. Compare and report the improvement (energy/force MAE reduction).

3. No EOS benchmark needed — the test set already provides direct comparison.

**If the user has NO DFT-labelled dataset:**

Follow Phases A–D below. EOS benchmark is used for evaluation (no test set available).

---

#### Phase A — Determine system complexity & generate candidate structures

1. **Classify the system:**
   - **Simple systems** — bulk crystals, random alloys, simple compounds.
   - **Complex systems** — defects, dopants, surfaces, interfaces, transition states,
     high-entropy alloys, amorphous structures, etc.

2. **For complex systems: ask the user if they already have structure files.**
   If yes, use the user's structures as the starting point. If no, generate them
   using the `atomic-structure` skill (or `matcraft-kit` for surfaces/defects).

3. **Generate candidate structures** for MD exploration:
   - Use the pretrained model **only for MD** to explore configuration space.
   - Use the `atomic-structure` skill to build, supercell, and perturb structures.

4. **Atom count rules for DFT calculations:**
   | System type | Supercell? | Target atoms |
   |---|---|---|
   | Simple (bulk, alloy) | Yes, if needed | ~50 atoms |
   | Complex (defect, surface, …) | No | original cell size |

   > Keep each DFT structure at roughly **50 atoms** when possible. For complex systems,
   > do NOT supercell — use the original cell as-is.

#### Phase B — DFT labeling

Run DFT single-point calculations on all candidate structures to obtain energy, force,
and virial labels.

- Use the `vasp` or `abacus` skill for DFT input preparation and execution.
- See `concepts/dft-calculation` for guidance on choosing a DFT code.
- Job submission is handled by the `dpdisp` skill (Bohrium).

**Frame budget & training steps (no user dataset):**

| System type | Max DFT frames | Training steps | Warmup steps |
|---|---|---|---|
| Simple | **30** | **3 000** | **230** |
| Complex | **100** | **10 000** | **780** |

> When the user has no dataset, all generated frames go to training — no test phase.
> When the user has a dataset, up to **100 frames** are used for training; excess frames
> become the test set.

#### Phase C — EOS benchmark (no-dataset path, simple systems only)

When the user has **no dataset**, there is no test set to evaluate against. Instead,
run an EOS benchmark to compare pretrained vs finetuned models against DFT ground truth.

> **Only for simple systems and only when the user has no dataset.**
> If the user has a dataset, skip this phase — the test set provides direct comparison.

1. **DFT relaxation** — relax the unit cell to find the ground-state structure.
2. **Generate deformed structures** — create 11 structures with volumes from −5% to +5%
   of the equilibrium volume (uniform spacing).
3. **DFT single-point** — compute energy for all 11 structures.
4. **Model prediction** — predict energies for the same 11 structures using both the
   pretrained model and the finetuned model.
5. **Compare** — plot E(V) curves: DFT (ground truth) vs pretrained vs finetuned.

> Steps 1–3 can run **in parallel** with Phase B (dataset DFT labeling) to save time.

#### Phase D — Finetune & evaluate

> **Do NOT reuse any existing workdir.** Always run `dpa4_prepare.py` to create a fresh
> workdir with the correct input.json, train/test split, and model copy.

1. Prepare the finetune workdir:
   ```
   # No user dataset (simple): 30 frames → all for training, no test
   run_skill_script(
       skill_name="dpa4",
       script_name="dpa4_prepare.py",
       args="prepare-finetune --workdir ./finetune_001 --train_data dft_data.extxyz --base_model /path/to/dpa4_model --numb_steps 3000 --warmup_steps 230"
   )

   # No user dataset (complex): 100 frames → all for training, no test
   run_skill_script(
       skill_name="dpa4",
       script_name="dpa4_prepare.py",
       args="prepare-finetune --workdir ./finetune_001 --train_data dft_data.extxyz --base_model /path/to/dpa4_model --numb_steps 10000 --warmup_steps 780"
   )

   # User has dataset: 100 frames for training, rest for test
   run_skill_script(
       skill_name="dpa4",
       script_name="dpa4_prepare.py",
       args="prepare-finetune --workdir ./finetune_001 --train_data user_data.extxyz --base_model /path/to/dpa4_model --numb_steps 10000 --warmup_steps 780 --max_train_frames 100"
   )
   ```

2. Submit finetune job on Bohrium via the `dpdisp` skill.
   - If test data exists: include `test_data` in `forward_files`, run `dp test` after training.
   - If no test data: only `train_data` in `forward_files`, skip `dp test`.

3. **Evaluate:**
   - **User has dataset:** Run `dp test` on the test set with **both** the pretrained
     model and the finetuned model. Report the comparison:
     - Pretrained: energy MAE = X, force MAE = Y
     - Finetuned: energy MAE = X', force MAE = Y'
     - Improvement: energy MAE reduced by Z%, force MAE reduced by W%
   - **No dataset:** Compare EOS curves (Phase C). Report DFT vs pretrained vs finetuned.

---

## Environment variables

DPA4 requires **all** standard Bohrium variables plus two DPA4-specific variables:

| Variable | Description |
|---|---|
| `BOHRIUM_EMAIL` | Bohrium account e-mail |
| `BOHRIUM_PASSWORD` | Bohrium account password |
| `BOHRIUM_PROJECT_ID` | Bohrium project ID (integer) |
| `BOHRIUM_DPA4_MACHINE` | Machine/scass type for training, e.g. `1 * NVIDIA V100_32g` |
| `BOHRIUM_DPA4_IMAGE` | Container image URI with DPA4-compatible deepmd-kit (e.g. `registry.dp.tech/dptech/dp/native/hub/custom_images/dpa4:260522-1779446700`) |
| `BOHRIUM_DPA4_MODEL` | Path to the DPA4 pretrained model file |

> **Note:** The base model for DPA4 must be a **file** (not a directory).
> The prepare script copies it into the workdir for remote submission.

---

## Phase 1 — Preparation

`dpa4_prepare.py` converts raw structures to `deepmd/npy` and writes `input.json`.
Each sub-command prints a JSON summary with the exact `dp` command for Phase 2.

Check `BOHRIUM_DPA4_MODEL` for the default pretrained model, or pass `--base_model` explicitly.

### 1a. Finetune a DPA4 model (single-task)

```
run_skill_script(
    skill_name="dpa4",
    script_name="dpa4_prepare.py",
    args="prepare-finetune --workdir <workdir> --train_data file1.xyz [file2.xyz ...] --base_model /path/to/dpa4_model [--version neo] [--numb_steps 10000] [--warmup_steps 780] [--max_train_frames 100] [--type_map Fe Ni Cu ...] [--copy_model]"
)
```

The `--version` flag selects the matching input.json template. Default is `neo`.
The `--max_train_frames` flag caps training frames; excess goes to `test_data/` (0 = all for training, no test).

**Contents of `<workdir>` after preparation:**

| Path | Description |
|---|---|
| `input.json` | Training configuration for `dp --pt train` (version-specific format) |
| `train_data/` | deepmd/npy training split |
| `test_data/` | deepmd/npy test split (only when `--max_train_frames` is set and data exceeds it) |
| `<model>` | Copy of the DPA4 pretrained model file |

> **Remote submission:** Include `test_data` in `forward_files` only when it exists.
> The prepare script copies the model file into the workdir.

### 1b. Convert test data to deepmd/npy

For standalone testing or benchmark evaluation:

```
run_skill_script(
    skill_name="dpa4",
    script_name="dpa4_prepare.py",
    args="convert-data --data test.extxyz [--outdir ./test_data] [--mixed_type] [--nframes 200]"
)
```

The command prints a JSON result with `system_dirs` and `dp_test_commands`.

---

## Phase 2 — Execution (remote on Bohrium)

### Step 1 — Prepare locally (see Phase 1)

> Use the **fresh workdir** generated by `dpa4_prepare.py`. Do NOT point to old directories.

### Step 2 — Generate submission.template.json

Use `remote_profile` with an `input_data` sub-object for Bohrium.

**Finetune only (no test data):**

```json
{
  "work_base": ".",
  "machine": {
    "batch_type": "Bohrium",
    "context_type": "BohriumContext",
    "local_root": ".",
    "remote_profile": {
      "email": "${BOHRIUM_EMAIL}",
      "password": "${BOHRIUM_PASSWORD}",
      "program_id": ${BOHRIUM_PROJECT_ID},
      "input_data": {
        "job_type": "container",
        "log_file": "train_log",
        "scass_type": "${BOHRIUM_DPA4_MACHINE}",
        "platform": "ali",
        "image_name": "${BOHRIUM_DPA4_IMAGE}"
      }
    }
  },
  "resources": { "group_size": 1 },
  "task_list": [
    {
      "command": "dp --pt train input.json --skip-neighbor-stat --finetune <model> > train_log 2>&1 && dp --pt freeze -c model.ckpt.pt -o frozen",
      "task_work_path": "./finetune_001",
      "forward_files": ["input.json", "train_data", "<model>"],
      "backward_files": ["model.ckpt.pt", "frozen.pt2", "lcurve.out", "train_log"]
    }
  ]
}
```

**Finetune + test (user has dataset):**

```json
{
  "work_base": ".",
  "machine": { "..." : "..." },
  "resources": { "group_size": 1 },
  "task_list": [
    {
      "command": "dp --pt train input.json --skip-neighbor-stat --finetune <model> > train_log 2>&1 && dp --pt freeze -c model.ckpt.pt -o frozen && dp --pt test -m frozen.pt2 -s test_data -d result-test -l log-test",
      "task_work_path": "./finetune_001",
      "forward_files": ["input.json", "train_data", "test_data", "<model>"],
      "backward_files": ["model.ckpt.pt", "frozen.pt2", "lcurve.out", "train_log", "log-test", "result-test*"]
    }
  ]
}
```

> `<model>` is the base model name inside the workdir — the prepare script prints it as
> `model_name` in its JSON output.

### Step 3 — Substitute, validate, and submit

```bash
envsubst '${BOHRIUM_EMAIL} ${BOHRIUM_PASSWORD} ${BOHRIUM_PROJECT_ID} ${BOHRIUM_DPA4_MACHINE} ${BOHRIUM_DPA4_IMAGE}' \
    < submission.template.json > submission.json

uv run -m json.tool submission.json >/dev/null
uvx --with dpdispatcher dargs check -f dpdispatcher.entrypoints.submit.submission_args submission.json

# Always use --with oss2 for Bohrium jobs
uvx --from dpdispatcher --with oss2 dpdisp submit submission.json
```

For long-running training jobs, wrap in `tmux` to survive SSH disconnects:

```bash
tmux new-session -d -s dpa4_train \
    "uvx --from dpdispatcher --with oss2 dpdisp submit submission.json"
tmux ls
```

---

## DPA4 Command Reference

DPA4 uses different flags compared to DPA-1/DPA-2.

```bash
# Finetune from a pretrained DPA4 model (--skip-neighbor-stat is required for train only)
dp --pt train input.json --skip-neighbor-stat --finetune <model> > train_log 2>&1

# Freeze the trained model
dp --pt freeze -c model.ckpt.pt -o frozen

# Test (frozen model)
dp --pt test -m frozen.pt2 -s <test_data_dir> -d result-test -l log-test

# Test (pretrained model directory — for standalone evaluation)
dp --pt test -m <model> -s <test_data_dir> -d result-test -l log-test
```

Key differences from DPA-1/DPA-2:
- `--skip-neighbor-stat` required for training only (not test/freeze)
- No `--use-pretrain-script` or `--model-branch` flags
- Freeze produces `frozen.pt2` (not `frozen_model.pb`)
- Base model is a **file** (e.g. zip archive), not a directory

---

## Output files

| File | Description |
|---|---|
| `model.ckpt.pt` | Saved PyTorch checkpoint |
| `frozen.pt2` | Frozen model for inference |
| `lcurve.out` | Training loss curve (step, energy MAE, force MAE, …) |
| `train_log` | Training stdout/stderr |
| `result-test*` | Test result files (per-frame energies, forces, virials) |
| `log-test` | Test evaluation log |

---

## Constraints

- `dpa4_prepare.py` requires `ase`, `dpdata`, and `numpy` in the local Python environment.
- All input structures must be **DFT-labelled** (energy + forces). Unlabeled structures
  raise an error during dpdata export.
- Base model must be a **file** (not a directory). Model version and input parameters must
  match exactly — do not mix across versions.
- `deepmd/npy` systems are written per chemical formula; use `--mixed_type` for variable
  composition within a single directory.
- All `task_work_path` entries must share the same `work_base` (dpdispatcher requirement).
- **Frame budget & training steps (no user dataset):** simple ≤30 frames / 3 000 steps / 230 warmup; complex ≤100 frames / 10 000 steps / 780 warmup.
- **User dataset:** cap training at 100 frames (`--max_train_frames 100`); excess becomes test set.
- **Atom count:** ~50 atoms/DFT structure. Simple systems may supercell; complex systems
  (defects, dopants, surfaces, interfaces, transition states, high-entropy alloys) must NOT.
- **EOS benchmark** is for **no-dataset path only** (simple systems). When the user has a
  dataset, the test set replaces the EOS benchmark for evaluation.
- **Evaluation always compares pretrained vs finetuned** — either via EOS curves (no dataset)
  or via test-set MAE (user has dataset).
