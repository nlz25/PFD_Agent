---
name: mattergen
description: Generate crystal structures with MatterGen through its official CLI.
tags: [MatterGen, crystal generation, inverse design]
tools: [run_bash]
dependent_skills: []
---
# MatterGen Skill

Use this skill to generate inorganic crystal structures with MatterGen through the helper script `generate.py` in this directory. The script creates a timestamped output directory, runs MatterGen from the configured environment, captures logs, and returns a JSON summary.


## Available Pre-trained Checkpoints

Common upstream checkpoint names include:
- `mattergen_base`
- `mp_20_base`
- `chemical_system`
- `space_group`
- `dft_mag_density`
- `dft_band_gap`
- `ml_bulk_modulus`
- `dft_mag_density_hhi_score`
- `chemical_system_energy_above_hull`


## Generation

Prefer running `generate.py`.

Parameter rule:
- If `--pretrained-name mattergen_base` is used, `--properties-to-condition-on` is optional and generation can be unconditional.
- If any other pre-trained checkpoint is used, `--properties-to-condition-on` must be provided and should match the property or properties expected by that checkpoint.

Checkpoint to `--properties-to-condition-on` mapping:
- `mattergen_base`: not required
- `mp_20_base`: no canonical conditioning field is documented in the local MatterGen sources; confirm the expected conditioning dictionary before use
- `chemical_system`: `"{'chemical_system': 'Li-O'}"`
- `space_group`: `"{'space_group': 62}"`
- `dft_mag_density`: `"{'dft_mag_density': 0.5}"`
- `dft_band_gap`: `"{'dft_band_gap': 1.5}"`
- `ml_bulk_modulus`: `"{'ml_bulk_modulus': 120.0}"`
- `dft_mag_density_hhi_score`: `"{'dft_mag_density': 0.5, 'hhi_score': 2000.0}"`
- `chemical_system_energy_above_hull`: `"{'chemical_system': 'Li-O', 'energy_above_hull': 0.05}"`

### Unconditional generation

```bash
python agents/MatCreator/knowledge/skills/mattergen/generate.py \
  --pretrained-name mattergen_base \
  --batch-size 16 \
  --num-batches 1
```

### Property or Multi-property conditioned generation

Use this form for every checkpoint other than `mattergen_base`.

```bash
python agents/MatCreator/knowledge/skills/mattergen/generate.py \
  --pretrained-name chemical_system_energy_above_hull \
  --batch-size 16 \
  --properties-to-condition-on "{'energy_above_hull': 0.05, 'chemical_system': 'Li-O'}" \
  --diffusion-guidance-factor 2.0
```


## Expected Outputs

The timestamped output directory typically contains:
- `generated_crystals_cif.zip`
- `generated_crystals.extxyz`
- `generation.log`

Report at minimum:
- The generated `extxyz` path
