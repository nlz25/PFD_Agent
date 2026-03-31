---
name: cgcnn_predictor
description: Predict crystal properties with local CGCNN models.
tools: []
dependent_skills: []
---
# CGCNN Predictor Skill

Use this skill to predict crystal properties with a local CGCNN setup, including regression targets such as formation energy, band gap, Fermi energy, or elastic properties, as well as classification tasks such as metal versus semiconductor.



If the help command runs without import errors, the environment is ready.

## Dataset Format

Prepare the dataset directory as:

```text
dataset_dir/
├── id_prop.csv
├── atom_init.json
├── id0.cif
├── id1.cif
├── ...
```

Requirements:
- `id_prop.csv` must have two columns: crystal ID and target value
- For prediction, the second column can be a placeholder number, but it must exist
- Each crystal ID must match a CIF filename without the `.cif` suffix
- `atom_init.json` should usually be copied from `data/sample-regression/atom_init.json` under `CGCNN_ROOT`

This dataset layout is required by the upstream CGCNN prediction workflow.

## Common Pre-trained Models

Common checkpoints in `pre-trained/` include:
- `formation-energy-per-atom.pth.tar`
- `final-energy-per-atom.pth.tar`
- `band-gap.pth.tar`
- `efermi.pth.tar`
- `bulk-moduli.pth.tar`
- `shear-moduli.pth.tar`
- `poisson-ratio.pth.tar`
- `semi-metal-classification.pth.tar`

## Prediction Command

Run prediction from cgcnn dir:

```bash
python predict.py <model_path> <dataset_dir>
```

Examples:

```bash
conda init
conda activate cgcnn
cd "${CGCNN_ROOT}"
python predict.py pre-trained/formation-energy-per-atom.pth.tar dataset_dir
```

The run generates `test_results.csv` in the dataset directory. For classification tasks, predicted values are probabilities between 0 and 1.

## What to Report

- The absolute path of the dataset directory
- The exact model or checkpoint used
- The absolute path of the generated `test_results.csv`
- The prediction result for each crystal together with its CIF path or crystal ID
