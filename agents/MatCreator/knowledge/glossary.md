# Domain Glossary

Use this reference to correctly interpret user messages.

## Models & Methods

- **DPA** / **DPA model** / **DPA-2** / **DPA-3**: Deep Potential Atomistic model — a pre-trained universal machine-learning interatomic potential (force field). Usage: zero-shot inference, base model for fine-tuning or as teacher model in distillation workflows.
- **DP** / **DeePMD**: Deep Potential Molecular Dynamics — the broader framework for DPA models.
- **ML force field** / **MLFF** / **MLP**: Machine-learning potential / force field.

## Workflows

- **PFD**: Pre-training → Fine-tuning → Distillation — the iterative active-learning workflow for ML force fields.
- **fine-tune** / **finetune**: Adapt a pre-trained DPA model to a target system using DFT-labeled data (ABACUS).
- **distillation** / **distill**: Train a smaller/faster student model by using a DPA teacher model for labeling instead of DFT.
- **active learning**: Iterative loop of structure exploration, uncertainty-based selection, labeling, and training until convergence.

## Simulation & Labeling

- **ABACUS**: Ab initio Calculation for UT Austin Software — an open-source DFT code used for first-principles labeling.
- **VASP**: Vienna Ab initio Simulation Package — another DFT code supported for labeling.
- **MD** / **molecular dynamics**: Simulation used to explore the configuration space of a material.
- **DFT**: Density Functional Theory — quantum mechanical calculation used for ground-truth labeling.

## Data & Training Terms

- **convergence criterion**: A threshold (e.g., force error) that determines when the iterative PFD loop can stop.
- **curation** / **data curation**: Entropy/uncertainty-based selection of diverse, informative structures from MD trajectories.
- **supercell**: Enlarged periodic simulation cell used for MD exploration.
- **perturbation**: Random displacement of atom positions or cell shape to generate diverse initial structures.
