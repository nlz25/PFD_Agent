---
name: guide_for_model_training
description: General guideline for model training.
tags: [model_training, fine-tuning, pre-trained, workflow]
allowed_agents: [dpa_agent, abacus_agent, structure_agent, plot_agent]
triggers: [train, training, model, fine-tune, finetune, pre-trained, force field]
---
Model training is a dynamic, iterative process. Follow this decision flow and revisit earlier steps if results are unsatisfactory.

## Decision Flow

### Stage 1 — Check the Pre-trained Model
- Always try the pre-trained model first.
- If the available dataset already contains the target system, validate the pre-trained model against it before doing anything else.
- If validation passes, stop here — no training is needed.

### Stage 2 — Fine-tune on Available Data
- If dataset available and pre-trained model does not perform adequately, fine-tune it using the existing dataset.
- Prefer this over more complex workflows whenever data is available.
- Evaluate the fine-tuned model; if results are sufficient, stop here.

### Stage 3 — PFD Workflow (Last Resort)
- If no available dataset or fine-tuned model is insufficient, launch the PFD workflow.
- PFD iteratively generates new data via MD exploration, labels it, and retrains until convergence.
- See `pfd-finetuning` skill for full details.

## General Notes
- Always evaluate the model after each step before proceeding to the next.
- Prefer the simplest effective strategy — avoid PFD if fine-tuning suffices.
- Each step may require multiple rounds of interaction and parameter adjustment.
