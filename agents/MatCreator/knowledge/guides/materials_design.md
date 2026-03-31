---
name: materials-design
description: Iterative workflow for materials design through candidate generation, property prediction, screening, and refinement.
skills: [mattergen, structure_conversion, ase_deepmd, cgcnn_predictor]
tags: [materials design, inverse design, iterative, screening, generation]
---
Use this guide when the goal is to discover or optimize crystal materials by repeatedly generating candidates, relaxing them, predicting target properties, and narrowing the search space.

## Standard Loop

1. Define the design target
- Confirm the chemical system, target properties, and screening rules, including thresholds, ranking metrics, batch size, and maximum iterations.

2. Generate candidate structures
- If a generative model is available, prefer `mattergen` to create a batch of candidate crystals.
- Always save generated candidates to a dedicated iteration directory and report absolute paths.

3. Relax generated structures
- Before screening, use `ase_deepmd` to relax generated structures with a suitable DeePMD potential.
- Save relaxed structures to a dedicated output path or directory and report the absolute paths.
- If relaxation fails for some candidates, report which ones failed and continue with the valid relaxed structures when appropriate.

4. Convert relaxed structures for downstream property prediction
- Convert or reorganize generated structures into the format required by the downstream predictor or validation workflow.
- If the generated file contains multiple structures, split or reorganize it as needed for the next tool.
- Report the converted output path or directory, the number of structures prepared, and the final structure format used.

5. Predict properties and rank candidates
- Use an available property-prediction workflow on the screened candidate set, such as `cgcnn_predictor` or another model/tool appropriate for the target property.
- Always report the exact model, checkpoint, or tool used, the property values returned, and the structure paths associated with each result.
- Rank or filter candidates according to the user-defined objective.

6. Select the next round
- Keep the top candidates and summarize why they were selected.
- If no candidate passes the screening rule, broaden the search by adjusting generation conditions, composition ranges, or the number of samples.
- If the candidate pool collapses to near-duplicates, increase diversity in the next generation round.

7. Iterate as needed
- The agent may decide whether to continue iterating based on screening results, diversity, confidence, and remaining budget.
- Continue the loop when the current candidates are still weak, too similar, or insufficient for the target.
- Stop when enough strong candidates are found, rankings become stable, or the budget is exhausted.

8. Optionally validate the final shortlist with DFT
- After screening, optionally run DFT on a few top shortlisted candidates.
- Report which candidates were validated, what workflow was used, and whether the screening results were confirmed.
- If DFT strongly disagrees with the screening model, summarize the mismatch and decide whether another iteration is needed.

## Recommended Defaults

- Start with 10 to 100 generated candidates per round, depending on compute budget.
- Relax candidates before property prediction when a reliable DeePMD potential is available.
- Use a fast property predictor for early screening and save stricter evaluation for the final shortlist.
- Reserve DFT for optional final validation of the best few candidates rather than for every iteration.
- Keep a per-iteration summary table with generated, relaxed, and converted structure paths, prediction model, predicted properties, and pass/fail status.

## Decision Notes

- Prefer `mattergen` when the search space is broad or open-ended.
- Prefer batch prediction when many candidates are available.
- If the user wants physically validated final candidates, finish the screening loop first and then optionally hand the shortlist to a DFT workflow for higher-fidelity validation.
