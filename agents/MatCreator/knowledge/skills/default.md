---
name: default
description: General workflows for single-step or sequential tasks without iterative training loops.
tags: [general, sequential, simple]
allowed_agents: [database_agent, structure_agent, abacus_agent, vasp_agent, dpa_agent, plot_agent]
triggers: [pipeline, sequential, one-off, then, after, workflow]
workflow_type: default
---
General workflow execution:
- Execute tasks according to the approved plan.
- Follow plan steps sequentially and delegate to the appropriate sub-agent.
- Report results with absolute paths and key metrics after each step.
- On errors, report exact message, propose an alternative, and ask for confirmation before deviation.
