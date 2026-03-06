---
name: database_search
description: General guidelines for querying materials datasets from the database.
tags: [database, query, dataset, composition, frames, ASE]
allowed_agents: [database_agent]
triggers: [search, query, dataset, find, composition, frames, structures, database]
---

## Querying Datasets

Always follow two steps: **find the domain node first, then query frames within it.**

### Step 1 — Find the domain node

Use `database_sql_agent` → `validate_sql_code_query` → `query_information_database`.

Search by `name`, `field`, `type`, `code`, etc. If unsure, list all nodes and ask the user to confirm. 

Note: Usually you can't directly search for domain datasets by chemical composition.

### Step 2 — Query frames in the selected dataset

Use `query_compounds(selection, db_path)` on the `path` returned in Step 1.

- `selection`: a string selector passed to ASE DB, e.g. `'Si'` (any formula containing Si), `'Si,O'` (contains both), `'formula=Si32'` (exact), or `'energy<-1.0'`.
- `db_path`: absolute path to the ASE `.db` file.
- Returns `count`, `unique_formulas`, and `sample_ids`.

Note: datasets are elements-centric — `'Si'` will also match SiC, SiO₂, etc. Use `'formula=Si32'` for exact matches.
