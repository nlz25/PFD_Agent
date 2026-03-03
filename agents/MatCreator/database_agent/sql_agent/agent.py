from __future__ import annotations

import os
from typing import Literal

from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from pydantic import BaseModel, Field

from ...constants import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL

_SCHEMA_GUIDE = """
The info database has three tables. Always prefer JOINs over subqueries.

TABLE: nodes
  node_id     INTEGER PK   -- unique DFT-setting group
  name        TEXT         -- equals the Fields collection label (e.g. "Catalysis", "Alloy")
  functional  TEXT         -- PBE | PBEsol | LDA | SCAN | HSE06 ...
  code        TEXT         -- VASP | ABACUS | QE | CP2K ...
  description TEXT
  created_at  TEXT

TABLE: datasets
  dataset_id  INTEGER PK
  node_id     INTEGER FK -> nodes.node_id
  elements    TEXT         -- hyphen-joined sorted symbols, e.g. "Fe-O"
  n_elements  INTEGER      -- number of distinct elements
  system_type TEXT         -- Bulk | Cluster | Surface | Interface ...
  field       TEXT         -- collection label (same as nodes.name)
  entries     INTEGER      -- frame count in the .db file
  source      TEXT         -- URL / DOI / provenance label
  path        TEXT         -- relative path to the ASE .db file
  has_forces  INTEGER      -- 1 if forces are stored
  has_stress  INTEGER      -- 1 if stress is stored
  has_energy  INTEGER      -- 1 if energies are stored
  energy_min  REAL
  energy_max  REAL
  created_at  TEXT


RULES:
1. EXACT composition: use datasets.elements = 'Fe-O' (sorted, hyphen-joined).
2. NEVER use LIKE with wildcards on the elements column.
3. LIKE is permitted on system_type, field, source for fuzzy text matching.
4. Always SELECT d.path so the caller can open the .db file.
5. Never write UPDATE/INSERT/DELETE/DROP/ALTER/PRAGMA or ATTACH.
6. Use LIMIT only if the user specifies a cap (infer 20 for "a few" / "top").
7. Parenthesize OR groups; use explicit AND/OR.

EXAMPLES (follow these patterns):
1) Exact formula in a given domain/name
    User: "Find Si datasets in Catalysis"
    SQL: SELECT d.path AS path, d.elements AS elements, d.system_type AS system_type,
                    d.entries AS entries, n.name AS node_name
          FROM datasets d
          JOIN nodes n ON d.node_id = n.node_id
          WHERE d.elements = 'Si' AND n.name = 'Catalysis'

2) Exact formula + DFT functional
    User: "Find Si-O datasets with PBE"
    SQL: SELECT d.path AS path, d.elements AS elements, d.entries AS entries,
                    n.functional AS functional, n.name AS node_name
          FROM datasets d
          JOIN nodes n ON d.node_id = n.node_id
          WHERE d.elements = 'O-Si' AND n.functional = 'PBE'

3) Fuzzy system type with exact formula
    User: "A few bulk Si datasets"
    SQL: SELECT d.path AS path, d.elements AS elements, d.system_type AS system_type,
                    d.entries AS entries, n.name AS node_name
          FROM datasets d
          JOIN nodes n ON d.node_id = n.node_id
          WHERE d.elements = 'Si' AND d.system_type LIKE '%Bulk%'
          LIMIT 20

4) Prohibited pattern (do not generate)
    SELECT ... FROM dataset_elements e WHERE e.element = 'Si'
"""


class SqlAgentInput(BaseModel):
    """Structured request passed from the database agent."""

    request: str = Field(
        ...,
        description=(
            "Natural-language description of the dataset query the user wants. Include any"
            " prior context the SQL agent should know."
        ),
    )
    preferred_limit: int = Field(
        default=1000,
        description="Optional max rows; positive integer if provided."
    )


class SqlAgentOutput(BaseModel):
    """Response contract for SQL generation."""

    sql: str = Field(
        ...,
        description=(
            "Single SELECT statement targeting nodes/datasets."
            " No trailing semicolon or markdown fences."
        ),
    )
    rationale: str = Field(
        ...,
        description="One short sentence (<=25 words) summarizing how the SQL satisfies the request.",
        max_length=1000,
    )
    confidence: Literal["high", "medium", "low"] = Field(
        default="medium",
        description="Heuristic confidence in the generated SQL. Use 'low' if assumptions were necessary.",
    )


_SQL_AGENT_INSTRUCTION = f"""
You are a SQL agent. Accept natural-language dataset queries and return the corresponding SQL.
Below is the schema and the rules you must follow:

{_SCHEMA_GUIDE}

Formatting requirements:
- Wrap text literals in single quotes; escape embedded single quotes.
- Include ORDER BY when ranked results are implied.
- Use explicit column aliases (e.g. d.path AS path) to disambiguate JOINs.

Output contract:
- Respond with JSON conforming to SqlAgentOutput (sql / rationale / confidence). No extra keys.
- The sql field must be a single SELECT statement with no trailing semicolon.
- Set confidence="low" and describe assumptions in rationale when the request is ambiguous.

Do not call any tools or external services.
"""


_model_name = os.environ.get("LLM_MODEL", LLM_MODEL)
_model_api_key = os.environ.get("LLM_API_KEY", LLM_API_KEY)
_model_base_url = os.environ.get("LLM_BASE_URL", LLM_BASE_URL)

sql_agent = LlmAgent(
    name="database_sql_agent",
    model=LiteLlm(
        model=_model_name,
        base_url=_model_base_url,
        api_key=_model_api_key,
    ),
    description="Produces safe SELECT statements over nodes/datasets for the Database Agent.",
    instruction=_SQL_AGENT_INSTRUCTION,
    input_schema=SqlAgentInput,
    output_schema=SqlAgentOutput,
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
)
