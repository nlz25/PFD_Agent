from __future__ import annotations

import os
from typing import Literal

from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from pydantic import BaseModel, Field

from ...constants import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL

_SCHEMA_GUIDE = """
The info database has a single table: nodes.
Each row represents one domain dataset node.

TABLE: nodes
  node_id     INTEGER PK   -- unique node identifier
  name        TEXT         -- domain label (e.g. "Domain_Cluster", "Domain_SemiCond")
  description TEXT         -- human-readable description of the dataset contents
  system_type TEXT         -- Bulk | Cluster | Surface | Interface ...
  field       TEXT         -- scientific/application field (e.g. Catalysis, Semiconductor)
  entries     INTEGER      -- total frame count in the .db file
  source      TEXT         -- URL / DOI / provenance label
  path        TEXT         -- relative path to the ASE .db file
  code        TEXT         -- VASP | ABACUS | QE | CP2K ...
  functional  TEXT         -- PBE | PBEsol | LDA | SCAN | HSE06 ...
  pseudopot   TEXT         -- pseudopotential / PAW label
  metadata    TEXT         -- extra metadata string
  created_at  TEXT


RULES:
1. LIKE is permitted on name, system_type, field, source, description for fuzzy text matching.
2. Always SELECT path so the caller can open the .db file.
3. Never write UPDATE/INSERT/DELETE/DROP/ALTER/PRAGMA or ATTACH.
4. Use LIMIT only if the user specifies a cap (infer 20 for "a few" / "top").
5. Parenthesize OR groups; use explicit AND/OR.
6. No JOINs are needed — all columns live in the single nodes table.

EXAMPLES (follow these patterns):
1) Find nodes by field
    User: "Find Catalysis datasets"
    SQL: SELECT name, system_type, field, entries, path
          FROM nodes
          WHERE field = 'Catalysis'

2) Find nodes by DFT functional
    User: "Find datasets computed with PBE"
    SQL: SELECT name, functional, entries, path
          FROM nodes
          WHERE functional = 'PBE'

3) Find nodes by domain name
    User: "Show me the Domain_Cluster node"
    SQL: SELECT name, description, system_type, entries, path
          FROM nodes
          WHERE name = 'Domain_Cluster'

4) Fuzzy filter on system type
    User: "A few bulk datasets"
    SQL: SELECT name, system_type, field, entries, path
          FROM nodes
          WHERE system_type LIKE '%Bulk%'
          LIMIT 20

5) Combined filter
    User: "Cluster datasets computed with CP2K"
    SQL: SELECT name, code, functional, entries, path
          FROM nodes
          WHERE system_type LIKE '%Cluster%' AND code = 'CP2K'

6) List all available nodes
    User: "What datasets are available?" / "List all nodes"
    SQL: SELECT name, system_type, field, entries, path
          FROM nodes
          ORDER BY name
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
- No table aliases or JOINs are needed — query the nodes table directly.

Output contract:
- Respond with JSON conforming to SqlAgentOutput (sql / rationale / confidence). No extra keys.
- The sql field must be a single SELECT statement with no trailing semicolon.
- Set confidence="low" and describe assumptions in rationale when the request is ambiguous.
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
