from __future__ import annotations

import os
from typing import Literal

from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from pydantic import BaseModel, Field

from ...constants import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL

# deprecated
_DB_GUIDE = """
You can only query the `dataset_info` table. Each row corresponds to a dataset in the ASE database and
exposes the following columns (all strings unless noted):
- ID (INTEGER): unique identifier for the dataset.
- Elements (TEXT): hyphen-joined, lexicographically sorted symbols (e.g., "Al-Fe-Si").
- Type (TEXT): dataset system type (Cluster, Bulk, Surface, Interface, etc.).
- Fields (TEXT): research domain (Alloy, Catalysis, Semiconductor, ...).
- Entries (INTEGER): number of structures in the dataset.
- Source (TEXT): URL or DOI for provenance.
- Path (TEXT): relative path under ./ai-database pointing to the *.db file.

Rules:
1. Only search the data by the `Elements` column if the user does not provide other information except the chemical elements or formulas. 
2. Always try to match elements EXACTLY and return all the information of an entry, e.g. `SELECT * FROM dataset_info WHERE Elements = 'Al-Fe-Si'`. Try not to use the keyword 'LIKE' for queries unless the result of the last query was 0.
3. The response should be pure SQL code. 
4. Never write UPDATE/INSERT/DELETE/ALTER/PRAGMA or attach other tables.
5. Use LIMIT only if a numeric cap is given (assume 20 if the user says "a few" or "top" without a number).
6. Multi-condition filters should put AND/OR explicitly and parenthesize OR groups.
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
            "Single SELECT statement targeting dataset_info. Do not include a trailing semicolon"
            " or markdown fences."
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
You are a SQL agent, accept the prompts of database querying from the user and return 
the corresponding SQL code. Below is the description of the database and the rules you should follow:
{_DB_GUIDE}

Formatting requirements:
- When matching text fields, wrap literals in single quotes and escape embedded quotes if necessary.
- Apply LIMIT only if the user supplies one (or you infer "top"/"few" -> 20) and include ORDER BY when
  returning ranked results.
- **NEVER use LIKE with wildcards for the Elements column** (e.g., LIKE 'Si-%' or LIKE '%Si%'). 
  Always use exact match with the = operator. If the user provides incomplete element information,
  ask for clarification or return no results rather than using LIKE patterns.
- For other text columns (Type, Fields, Source), LIKE may be used if fuzzy matching is explicitly requested.

Output contract:
- Respond with JSON conforming to SqlAgentOutput (sql/rationale/confidence). No additional keys.
- The sql string must be a single line or multi-line SELECT without a trailing semicolon.
- If assumptions are required, mention them in the rationale and set confidence="low".

Do not call any tools or external services. Focus exclusively on translating the request into SQL.
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
    description="Produces safe SELECT statements over dataset_info for the Database Agent.",
    instruction=_SQL_AGENT_INSTRUCTION,
    input_schema=SqlAgentInput,
    output_schema=SqlAgentOutput,
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
)
