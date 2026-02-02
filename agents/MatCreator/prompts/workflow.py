"""Workflow instruction templates and guidance search utilities for MatCreator."""

from typing import Dict, List, Optional, Literal
from dataclasses import dataclass

from .subagents import SUBAGENTS


@dataclass
class WorkflowGuidance:
    """Container for workflow-specific execution guidance."""
    
    workflow_type: str
    instruction: str
    tags: List[str]
    keywords: List[str]
    description: str
    allowed_agents: List[str]


# Workflow instruction templates with metadata
WORKFLOW_GUIDANCE_LIBRARY = {
    "pfd": WorkflowGuidance(
        workflow_type="pfd",
        tags=["iterative", "active_learning", "model training"],
        keywords=[
        ],
        description="DO NOT USE unless neccessary! Iterative and robust pre-training, fine-tuning and distillation (PFD) workflow that generate machine learning force fields from pre-trained model.",
        allowed_agents=["structure_agent", "abacus_agent", "dpa_agent","plot_agent"],
        instruction="""
    PFD workflows coordinate iterative fine-tuning or distillation of ML force fields from a pre-trained model.

    **Standard loop**
    1. Structure building → 2. MD exploration → 3. Entropy-based data curation → 4. Labeling (ABACUS or DPA) → 5. Training → 6. Convergence check → repeat until criteria or max iterations.

    **Workflow variants**
    - Fine-tuning: ABACUS DFT labels, then continue training the existing model on ALL collected data.
    - Distillation: DPA teacher labels, then train a new student model from scratch.

    **Key parameters to confirm**
    - General: task type (fine-tune/distill), max iterations (default 1), convergence criterion (e.g., 0.002 eV/atom).
    - Structure: initial structure (built or given), supercell sizes, number and magnitude of perturbations.
    - MD: number of starting configs, ensemble (NVT/NPT/NVE), temperature(s), total time (ps), timestep/steps, save interval.
    - Curation: max_sel and optional chunk_size.
    - Labeling: ABACUS kspacing (default 0.2) or DPA head (default "MP_traj_v024_alldata_mixu").
    - Training: target epochs and train/test split.
    - Interaction: chat (confirm each step) or non-interactive batch (default; proceed if no error).

    **Specialized sub-agents**
    - structure_agent: structure building, perturbation, entropy-based curation.
    - abacus_agent: DFT calculations and labeling with ABACUS.
    - dpa_agent: MD simulation, labeling with DPA, and model training.
    - plot_agent: Visualization and plotting of results.
    **Outputs and failures**
    - After each step: report absolute artifact paths and key metrics; propose the next step.
    - On error: show the exact message and propose one concrete alternative; confirm before proceeding.

    **Response format**
    - Plan | Action (tool call) | Result (artifacts + metrics, absolute paths) | Next
    """
    ),
    
    "default": WorkflowGuidance(
        workflow_type="default",
        tags=["general", "sequential", "simple"],
        keywords=[
            "then", "after", "pipeline", "sequential", 
            "workflow", "series of", "followed by", "single", "one-off"
        ],
        description="General workflows for single or multi-step tasks without iteration",
        allowed_agents=list(SUBAGENTS.keys()),
        instruction="""
**General Workflow Execution:**

Execute tasks according to the approved plan.

**Execution rules:**
- Follow plan steps sequentially, delegate to appropriate sub-agents
- Report results with absolute paths and key metrics after each step
- On errors: report exact message, propose alternative, confirm with user
"""
    )
}


def search_workflow_guidance(
    workflow_type: Literal["pfd", "default"],
    user_query: Optional[str] = None
) -> WorkflowGuidance:
    """
    Search and retrieve workflow guidance based on workflow type.
    
    Args:
        workflow_type: Type of workflow (pfd, default)
        user_query: Optional user query for additional context matching
    
    Returns:
        WorkflowGuidance object with instruction and metadata
    """
    # Direct lookup by workflow type
    if workflow_type in WORKFLOW_GUIDANCE_LIBRARY:
        return WORKFLOW_GUIDANCE_LIBRARY[workflow_type].instruction
    
    # Fallback to default if type not found
    return WORKFLOW_GUIDANCE_LIBRARY["default"].instruction


def get_all_workflow_types() -> List[str]:
    """Return list of all available workflow types."""
    return list(WORKFLOW_GUIDANCE_LIBRARY.keys())


def get_workflow_description(workflow_type: str) -> str:
    """Get brief description of a workflow type."""
    guidance = WORKFLOW_GUIDANCE_LIBRARY.get(workflow_type)
    return guidance.description if guidance else "Unknown workflow type"


def get_allowed_agents_for_workflow(workflow_type: str) -> List[str]:
    """Return allowed sub-agent names for a workflow type (defaults to all)."""
    guidance = WORKFLOW_GUIDANCE_LIBRARY.get(workflow_type)
    if guidance and guidance.allowed_agents:
        return guidance.allowed_agents
    return list(SUBAGENTS.keys())


def list_workflow_descriptions() -> Dict[str, Dict[str, any]]:
    """
    List all workflow types with their descriptions and tags.
    
    Returns:
        Dictionary mapping workflow_type to dict containing 'description' and 'tags'
        
    Example:
        {
            "pfd": {
                "description": "Iterative PFD workflows with convergence checking...",
                "tags": ["iterative", "convergence", "active_learning", "ml_training"]
            },
            "single_task": {...},
            "multi_step": {...}
        }
    """
    return {
        workflow_type: {
            "description": guidance.description,
            "tags": guidance.tags
        }
        for workflow_type, guidance in WORKFLOW_GUIDANCE_LIBRARY.items()
    }


# Export for easy importing
__all__ = [
    "WorkflowGuidance",
    "WORKFLOW_GUIDANCE_LIBRARY",
    "search_workflow_guidance",
    "get_all_workflow_types",
    "get_workflow_description",
    "list_workflow_descriptions",
]
