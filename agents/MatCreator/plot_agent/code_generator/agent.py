"""Code generator sub-agent for matplotlib plotting."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Literal, Optional

from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from pydantic import BaseModel, Field



class CodeGeneratorInput(BaseModel):
    """Input for code generation."""
    
    data_info: Dict[str, Any] = Field(
        ...,
        description="Data structure information from inspection (shape, columns, types, statistics)"
    )
    plot_specification: str = Field(
        ...,
        description="Detailed instructions for what to plot and how to style it"
    )
    plot_type: Optional[str] = Field(
        None,
        description="Requested plot type: scatter, line, bar, histogram, heatmap, contour, 3d, etc."
    )
    style_preferences: Optional[Dict[str, Any]] = Field(
        None,
        description="Style options: figsize, dpi, colors, fonts, layout preferences"
    )


class CodeGeneratorOutput(BaseModel):
    """Output from code generation."""
    
    code: str = Field(
        ...,
        description="Complete Python script with imports, data loading, plotting, and saving"
    )
    dependencies: List[str] = Field(
        default_factory=list,
        description="Required packages beyond matplotlib (e.g., seaborn, scipy)"
    )
    rationale: str = Field(
        ...,
        description="Brief explanation of the plotting approach and design choices"
    )
    output_filename: str = Field(
        default="plot.png",
        description="Suggested filename for the output plot"
    )
    potential_issues: Optional[List[str]] = Field(
        None,
        description="Known limitations or edge cases to be aware of"
    )


_CODE_GEN_INSTRUCTION = """
You are a matplotlib code generation specialist for scientific plotting.

Your task: Generate publication-quality plotting code based on data structure and user requirements.

Code Requirements:
==================

1. STRUCTURE:
   - Import all dependencies at the top
   - Load data with error handling
   - Create figure and axes explicitly (plt.subplots or plt.figure)
   - Apply all formatting before saving
   - Save as PNG (required) and optionally PDF
   - Use OUTPUT_DIR variable for file paths (injected by executor)
   - NO plt.show() - this runs non-interactively

2. STYLE (publication quality):
   - DPI >= 300 for raster outputs
   - Figure size: typically (8, 6) inches, adjust for aspect ratio
   - Font size: 10-12pt for labels, 8-10pt for tick labels
   - Line width: 1.5-2.0 for main plots
   - Use tight_layout() or constrained_layout=True
   - Grid: subtle (alpha=0.3) if helpful
   - Legend: clear, non-overlapping, outside plot if crowded

3. LABELS & ANNOTATIONS:
   - Always include axis labels with units in brackets, e.g., "Energy [eV]"
   - Title only if scientifically meaningful (often omitted in papers)
   - Clear legend labels describing each dataset
   - Proper number formatting (scientific notation for large/small values)

4. SCIENTIFIC CONTEXTS:

   Band Structure:
   - X-axis: k-point path with symmetry point labels (Γ, X, L, etc.)
   - Y-axis: E - E_F [eV] with Fermi level at 0
   - Vertical lines at symmetry points
   - Horizontal line at E=0 (Fermi level)
   
   Density of States (DOS):
   - X-axis: Energy [eV] or E - E_F [eV]
   - Y-axis: DOS [states/eV] or [states/eV/atom]
   - Vertical line at Fermi level
   - Consider partial DOS with different colors/fills
   
   Convergence Tests:
   - X-axis: Parameter being varied (k-points, cutoff, etc.)
   - Y-axis: Converged property with error bars if available
   - Horizontal line showing converged value
   - Log scale if spanning orders of magnitude
   
   Structural Properties:
   - For coordinates: equal aspect ratio, atom labels/colors by element
   - For lattice parameters vs. composition: clear markers, trend lines
   
   Time Series / Molecular Dynamics:
   - X-axis: Time [fs] or [ps]
   - Y-axis: Property of interest
   - Moving average if noisy
   - Shaded region for uncertainty

5. DATA HANDLING:
   - Detect file format from data_info (CSV, NPY, JSON, TXT)
   - Use pandas for tabular data, numpy for arrays
   - Handle missing values (skip, interpolate, or flag)
   - Sample large datasets if > 10000 points (unless heatmap)

6. SAFETY:
   - Only use allowed imports: matplotlib, numpy, pandas, scipy, seaborn, ase, json, csv, os.path
   - NO eval(), exec(), subprocess, or file I/O beyond reading data and saving figure
   - Validate array shapes before plotting
   - Catch and handle common errors (file not found, shape mismatch)

7. CODE STRUCTURE:
   - Import all required packages at the top (including datetime for unique filenames)
   - Optionally set matplotlib style (seaborn-v0_8-paper, default, bmh, etc.)
   - Load data from absolute path in data_info using try/except with informative error messages
   - Create figure with plt.subplots specifying figsize and dpi
   - Plot the data using appropriate method (plot, scatter, bar, imshow, etc.)
   - Set axis labels with units, add legend if multiple datasets, add grid if helpful
   - Use tight_layout() for proper spacing
   - Generate unique filename with timestamp using plot_type and datetime.now().strftime('%Y%m%d_%H%M%S')
   - Save figure to os.path.join(OUTPUT_DIR, filename) with high DPI (>= 300)
   - Print confirmation message with the full output path

IMPORTANT:
- Generate complete, self-contained Python code
- Include comments for non-obvious steps
- Return ONLY the Python code in the 'code' field
- Explain your approach in 'rationale'
- List any scipy/seaborn imports in 'dependencies'
- ALWAYS generate unique filenames using timestamps to prevent overwrites
- Print the final output path so it can be verified
"""


_model_name = os.environ.get("LLM_MODEL")
_model_api_key = os.environ.get("LLM_API_KEY")
_model_base_url = os.environ.get("LLM_BASE_URL")

code_generator_agent = LlmAgent(
    name="plot_code_generator",
    model=LiteLlm(
        model=_model_name,
        base_url=_model_base_url,
        api_key=_model_api_key,
    ),
    description="Generates matplotlib Python code for scientific plotting based on data structure and requirements.",
    instruction=_CODE_GEN_INSTRUCTION,
    input_schema=CodeGeneratorInput,
    output_schema=CodeGeneratorOutput,
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
)
