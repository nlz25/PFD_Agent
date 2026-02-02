"""Code validator for matplotlib plotting scripts."""

from __future__ import annotations

import ast
from typing import Dict, List

from pydantic import BaseModel, Field


class CodeValidationResult(BaseModel):
    """Result of code validation."""
    
    is_valid: bool = Field(..., description="Whether the code passes all checks")
    errors: List[str] = Field(default_factory=list, description="Critical errors that prevent execution")
    warnings: List[str] = Field(default_factory=list, description="Non-critical issues or suggestions")
    allowed_imports: List[str] = Field(default_factory=list, description="Detected safe imports")
    forbidden_patterns: List[str] = Field(default_factory=list, description="Dangerous patterns detected")


# Whitelist of allowed imports for plotting
ALLOWED_MODULES = {
    'matplotlib', 'matplotlib.pyplot', 'mpl_toolkits',
    'numpy', 'np',
    'pandas', 'pd',
    'seaborn', 'sns',
    'scipy', 'scipy.stats', 'scipy.interpolate',
    'ase', 'ase.io',
    'json', 'csv', 'pathlib', 'os.path',
}

# Dangerous patterns to detect
FORBIDDEN_PATTERNS = [
    'eval', 'exec', 'compile',
    '__import__', 'importlib',
    'subprocess', 'os.system', 'os.popen',
    'open(', 'file(',  # File operations (we control these)
    'pickle', 'shelve',
]


def validate_code(code: str) -> CodeValidationResult:
    """
    Validate matplotlib plotting code for safety and correctness.
    
    Checks:
    1. Syntax validity (can be parsed as Python)
    2. Import whitelist (only safe modules)
    3. No dangerous operations (eval, exec, subprocess, etc.)
    4. Required matplotlib elements present
    
    Args:
        code: Python code string to validate
        
    Returns:
        CodeValidationResult with validation status and details
    """
    errors = []
    warnings = []
    allowed_imports = []
    forbidden = []
    
    # Check 1: Syntax validation
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        errors.append(f"Syntax error: {e}")
        return CodeValidationResult(
            is_valid=False,
            errors=errors,
            warnings=warnings,
        )
    
    # Check 2: Import validation
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                module = alias.name
                if not _is_module_allowed(module):
                    errors.append(f"Forbidden import: {module}")
                else:
                    allowed_imports.append(module)
                    
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if not _is_module_allowed(module):
                errors.append(f"Forbidden import from: {module}")
            else:
                allowed_imports.append(module)
    
    # Check 3: Forbidden patterns
    code_lower = code.lower()
    for pattern in FORBIDDEN_PATTERNS:
        if pattern in code:
            forbidden.append(pattern)
            if pattern in ['eval', 'exec', 'subprocess', 'os.system']:
                errors.append(f"Dangerous operation detected: {pattern}")
            else:
                warnings.append(f"Potentially unsafe pattern: {pattern}")
    
    # Check 4: Required matplotlib elements
    has_figure_creation = any([
        'plt.figure(' in code,
        'plt.subplots(' in code,
        'Figure(' in code,
    ])
    
    has_save = 'savefig' in code or 'plt.savefig' in code
    
    if not has_figure_creation:
        warnings.append("No explicit figure creation detected (plt.figure or plt.subplots)")
    
    if not has_save:
        warnings.append("No savefig call detected - plot may not be saved")
    
    # Check 5: Best practices
    if 'plt.show()' in code:
        warnings.append("plt.show() detected - remove for non-interactive execution")
    
    if 'tight_layout' not in code and 'constrained_layout' not in code:
        warnings.append("Consider using tight_layout() or constrained_layout for better spacing")
    
    # Determine overall validity
    is_valid = len(errors) == 0
    
    return CodeValidationResult(
        is_valid=is_valid,
        errors=errors,
        warnings=warnings,
        allowed_imports=list(set(allowed_imports)),
        forbidden_patterns=forbidden,
    )


def _is_module_allowed(module: str) -> bool:
    """Check if a module is in the whitelist."""
    # Check exact match
    if module in ALLOWED_MODULES:
        return True
    
    # Check prefix match (e.g., matplotlib.pyplot.* )
    for allowed in ALLOWED_MODULES:
        if module.startswith(allowed + '.'):
            return True
    
    return False
