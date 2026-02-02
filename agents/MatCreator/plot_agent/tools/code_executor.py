"""Code executor for matplotlib plotting scripts."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, Optional

from pydantic import BaseModel, Field


class ExecutionResult(BaseModel):
    """Result of code execution."""
    
    success: bool = Field(..., description="Whether execution completed without errors")
    output_path: Optional[str] = Field(None, description="Path to generated plot file if successful")
    stdout: str = Field(default="", description="Standard output from execution")
    stderr: str = Field(default="", description="Standard error from execution")
    exit_code: int = Field(default=0, description="Process exit code")
    execution_time: float = Field(default=0.0, description="Execution time in seconds")


def execute_plot_code(
    code: str,
    output_dir: Optional[str] = None,
    timeout: int = 30,
    python_executable: Optional[str] = None,
) -> ExecutionResult:
    """
    Execute matplotlib plotting code in an isolated subprocess.
    
    Safety features:
    - Runs in subprocess (isolated from main process)
    - Timeout protection
    - Captures stdout/stderr
    - Validates output file creation
    
    Args:
        code: Python code to execute
        output_dir: Directory to save plots (default: temp directory)
        timeout: Maximum execution time in seconds
        python_executable: Path to Python interpreter (default: current)
        
    Returns:
        ExecutionResult with execution status and output paths
    """
    import time
    
    if python_executable is None:
        python_executable = sys.executable
    
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="plot_output_")
    else:
        output_dir = os.path.abspath(output_dir)
        os.makedirs(output_dir, exist_ok=True)
    
    # Create a temporary Python script
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        script_path = f.name
        
        # Inject output directory into the code
        modified_code = f"""
import os
import sys

# Set output directory
OUTPUT_DIR = r"{output_dir}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Original code
{code}
"""
        f.write(modified_code)
    
    try:
        # Track existing files before execution to detect new ones
        existing_files = set()
        for ext in ['.png', '.pdf', '.jpg', '.svg']:
            existing_files.update(Path(output_dir).glob(f'*{ext}'))
        
        start_time = time.time()
        
        # Execute the script in subprocess
        result = subprocess.run(
            [python_executable, script_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=output_dir,
        )
        
        execution_time = time.time() - start_time
        
        # Find newly created files by comparing before/after
        current_files = set()
        for ext in ['.png', '.pdf', '.jpg', '.svg']:
            current_files.update(Path(output_dir).glob(f'*{ext}'))
        
        new_files = sorted(current_files - existing_files, key=os.path.getmtime, reverse=True)
        output_path = str(new_files[0]) if new_files else None
        
        success = result.returncode == 0 and output_path is not None
        
        return ExecutionResult(
            success=success,
            output_path=output_path,
            stdout=result.stdout,
            stderr=result.stderr,
            exit_code=result.returncode,
            execution_time=execution_time,
        )
        
    except subprocess.TimeoutExpired:
        return ExecutionResult(
            success=False,
            stderr=f"Execution timed out after {timeout} seconds",
            exit_code=-1,
        )
        
    except Exception as e:
        return ExecutionResult(
            success=False,
            stderr=f"Execution error: {str(e)}",
            exit_code=-1,
        )
        
    finally:
        # Clean up temporary script
        try:
            os.unlink(script_path)
        except:
            pass
