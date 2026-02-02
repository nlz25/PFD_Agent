"""Data inspection tool for scientific data files."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field


class DataInspectionResult(BaseModel):
    """Result of data inspection."""
    
    file_path: str = Field(..., description="Absolute path to the inspected file")
    file_format: str = Field(..., description="Detected file format (csv, json, npy, txt, etc.)")
    shape: tuple = Field(..., description="Shape of the data array/dataframe")
    columns: Optional[List[str]] = Field(None, description="Column names if applicable")
    dtypes: Optional[Dict[str, str]] = Field(None, description="Data types for each column")
    sample_data: Optional[str] = Field(None, description="First few rows/entries as string")
    statistics: Optional[Dict[str, Any]] = Field(None, description="Basic statistics (min, max, mean)")
    missing_values: Optional[Dict[str, int]] = Field(None, description="Count of NaN/None per column")
    recommendations: List[str] = Field(default_factory=list, description="Suggestions for plotting")


def inspect_data(file_path: str) -> DataInspectionResult:
    """
    Inspect a data file and return detailed information about its structure.
    
    Supported formats:
    - CSV/TSV: Tabular data
    - JSON: Structured data
    - NPY/NPZ: NumPy arrays
    - TXT/DAT: Space/tab-delimited numeric data
    
    Args:
        file_path: Path to the data file
        
    Returns:
        DataInspectionResult with file structure, statistics, and recommendations
    """
    file_path = os.path.abspath(file_path)
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    ext = Path(file_path).suffix.lower()
    
    # CSV/TSV files
    if ext in ['.csv', '.tsv']:
        return _inspect_csv(file_path, delimiter=',' if ext == '.csv' else '\t')
    
    # JSON files
    elif ext == '.json':
        return _inspect_json(file_path)
    
    # NumPy files
    elif ext in ['.npy', '.npz']:
        return _inspect_numpy(file_path)
    
    # Generic text files
    elif ext in ['.txt', '.dat']:
        return _inspect_text(file_path)
    
    else:
        raise ValueError(f"Unsupported file format: {ext}")


def _inspect_csv(file_path: str, delimiter: str = ',') -> DataInspectionResult:
    """Inspect CSV/TSV files."""
    try:
        df = pd.read_csv(file_path, delimiter=delimiter)
    except Exception as e:
        # Try without header
        df = pd.read_csv(file_path, delimiter=delimiter, header=None)
    
    columns = df.columns.tolist()
    dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}
    missing = {col: int(df[col].isna().sum()) for col in columns}
    
    # Basic statistics for numeric columns
    stats = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        stats[col] = {
            'min': float(df[col].min()),
            'max': float(df[col].max()),
            'mean': float(df[col].mean()),
            'std': float(df[col].std()),
        }
    
    # Recommendations
    recommendations = []
    numeric_cols = len(df.select_dtypes(include=[np.number]).columns)
    if numeric_cols >= 2:
        recommendations.append("Multiple numeric columns detected - suitable for scatter/line plots")
    if len(df) > 1000:
        recommendations.append("Large dataset - consider sampling or aggregation for performance")
    
    return DataInspectionResult(
        file_path=file_path,
        file_format='csv' if delimiter == ',' else 'tsv',
        shape=df.shape,
        columns=columns,
        dtypes=dtypes,
        sample_data=df.head(5).to_string(),
        statistics=stats,
        missing_values=missing,
        recommendations=recommendations,
    )


def _inspect_json(file_path: str) -> DataInspectionResult:
    """Inspect JSON files."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    recommendations = []
    
    if isinstance(data, list):
        shape = (len(data),)
        sample = json.dumps(data[:5], indent=2) if len(data) > 5 else json.dumps(data, indent=2)
        
        if all(isinstance(item, dict) for item in data):
            # List of dictionaries - can be converted to DataFrame
            keys = set()
            for item in data:
                keys.update(item.keys())
            recommendations.append("List of objects detected - can convert to tabular format")
            columns = list(keys)
        else:
            columns = None
            
    elif isinstance(data, dict):
        shape = (len(data),)
        sample = json.dumps(data, indent=2)[:500] + "..."
        columns = list(data.keys())
        recommendations.append("Dictionary detected - keys could be labels or categories")
    else:
        shape = (1,)
        sample = str(data)
        columns = None
    
    return DataInspectionResult(
        file_path=file_path,
        file_format='json',
        shape=shape,
        columns=columns,
        sample_data=sample,
        recommendations=recommendations,
    )


def _inspect_numpy(file_path: str) -> DataInspectionResult:
    """Inspect NumPy files."""
    ext = Path(file_path).suffix.lower()
    
    if ext == '.npy':
        arr = np.load(file_path)
        arrays = {'array': arr}
    else:  # .npz
        npz = np.load(file_path)
        arrays = {key: npz[key] for key in npz.files}
    
    recommendations = []
    
    # Analyze first/only array
    main_key = list(arrays.keys())[0]
    arr = arrays[main_key]
    
    shape = arr.shape
    dtype = str(arr.dtype)
    
    # Statistics for numeric arrays
    stats = {}
    if np.issubdtype(arr.dtype, np.number):
        stats[main_key] = {
            'min': float(np.min(arr)),
            'max': float(np.max(arr)),
            'mean': float(np.mean(arr)),
            'std': float(np.std(arr)),
        }
    
    # Recommendations based on shape
    if len(shape) == 1:
        recommendations.append("1D array - suitable for histograms or line plots")
    elif len(shape) == 2:
        if shape[0] > 10 and shape[1] > 10:
            recommendations.append("2D array - suitable for heatmaps or images")
        else:
            recommendations.append("2D array - suitable for scatter plots (columns as variables)")
    
    sample = f"Shape: {shape}, Dtype: {dtype}\nFirst values: {arr.flat[:10]}"
    
    return DataInspectionResult(
        file_path=file_path,
        file_format='npy' if ext == '.npy' else 'npz',
        shape=shape,
        dtypes={main_key: dtype},
        sample_data=sample,
        statistics=stats,
        recommendations=recommendations,
    )


def _inspect_text(file_path: str) -> DataInspectionResult:
    """Inspect generic text files with numeric data."""
    # Try to load as numeric array
    try:
        data = np.loadtxt(file_path)
        
        shape = data.shape if data.ndim > 0 else (1,)
        
        stats = {
            'data': {
                'min': float(np.min(data)),
                'max': float(np.max(data)),
                'mean': float(np.mean(data)),
                'std': float(np.std(data)),
            }
        }
        
        with open(file_path, 'r') as f:
            sample = ''.join(f.readlines()[:10])
        
        recommendations = []
        if data.ndim == 2 and data.shape[1] >= 2:
            recommendations.append(f"{data.shape[1]} columns detected - first column often represents X-axis")
        
        return DataInspectionResult(
            file_path=file_path,
            file_format='txt',
            shape=shape,
            sample_data=sample,
            statistics=stats,
            recommendations=recommendations,
        )
        
    except Exception as e:
        # Fall back to line-by-line inspection
        with open(file_path, 'r') as f:
            lines = f.readlines()[:20]
        
        return DataInspectionResult(
            file_path=file_path,
            file_format='txt',
            shape=(len(lines),),
            sample_data=''.join(lines),
            recommendations=["Could not parse as numeric data - manual inspection required"],
        )
