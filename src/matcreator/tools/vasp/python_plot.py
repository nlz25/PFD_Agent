import os
import sys
import io
import traceback
from pathlib import Path
from typing import Dict, Any, List, Optional
import pickle
import yaml
import uuid
import base64
from contextlib import redirect_stdout, redirect_stderr
from mcp.server.fastmcp import FastMCP
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pymatgen.core import Structure, Element, Lattice
from pymatgen.io.vasp import Vasprun
from pymatgen.electronic_structure.bandstructure import BandStructure
from pymatgen.electronic_structure.dos import CompleteDos


def safe_execute_plot_code(plot_code: str, data: Dict[str, Any], work_dir: str) -> tuple[bool, str, Optional[str]]:
    """安全执行画图代码"""
    try:
        # 重定向输出
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        
        # 创建执行环境，包含data变量和所需的库
        exec_globals = {
            'data': data,
            'plt': plt,
            'np': np,
            'pd': pd,
            'Structure': Structure,
            'Element': Element,
            'Lattice': Lattice,
            'BandStructure': BandStructure,
            'CompleteDos': CompleteDos,
            'Vasprun': Vasprun,
            '__builtins__': __builtins__
        }
        
        # 导入常用的pymatgen模块
        try:
            from pymatgen.electronic_structure.core import Spin
            from pymatgen.electronic_structure.plotter import BSPlotter, DosPlotter
            exec_globals['Spin'] = Spin
            exec_globals['BSPlotter'] = BSPlotter
            exec_globals['DosPlotter'] = DosPlotter
        except ImportError:
            pass  # 如果某些模块不可用，继续执行
        
        with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
            exec(plot_code, exec_globals)
        
        # 生成唯一的图片文件名
        plot_id = str(uuid.uuid4())
        plot_filename = f"plot_{plot_id}.png"
        plot_path = Path(work_dir) / plot_filename
        
        # 保存图片
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()  # 关闭图片以释放内存
        
        # 读取图片并转换为base64
        with open(plot_path, 'rb') as f:
            img_data = f.read()
            img_base64 = base64.b64encode(img_data).decode('utf-8')
        
        return True, str(plot_path), img_base64
        
    except Exception as e:
        plt.close()  # 确保在出错时也关闭图片
        error_msg = f"执行画图代码时出错: {str(e)}\n{traceback.format_exc()}"
        return False, error_msg, None