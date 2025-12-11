import sqlite3
import pickle
import json
from typing import Dict, Any, Optional, List
from pathlib import Path
import logging

class VaspCalculationDB:
    """VASP计算记录的SQLite数据库管理类"""
    
    def __init__(self, db_path: str):
        """
        初始化数据库连接
        
        Args:
            db_path: SQLite数据库文件路径
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
        
    def _init_database(self):
        """初始化数据库表结构"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS calculations (
                    calculation_id TEXT PRIMARY KEY,
                    slurm_id TEXT,
                    success BOOLEAN,
                    error TEXT,
                    status TEXT,
                    calculate_path TEXT,
                    calc_type TEXT,
                    
                    -- relaxation 相关字段
                    total_energy REAL,
                    max_force REAL,
                    ionic_steps INTEGER,
                    
                    -- scf/nscf 相关字段
                    efermi REAL,
                    is_metal BOOLEAN,
                    
                    -- 控制参数
                    soc BOOLEAN,
                    restart_id TEXT,
                    kpath TEXT,
                    n_kpoints INTEGER,
                    
                    -- 时间戳
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    
                    -- 复杂对象的BLOB存储
                    structure_blob BLOB,
                    band_structure_blob BLOB,
                    dos_blob BLOB,
                    eigenvalues_blob BLOB,
                    band_gap_blob BLOB,
                    stress_blob BLOB,
                    incar_tags_blob BLOB,
                    cbm_blob BLOB,
                    vbm_blob BLOB
                )
            """)
            
            # 创建索引以提高查询性能
            conn.execute("CREATE INDEX IF NOT EXISTS idx_calc_type ON calculations(calc_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_status ON calculations(status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_restart_id ON calculations(restart_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON calculations(created_at)")
            
    def write_record(self, calculation_id: str, data: dict):
        """
        写入计算记录
        
        Args:
            calculation_id: 计算ID
            data: 计算数据字典
        """
        # 提取简单字段
        simple_fields = {
            'calculation_id': calculation_id,
            'slurm_id': data.get('slurm_id'),
            'success': data.get('success'),
            'error': data.get('error'),
            'status': data.get('status'),
            'calculate_path': data.get('calculate_path'),
            'calc_type': data.get('calc_type'),
            'total_energy': data.get('total_energy'),
            'max_force': data.get('max_force'),
            'ionic_steps': data.get('ionic_steps'),
            'efermi': data.get('efermi'),
            'is_metal': data.get('is_metal'),
            'soc': data.get('soc'),
            'restart_id': data.get('restart_id'),
            'kpath': data.get('kpath'),
            'n_kpoints': data.get('n_kpoints'),
        }
        
        # 序列化复杂对象
        blob_fields = {}
        complex_field_mapping = {
            'structure': 'structure_blob',
            'band_structure': 'band_structure_blob',
            'dos': 'dos_blob',
            'eigenvalues': 'eigenvalues_blob',
            'eigen_values': 'eigenvalues_blob',  # 兼容不同命名
            'band_gap': 'band_gap_blob',
            'stress': 'stress_blob',
            'incar_tags': 'incar_tags_blob',
            'cbm': 'cbm_blob',
            'vbm': 'vbm_blob'
        }
        
        for data_key, blob_key in complex_field_mapping.items():
            if data_key in data and data[data_key] is not None:
                try:
                    blob_fields[blob_key] = pickle.dumps(data[data_key])
                except Exception as e:
                    logging.warning(f"Failed to serialize {data_key}: {e}")
                    blob_fields[blob_key] = None
            else:
                blob_fields[blob_key] = None
        
        # 合并所有字段
        all_fields = {**simple_fields, **blob_fields}
        
        with sqlite3.connect(self.db_path) as conn:
            # 使用INSERT OR REPLACE以支持更新
            placeholders = ', '.join(['?' for _ in all_fields])
            columns = ', '.join(all_fields.keys())
            
            conn.execute(f"""
                INSERT OR REPLACE INTO calculations ({columns})
                VALUES ({placeholders})
            """, list(all_fields.values()))
            
            # 更新时间戳
            conn.execute(
                "UPDATE calculations SET updated_at = CURRENT_TIMESTAMP WHERE calculation_id = ?",
                (calculation_id,)
            )
            
    def read_record(self, calculation_id: str) -> Optional[Dict[str, Any]]:
        """
        读取计算记录
        
        Args:
            calculation_id: 计算ID
            
        Returns:
            计算数据字典，如果不存在则返回None
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row  # 使结果可以按列名访问
            cursor = conn.execute(
                "SELECT * FROM calculations WHERE calculation_id = ?", 
                (calculation_id,)
            )
            row = cursor.fetchone()
            
            if row is None:
                return None
                
            # 转换为字典并反序列化复杂对象
            data = dict(row)
            
            # 反序列化BLOB字段
            blob_field_mapping = {
                'structure_blob': 'structure',
                'band_structure_blob': 'band_structure',
                'dos_blob': 'dos',
                'eigenvalues_blob': 'eigenvalues',
                'band_gap_blob': 'band_gap',
                'stress_blob': 'stress',
                'incar_tags_blob': 'incar_tags',
                'cbm_blob': 'cbm',
                'vbm_blob': 'vbm'
            }
            
            for blob_key, data_key in blob_field_mapping.items():
                if data[blob_key] is not None:
                    try:
                        data[data_key] = pickle.loads(data[blob_key])
                    except Exception as e:
                        logging.warning(f"Failed to deserialize {data_key}: {e}")
                        data[data_key] = None
                else:
                    data[data_key] = None
                # 删除blob字段，保持接口兼容
                del data[blob_key]
                
            # 删除时间戳字段，保持接口兼容
            data.pop('created_at', None)
            data.pop('updated_at', None)
            
            return data
    
    def list_calculations(self, 
                         calc_type: Optional[str] = None,
                         status: Optional[str] = None,
                         limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        列出计算记录
        
        Args:
            calc_type: 计算类型过滤
            status: 状态过滤
            limit: 返回记录数限制
            
        Returns:
            计算记录列表
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            query = "SELECT calculation_id, calc_type, status, total_energy, efermi, created_at FROM calculations"
            params = []
            conditions = []
            
            if calc_type:
                conditions.append("calc_type = ?")
                params.append(calc_type)
            
            if status:
                conditions.append("status = ?")
                params.append(status)
                
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
                
            query += " ORDER BY created_at DESC"
            
            if limit:
                query += " LIMIT ?"
                params.append(limit)
                
            cursor = conn.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
    
    def delete_record(self, calculation_id: str) -> bool:
        """
        删除计算记录
        
        Args:
            calculation_id: 计算ID
            
        Returns:
            是否成功删除
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "DELETE FROM calculations WHERE calculation_id = ?",
                (calculation_id,)
            )
            return cursor.rowcount > 0
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取数据库统计信息
        
        Returns:
            统计信息字典
        """
        with sqlite3.connect(self.db_path) as conn:
            stats = {}
            
            # 总记录数
            cursor = conn.execute("SELECT COUNT(*) FROM calculations")
            stats['total_calculations'] = cursor.fetchone()[0]
            
            # 按计算类型统计
            cursor = conn.execute("SELECT calc_type, COUNT(*) FROM calculations GROUP BY calc_type")
            stats['by_calc_type'] = dict(cursor.fetchall())
            
            # 按状态统计
            cursor = conn.execute("SELECT status, COUNT(*) FROM calculations GROUP BY status")
            stats['by_status'] = dict(cursor.fetchall())
            
            return stats