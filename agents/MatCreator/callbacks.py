"""Database-backed callbacks for PFD agent session tracking and tool logging."""

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional, List
from pathlib import Path
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, select
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from google.adk.agents.callback_context import CallbackContext
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext

# Database path - stores in the agents directory
DB_DIR = Path(__file__).parent.parent
DB_PATH = f"sqlite:///{DB_DIR}/matcreator_memory.db"

Base = declarative_base()


class SessionMetadata(Base):
    """Database model for PFD session metadata."""
    
    __tablename__ = "pfd_session_metadata"
    
    session_id = Column(String(255), primary_key=True)
    app_name = Column(String(255), nullable=True)
    user_id = Column(String(255), nullable=True, index=True)
    workflow_type = Column(String(50), nullable=True)  # pfd_finetune or pfd_distillation
    workflow_log_path = Column(Text, nullable=True)
    long_term_goal = Column(Text, nullable=True)
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)
    additional_metadata = Column(Text, nullable=True)
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "session_id": self.session_id,
            "app_name": self.app_name,
            "user_id": self.user_id,
            "workflow_type": self.workflow_type,
            "workflow_log_path": self.workflow_log_path,
            "long_term_goal": self.long_term_goal,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
        if self.additional_metadata:
            try:
                result["additional_metadata"] = json.loads(self.additional_metadata)
            except:
                result["additional_metadata"] = {}
        return result


class ToolExecution(Base):
    """Database model for PFD tool execution logs."""
    
    __tablename__ = "pfd_tool_executions"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    session_id = Column(String(255), nullable=True, index=True)
    app_name = Column(String(255), nullable=True, index=True)
    user_id = Column(String(255), nullable=True, index=True)
    agent_name = Column(String(255), nullable=True, index=True)
    tool_name = Column(String(255), nullable=False, index=True)
    workflow_step = Column(String(100), nullable=True, index=True)  # e.g., exploration_md, labeling_abacus_scf
    inputs = Column(Text, nullable=True)
    output = Column(Text, nullable=True)
    status = Column(String(50), nullable=False)
    error = Column(Text, nullable=True)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "session_id": self.session_id,
            "app_name": self.app_name,
            "user_id": self.user_id,
            "agent_name": self.agent_name,
            "tool_name": self.tool_name,
            "workflow_step": self.workflow_step,
            "inputs": json.loads(self.inputs) if self.inputs else {},
            "output": json.loads(self.output) if self.output else None,
            "status": self.status,
            "error": self.error
        }


# Database engine and session
_engine = create_engine(DB_PATH, echo=False)
Base.metadata.create_all(_engine)
_Session = sessionmaker(bind=_engine)


def _serialize(obj: Any) -> str:
    """Convert object to JSON string."""
    if obj is None:
        return None
    try:
        return json.dumps(obj)
    except:
        return json.dumps(str(obj))


# Tool execution logging functions
def log_tool_execution(
    tool_name: str,
    inputs: Dict[str, Any],
    output: Any,
    status: str = "success",
    error: Optional[str] = None,
    session_id: Optional[str] = None,
    agent_name: Optional[str] = None,
    workflow_step: Optional[str] = None
) -> int:
    """Log a tool execution to database."""
    session_id = session_id or os.environ.get("CURRENT_SESSION_ID")
    app_name = os.environ.get("CURRENT_APP_NAME")
    user_id = os.environ.get("CURRENT_USER_ID")
    
    db = _Session()
    try:
        execution = ToolExecution(
            timestamp=datetime.now(timezone.utc),
            session_id=session_id,
            app_name=app_name,
            user_id=user_id,
            agent_name=agent_name,
            tool_name=tool_name,
            workflow_step=workflow_step,
            inputs=_serialize(inputs),
            output=_serialize(output),
            status=status,
            error=error
        )
        db.add(execution)
        db.commit()
        db.refresh(execution)
        return execution.id
    finally:
        db.close()


def get_session_executions(
    session_id: Optional[str] = None,
    tool_name: Optional[str] = None,
    workflow_step: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Get tool executions for a session, optionally filtered by tool name or workflow step."""
    session_id = session_id or os.environ.get("CURRENT_SESSION_ID")
    if not session_id:
        return []
    
    db = _Session()
    try:
        stmt = select(ToolExecution).where(ToolExecution.session_id == session_id)
        
        if tool_name:
            stmt = stmt.where(ToolExecution.tool_name == tool_name)
        if workflow_step:
            stmt = stmt.where(ToolExecution.workflow_step == workflow_step)
            
        stmt = stmt.order_by(ToolExecution.timestamp.asc())
        result = db.execute(stmt)
        return [e.to_dict() for e in result.scalars().all()]
    finally:
        db.close()


# Session metadata functions
def set_session_metadata(
    workflow_type: Optional[str] = None,
    workflow_log_path: Optional[str] = None,
    long_term_goal: Optional[str] = None,
    additional_metadata: Optional[Dict[str, Any]] = None,
    session_id: Optional[str] = None
) -> str:
    """
    Update session metadata, mandatory when planning new or making significant changes to tasks.
    
    Args:
        workflow_type: Type of workflow (pfd_finetune or pfd_distillation)
        workflow_log_path: Path to the workflow log file
        long_term_goal: Long-term goal or objective for this session
        additional_metadata: Additional custom metadata (JSON-serializable dict)
        session_id: Specific session ID to update (uses current session if not provided)
    
    Returns:
        str: Confirmation message or error
    """
    session_id = session_id or os.environ.get("CURRENT_SESSION_ID")
    if not session_id:
        return "Error: No active session"
    
    app_name = os.environ.get("CURRENT_APP_NAME")
    user_id = os.environ.get("CURRENT_USER_ID")
    
    db = _Session()
    try:
        metadata = db.query(SessionMetadata).filter_by(session_id=session_id).first()
        
        if metadata:
            if workflow_type is not None:
                metadata.workflow_type = workflow_type
            if workflow_log_path is not None:
                metadata.workflow_log_path = workflow_log_path
            if long_term_goal is not None:
                metadata.long_term_goal = long_term_goal
            if additional_metadata is not None:
                metadata.additional_metadata = json.dumps(additional_metadata)
            metadata.updated_at = datetime.now(timezone.utc)
        else:
            metadata = SessionMetadata(
                session_id=session_id,
                app_name=app_name,
                user_id=user_id,
                workflow_type=workflow_type,
                workflow_log_path=workflow_log_path,
                long_term_goal=long_term_goal,
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
                additional_metadata=json.dumps(additional_metadata) if additional_metadata else None
            )
            db.add(metadata)
        
        db.commit()
        return f"✓ PFD session metadata updated for {session_id}"
    finally:
        db.close()


def get_session_metadata(session_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Get the metadata for the current session. Mandatory for complex workflows."""
    session_id = session_id or os.environ.get("CURRENT_SESSION_ID")
    if not session_id:
        return None
    
    db = _Session()
    try:
        metadata = db.query(SessionMetadata).filter_by(session_id=session_id).first()
        return metadata.to_dict() if metadata else None
    finally:
        db.close()


def get_session_context(
    num_recent: int = 5,
    session_id: Optional[str] = None,
    include_workflow_steps: bool = True
) -> Dict[str, Any]:
    """
    Retrieve comprehensive context for the current or specific session.
    
    Args:
        num_recent: Number of most recent tool executions to include
        session_id: Specific session ID to query (uses current if not provided)
        include_workflow_steps: Whether to group actions by workflow step
    
    Returns:
        dict: Session context including metadata, tool executions, and workflow progress
    """
    session_id = session_id or os.environ.get("CURRENT_SESSION_ID")
    if not session_id:
        return {"error": "No active session"}
    
    metadata = get_session_metadata(session_id)
    executions = get_session_executions(session_id)
    recent = executions[-num_recent:] if len(executions) > num_recent else executions
    
    context = {
        "session_id": session_id,
        "workflow_type": metadata.get('workflow_type') if metadata else None,
        "workflow_log_path": metadata.get('workflow_log_path') if metadata else None,
        "long_term_goal": metadata.get('long_term_goal') if metadata else None,
        "total_actions": len(executions),
        "recent_actions": recent,
        "metadata": metadata
    }
    
    if include_workflow_steps and executions:
        # Group by workflow step
        steps = {}
        for execution in executions:
            step = execution.get('workflow_step')
            if step:
                if step not in steps:
                    steps[step] = []
                steps[step].append(execution)
        context["workflow_steps"] = steps
    
    return context


# Callbacks
def before_agent_callback(callback_context: CallbackContext):
    """Set environment variables for PFD session context."""
    session_id = callback_context._invocation_context.session.id
    user_id = callback_context._invocation_context.session.user_id
    app_name = callback_context._invocation_context.session.app_name
    
    os.environ["CURRENT_SESSION_ID"] = session_id
    os.environ["CURRENT_USER_ID"] = user_id
    os.environ["CURRENT_APP_NAME"] = app_name
    
    return None


# Tool name mapping for workflow steps
WORKFLOW_STEP_MAP = {
    "abacus_prepare": "labeling_abacus_scf_preparation",
    "abacus_prepare_batch": "labeling_abacus_scf_preparation",
    "abacus_calculation_scf": "labeling_abacus_scf_calculation",
    "abacus_calculation_scf_batch": "labeling_abacus_scf_calculation",
    "collect_abacus_scf_results": "labeling_abacus_scf_collect_results",
    "collect_abacus_scf_results_batch": "labeling_abacus_scf_collect_results",
    "run_molecular_dynamics": "exploration_md",
    "filter_by_entropy": "data_curation",
    "perturb_atoms": "structure_perturbation",
    "build_bulk_crystal": "structure_building",
    "build_supercell": "structure_building",
    "training": "model_training",
    "ase_calculation": "labeling_ase_calculation"
}


def after_tool_callback(
    tool: BaseTool,
    args: Dict[str, Any],
    tool_context: ToolContext,
    tool_response: Dict
):
    """
    Log PFD tool execution to database.
    
    All tool executions are automatically logged to the database with:
    - Tool name, inputs, outputs, status
    - Session ID and agent name
    - Workflow step classification
    - Timestamp for tracking execution order
    """
    try:
        tool_name = getattr(tool, 'name', getattr(tool, '__name__', str(tool)))
        session_id = getattr(tool_context.session, 'id', None)
        agent_name = getattr(tool_context, 'agent_name', None)
        
        # Determine workflow step
        workflow_step = WORKFLOW_STEP_MAP.get(tool_name)
        
        # Extract output and status
        output = None
        status = "success"
        error = None
        
        if isinstance(tool_response, dict):
            if "error" in tool_response:
                status = "error"
                error = tool_response.get("error")
                output = tool_response
            elif "structuredContent" in tool_response:
                output = tool_response["structuredContent"].get("result")
            else:
                output = tool_response
        else:
            output = tool_response
        
        # Log to database
        log_tool_execution(
            tool_name=tool_name,
            inputs=args,
            output=output,
            status=status,
            error=error,
            session_id=session_id,
            agent_name=agent_name,
            workflow_step=workflow_step
        )
    
    except Exception as e:
        print(f"Warning: Failed to log tool execution: {e}")
    
    return None
