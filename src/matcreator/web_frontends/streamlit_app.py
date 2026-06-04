"""
Speaker Agent Chat Application
==============================

This Streamlit application provides a chat interface for interacting with the ADK Speaker Agent.
It allows users to create sessions, send messages, and receive both text and audio responses.

Requirements:
------------
- ADK API Server running on localhost:8000
- Speaker Agent registered and available in the ADK
- Streamlit and related packages installed

Usage:
------
1. Start the ADK API Server: `adk api_server`
2. Ensure the Speaker Agent is registered and working
3. Run this Streamlit app: `streamlit run apps/speaker_app.py`
4. Click "Create Session" in the sidebar
5. Start chatting with the Speaker Agent

Architecture:
------------
- Session Management: Creates and manages ADK sessions for stateful conversations
- Message Handling: Sends user messages to the ADK API and processes responses
- Audio Integration: Extracts audio file paths from responses and displays players

API Assumptions:
--------------
1. ADK API Server runs on localhost:8000
2. Speaker Agent is registered with app_name="speaker"
3. The Speaker Agent uses ElevenLabs TTS and saves audio files locally
4. Audio files are accessible from the path returned in the API response
5. Responses follow the ADK event structure with model outputs and function calls/responses

"""
import streamlit as st
import requests
import json
import os
import uuid
import time
from pathlib import Path
import streamlit.components.v1 as components
from streamlit_float import *
from ase.io import read
import py3Dmol

# Resolve workspace root (mirrors workspace.py logic).
# Paths returned by the agent are relative to this directory.
_env_workspace = os.environ.get("MATCLAW_WORKSPACE", "")
WORKSPACE_ROOT: Path = (
    Path(_env_workspace).expanduser().resolve()
    if _env_workspace
    else (Path(__file__).parent.parent / "agents" / "MatCreator" / ".workspace").resolve()
)


def resolve_path(path: str) -> str:
    """Return an absolute path, resolving relative paths against WORKSPACE_ROOT."""
    if not path:
        return path
    p = Path(path)
    if p.is_absolute():
        return str(p)
    return str(WORKSPACE_ROOT / p)


def is_structure_file(path: str | Path) -> bool:
    """Return True for structure files that can be previewed in the UI."""
    p = Path(path)
    suffix = p.suffix.lower()
    return suffix in {".cif", ".xyz", ".extxyz", ".vasp"} or p.name.upper() in {
        "POSCAR",
        "CONTCAR",
    }


def _session_workdir(session_id: str | None = None) -> Path | None:
    """Return the active session work directory."""
    sid = session_id or st.session_state.get("session_id")
    if not sid:
        return None
    _env_session_dir = os.environ.get("MATCLAW_SESSION_DIR", "")
    if _env_session_dir:
        return Path(_env_session_dir).expanduser().resolve()
    return WORKSPACE_ROOT / "sessions" / sid


def _extract_response_paths(payload) -> dict[str, str]:
    """Recursively extract known artifact paths from a tool response payload."""
    found = {}
    path_keys = {"structure_path", "plot_path", "artifact_path"}

    def walk(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key in path_keys and item:
                    found[key] = str(item)
                else:
                    walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(payload)
    return found

# Set page config
st.set_page_config(
    page_title="MatCreator",
    page_icon="🔊",
    layout="wide"
)

# Constants
API_BASE_URL = "http://localhost:8000"
APP_NAME = "MatCreator"

# Initialize session state variables
if "user_id" not in st.session_state:
    st.session_state.user_id = None

if "login_username" not in st.session_state:
    st.session_state.login_username = ""
    
if "session_id" not in st.session_state:
    st.session_state.session_id = None
    
if "messages" not in st.session_state:
    st.session_state.messages = []

if "audio_files" not in st.session_state:
    st.session_state.audio_files = []

# Keep simple artifact/state tracking
if "artifacts" not in st.session_state:
    st.session_state.artifacts = []

# Track structure paths for visualization
if "structure_paths" not in st.session_state:
    st.session_state.structure_paths = []

# Track uploader key to clear it after each message
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

# Track available sessions
if "available_sessions" not in st.session_state:
    st.session_state.available_sessions = []

# Track evaluation sets
if "eval_sets" not in st.session_state:
    st.session_state.eval_sets = []

# Track eval results
if "eval_results" not in st.session_state:
    st.session_state.eval_results = []

# Track selected eval set
if "selected_eval_set" not in st.session_state:
    st.session_state.selected_eval_set = None

# Track selected eval cases for deletion
if "selected_eval_cases" not in st.session_state:
    st.session_state.selected_eval_cases = []

# Track current view mode
if "view_mode" not in st.session_state:
    st.session_state.view_mode = "session"  # "session" or "evaluation"

# Track configured metrics for evaluation
if "configured_metrics" not in st.session_state:
    st.session_state.configured_metrics = []

# Track the structure currently shown in the right panel
if "selected_structure_path" not in st.session_state:
    st.session_state.selected_structure_path = None


def _reset_user_session_state() -> None:
    """Clear user-scoped UI state when switching identities."""
    st.session_state.session_id = None
    st.session_state.messages = []
    st.session_state.audio_files = []
    st.session_state.artifacts = []
    st.session_state.structure_paths = []
    st.session_state.available_sessions = []
    st.session_state.selected_structure_path = None
    st.session_state.eval_sets = []
    st.session_state.eval_results = []
    st.session_state.selected_eval_set = None
    st.session_state.selected_eval_cases = []


def _login_user(username: str) -> bool:
    """Set the active user if the provided username is valid."""
    normalized = username.strip()
    if not normalized:
        st.warning("Please enter a username.")
        return False

    if normalized != st.session_state.user_id:
        _reset_user_session_state()
    st.session_state.user_id = normalized
    return True


def _logout_user() -> None:
    """Clear the active user and all user-scoped UI state."""
    _reset_user_session_state()
    st.session_state.user_id = None
    st.session_state.login_username = ""


def create_metric_config(metric_name, threshold, judge_model="gemini-2.5-flash", num_samples=5, **kwargs):
    """Create a metric configuration dict for the API."""
    metric_config = {
        "metricName": metric_name,
        "threshold": threshold,  # Top-level threshold for backward compatibility
        "criterion": {
            "threshold": threshold
        }
    }
    
    # Add judge model options for LLM-based metrics (require API key)
    llm_metrics = [
        "response_evaluation_score",
        "safety_v1",
        "final_response_match_v2",
        "rubric_based_final_response_quality_v1",
        "hallucinations_v1",
        "rubric_based_tool_use_quality_v1",
        "per_turn_user_simulator_quality_v1"
    ]
    
    if metric_name in llm_metrics:
        metric_config["criterion"]["judgeModelOptions"] = {
            "judgeModel": judge_model,
            "numSamples": num_samples
        }
    
    # Add metric-specific options
    if metric_name == "tool_trajectory_avg_score" and "match_type" in kwargs:
        metric_config["criterion"]["matchType"] = kwargs["match_type"]
    
    if metric_name == "hallucinations_v1" and "evaluate_intermediate" in kwargs:
        metric_config["criterion"]["evaluateIntermediateNlResponses"] = kwargs["evaluate_intermediate"]
    
    return metric_config


def render_content_parts(parts, role="user"):
    """Render content parts (text, function calls, function responses) in a chat-like format."""
    if not parts:
        return
    
    for part in parts:
        # Handle text content
        if "text" in part:
            st.markdown(part["text"])
        
        # Handle function calls
        if "functionCall" in part:
            fc = part["functionCall"]
            with st.expander(f"🔧 Function Call: {fc.get('name', 'Unknown')}", expanded=False):
                st.json(fc.get('args', {}))
        
        # Handle function responses
        if "functionResponse" in part:
            fr = part["functionResponse"]
            with st.expander(f"📥 Function Response: {fr.get('name', 'Unknown')}", expanded=False):
                response_data = fr.get('response', {})
                if response_data:
                    st.json(response_data)
        
        # Handle inline data (images, etc.)
        if "inlineData" in part:
            inline = part["inlineData"]
            mime_type = inline.get("mimeType", "")
            if mime_type.startswith("image/"):
                try:
                    import base64
                    img_data = base64.b64decode(inline.get("data", ""))
                    st.image(img_data, caption=inline.get("displayName", "Image"))
                except:
                    st.caption(f"📷 {inline.get('displayName', 'Image')}")


def render_invocation(invocation_data):
    """Render an invocation (user content and final response) as a conversation."""
    if not invocation_data:
        return
    
    invocation_id = invocation_data.get("invocationId", "Unknown")
    st.markdown(f"**Invocation ID:** `{invocation_id}`")
    
    # Render user content
    user_content = invocation_data.get("userContent", {})
    if user_content:
        user_parts = user_content.get("parts", [])
        if user_parts:
            with st.chat_message("user"):
                render_content_parts(user_parts, role="user")
    
    # Render final response
    final_response = invocation_data.get("finalResponse", {})
    if final_response:
        response_parts = final_response.get("parts", [])
        if response_parts:
            with st.chat_message("assistant"):
                render_content_parts(response_parts, role="assistant")


def render_stream_timeline(container, timeline):
    """Render streaming events in the exact order they arrive."""
    with container.container():
        for event in timeline:
            event_type = event.get("type")

            if event_type == "thought":
                with st.expander("🤔 Thinking...", expanded=False):
                    st.markdown(event.get("text", ""))

            elif event_type == "function_call":
                with st.expander(event.get("name", "Unknown"), expanded=False, icon="🔧"):
                    st.json(event.get("args", {}))

            elif event_type == "function_response":
                with st.expander(event.get("name", "Unknown"), expanded=False, icon="📥"):
                    st.json(event.get("response", {}))

            elif event_type == "text":
                st.markdown(event.get("text", ""))


def merge_replayed_text(current_text: str, incoming_text: str) -> str:
    """Merge text chunks that may be deltas or replayed full snapshots."""
    if not incoming_text:
        return current_text
    if not current_text:
        return incoming_text
    if incoming_text.startswith(current_text):
        return incoming_text
    if current_text.endswith(incoming_text):
        return current_text

    max_overlap = min(len(current_text), len(incoming_text))
    for overlap in range(max_overlap, 0, -1):
        if current_text.endswith(incoming_text[:overlap]):
            return current_text + incoming_text[overlap:]

    return current_text + incoming_text


def compact_repeated_prefix_snapshots(text: str) -> str:
    """Reduce strings shaped like A + A+B + A+B+C to the latest snapshot."""
    if not text:
        return text

    compacted = text
    changed = True
    while changed:
        changed = False
        max_prefix = len(compacted) // 2
        for size in range(max_prefix, 3, -1):
            prefix = compacted[:size]
            rest = compacted[size:]
            if rest.startswith(prefix):
                compacted = rest
                changed = True
                break
    return compacted


def merge_unique_dict_items(existing: list[dict], incoming: list[dict]) -> list[dict]:
    """Append dict items while preserving order and removing exact duplicates."""
    merged = list(existing)
    seen = {json.dumps(item, sort_keys=True, ensure_ascii=True) for item in existing}
    for item in incoming:
        marker = json.dumps(item, sort_keys=True, ensure_ascii=True)
        if marker in seen:
            continue
        seen.add(marker)
        merged.append(item)
    return merged


def merge_agent_snapshot(prev: dict, curr: dict) -> dict | None:
    """Merge consecutive assistant events when later events replay earlier text."""
    prev_content = prev.get("content", "")
    curr_content = curr.get("content", "")
    prev_thought = prev.get("thought", "")
    curr_thought = curr.get("thought", "")

    content_replay = bool(
        prev_content
        and curr_content
        and (
            curr_content.startswith(prev_content)
            or prev_content.endswith(curr_content)
        )
    )
    thought_replay = bool(
        prev_thought
        and curr_thought
        and (
            curr_thought.startswith(prev_thought)
            or prev_thought.endswith(curr_thought)
        )
    )
    metadata_only = not curr_content and not curr_thought

    if not (content_replay or thought_replay or metadata_only):
        return None

    merged = dict(prev)
    if curr_content:
        merged["content"] = compact_repeated_prefix_snapshots(
            merge_replayed_text(prev_content, curr_content)
        )
    if curr_thought:
        merged["thought"] = compact_repeated_prefix_snapshots(
            merge_replayed_text(prev_thought, curr_thought)
        )

    for key in ("structure_path", "plot_path", "artifact_path"):
        if curr.get(key):
            merged[key] = curr[key]

    merged["function_calls"] = merge_unique_dict_items(
        prev.get("function_calls", []),
        curr.get("function_calls", []),
    )
    merged["function_responses"] = merge_unique_dict_items(
        prev.get("function_responses", []),
        curr.get("function_responses", []),
    )
    return merged


def collapse_replayed_messages(messages: list[dict]) -> list[dict]:
    """Collapse ADK replay/snapshot events into stable chat messages."""
    collapsed = []
    for msg in messages:
        if not collapsed:
            collapsed.append(msg)
            continue

        prev = collapsed[-1]
        if prev.get("role") == "agent" and msg.get("role") == "agent":
            merged = merge_agent_snapshot(prev, msg)
            if merged is not None:
                collapsed[-1] = merged
                continue

        if msg.get("role") == "agent" and msg.get("content"):
            msg = dict(msg)
            msg["content"] = compact_repeated_prefix_snapshots(msg["content"])
        if msg.get("role") == "agent" and msg.get("thought"):
            msg = dict(msg)
            msg["thought"] = compact_repeated_prefix_snapshots(msg["thought"])

        collapsed.append(msg)
    return collapsed


def dedupe_agent_thoughts(messages: list[dict]) -> list[dict]:
    """Remove repeated thought snapshots while preserving tool/text messages."""
    deduped = []
    seen_thoughts: list[tuple[str, int]] = []
    for msg in messages:
        if msg.get("role") != "agent" or not msg.get("thought"):
            deduped.append(msg)
            continue

        thought = compact_repeated_prefix_snapshots(msg["thought"]).strip()
        normalized = " ".join(thought.split())
        duplicate = any(normalized == seen for seen, _idx in seen_thoughts)
        if duplicate:
            msg = dict(msg)
            msg.pop("thought", None)
        else:
            for seen, idx in list(seen_thoughts):
                if seen and seen in normalized:
                    deduped[idx] = dict(deduped[idx])
                    deduped[idx].pop("thought", None)
            seen_thoughts = [
                (seen, idx)
                for seen, idx in seen_thoughts
                if not (seen and seen in normalized)
            ]
            if any(normalized in seen for seen, _idx in seen_thoughts):
                msg = dict(msg)
                msg.pop("thought", None)
            else:
                seen_thoughts.append((normalized, len(deduped)))
                msg = dict(msg)
                msg["thought"] = thought
        deduped.append(msg)
    return deduped


def upsert_timeline_text(timeline: list[dict], text: str) -> None:
    """Keep only one live assistant text block while streaming cumulative snapshots."""
    timeline[:] = [event for event in timeline if event.get("type") != "text"]
    if text:
        timeline.append({"type": "text", "text": text})


def upsert_timeline_thought(timeline: list[dict], text: str) -> None:
    """Merge adjacent thought snapshots instead of rendering repeated expanders."""
    if not text:
        return
    compacted = compact_repeated_prefix_snapshots(text)
    if timeline and timeline[-1].get("type") == "thought":
        timeline[-1]["text"] = compact_repeated_prefix_snapshots(
            merge_replayed_text(timeline[-1].get("text", ""), compacted)
        )
        return
    timeline.append({"type": "thought", "text": compacted})


def upsert_timeline_event(timeline: list[dict], event: dict) -> None:
    """Append tool events once, updating repeats that share the same id/type."""
    event_id = event.get("id")
    event_type = event.get("type")
    if event_id:
        for idx, existing in enumerate(timeline):
            if existing.get("type") == event_type and existing.get("id") == event_id:
                timeline[idx] = event
                return
    if timeline and timeline[-1] == event:
        return
    timeline.append(event)


def visualize_structure(structure_path, height=400, width=600):
    """Visualize atomic structure using ASE and py3Dmol, with unit cell box."""
    try:
        # Read structure with ASE
        atoms = read(structure_path)
        
        # Convert to XYZ format string for py3Dmol
        from io import StringIO
        xyz_str = StringIO()
        from ase.io import write
        write(xyz_str, atoms, format='xyz')
        xyz_data = xyz_str.getvalue()
        
        # Create py3Dmol view
        view = py3Dmol.view(width=width, height=height)
        view.addModel(xyz_data, 'xyz')
        view.setStyle({'sphere': {'scale': 0.3}, 'stick': {'radius': 0.15}})

        # Draw unit cell lattice box if a periodic cell is defined
        if atoms.pbc.any():
            a, b, c = atoms.cell[0], atoms.cell[1], atoms.cell[2]
            # 8 corners of the parallelepiped
            corners = [
                a * 0,          # 0: origin
                a,              # 1
                b,              # 2
                c,              # 3
                a + b,          # 4
                a + c,          # 5
                b + c,          # 6
                a + b + c,      # 7
            ]

            def _pt(v):
                return {'x': float(v[0]), 'y': float(v[1]), 'z': float(v[2])}

            # 12 edges connecting the corners
            edges = [
                (0, 1), (0, 2), (0, 3),
                (1, 4), (1, 5),
                (2, 4), (2, 6),
                (3, 5), (3, 6),
                (4, 7), (5, 7), (6, 7),
            ]
            for i, j in edges:
                view.addLine({
                    'start': _pt(corners[i]),
                    'end':   _pt(corners[j]),
                    'color': 'black',
                    'linewidth': 4,
                })

        view.zoomTo()
        
        # Render in Streamlit
        html = view._make_html()
        components.html(html, height=height, width=width)
        
        # Show additional info
        st.caption(f"Formula: {atoms.get_chemical_formula()}, Atoms: {len(atoms)}")
        
        return True
    except Exception as e:
        st.error(f"Failed to visualize structure: {e}")
        return False

def list_sessions():
    """Fetch all sessions for the current user from the API."""
    try:
        response = requests.get(
            f"{API_BASE_URL}/apps/{APP_NAME}/users/{st.session_state.user_id}/sessions",
            headers={"Content-Type": "application/json"}
        )
        if response.status_code == 200:
            #print("Sessions response:", response.json()[-1])
            sessions = response.json()
            # Sort by lastUpdateTime (timestamp-based) descending
            st.session_state.available_sessions = sorted(
                sessions, 
                key=lambda s: s.get('lastUpdateTime', 0.0), 
                reverse=True
            )
            return True
        else:
            st.warning(f"Failed to fetch sessions: {response.text}")
            return False
    except Exception as e:
        st.error(f"Error fetching sessions: {e}")
        return False


def list_eval_sets():
    """Fetch all eval sets for the app."""
    try:
        response = requests.get(
            f"{API_BASE_URL}/apps/{APP_NAME}/eval-sets",
            headers={"Content-Type": "application/json"}
        )
        if response.status_code == 200:
            result = response.json()
            st.session_state.eval_sets = result.get("evalSetIds", [])
            return True
        else:
            st.warning(f"Failed to fetch eval sets: {response.text}")
            return False
    except Exception as e:
        st.error(f"Error fetching eval sets: {e}")
        return False


def create_eval_set(eval_set_name: str =None, description: str = ""):
    """Create a new evaluation set."""
    try:
        # Generate unique eval set ID (alphanumeric + underscore only)
        if eval_set_name is None:
            eval_set_name = f"eval_{uuid.uuid4().hex[:12]}_{int(time.time())}"
        
        eval_set_data = {
            "evalSet": {
                #"eval_set_id": eval_set_id,
                "eval_set_id": eval_set_name,
                "name": eval_set_name,
                "description": description,
                "eval_cases": []
            }
        }
        response = requests.post(
            f"{API_BASE_URL}/apps/{APP_NAME}/eval-sets",
            headers={"Content-Type": "application/json"},
            data=json.dumps(eval_set_data)
        )
        if response.status_code == 200:
            list_eval_sets()
            return True
        else:
            st.error(f"Failed to create eval set: {response.text}")
            return False
    except Exception as e:
        st.error(f"Error creating eval set: {e}")
        return False


def add_session_to_eval_set(eval_set_id: str, session_id: str):
    """Add current session to an eval set."""
    try:
        # Generate unique eval case ID (alphanumeric + underscore only)
        eval_case_id = f"eval_case_{uuid.uuid4().hex[:12]}_{int(time.time())}"
        
        payload = {
            "evalId": eval_case_id,
            "evalSetId": eval_set_id,
            "sessionId": session_id,
            "userId": st.session_state.user_id
        }
        response = requests.post(
            f"{API_BASE_URL}/apps/{APP_NAME}/eval_sets/{eval_set_id}/add_session",
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload)
        )
        if response.status_code == 200:
            return True
        else:
            st.error(f"Failed to add session to eval set: {response.text}")
            return False
    except Exception as e:
        st.error(f"Error adding session to eval set: {e}")
        return False


def run_evaluation(
    eval_set_id: str, 
    eval_case_ids: list = None,
    eval_metrics: list = []
    ):
    """Run evaluation on an eval set."""
    try:
        payload = {
            "evalCaseIds": eval_case_ids or [],
            "evalMetrics": eval_metrics  # Can be extended to support custom metrics
        }
        response = requests.post(
            f"{API_BASE_URL}/apps/{APP_NAME}/eval-sets/{eval_set_id}/run",
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload)
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Failed to run evaluation: {response.text}")
            return None
    except Exception as e:
        st.error(f"Error running evaluation: {e}")
        return None


def list_eval_results():
    """Fetch all eval results for the app."""
    try:
        response = requests.get(
            f"{API_BASE_URL}/apps/{APP_NAME}/eval-results",
            headers={"Content-Type": "application/json"}
        )
        if response.status_code == 200:
            result = response.json()
            # Sort by eval result ID in descending order (newest first)
            eval_results = result.get("evalResultIds", [])
            st.session_state.eval_results = sorted(eval_results, reverse=True)
            return True
        else:
            st.warning(f"Failed to fetch eval results: {response.text}")
            return False
    except Exception as e:
        st.error(f"Error fetching eval results: {e}")
        return False


def get_eval_result(eval_result_id: str):
    """Get detailed results for a specific evaluation."""
    try:
        response = requests.get(
            f"{API_BASE_URL}/apps/{APP_NAME}/eval-results/{eval_result_id}",
            headers={"Content-Type": "application/json"}
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Failed to get eval result: {response.text}")
            return None
    except Exception as e:
        st.error(f"Error getting eval result: {e}")
        return None


def get_eval_case_list(eval_set_id: str):
    """Get details of a specific eval set including its eval case IDs."""
    try:
        response = requests.get(
            f"{API_BASE_URL}/apps/{APP_NAME}/eval_sets/{eval_set_id}/evals",
            headers={"Content-Type": "application/json"}
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Failed to get eval set details: {response.text}")
            return None
    except Exception as e:
        st.error(f"Error getting eval set details: {e}")
        return None


def get_eval_case_details(eval_set_id: str, eval_case_id: str):
    """Get detailed information for a specific eval case."""
    try:
        response = requests.get(
            f"{API_BASE_URL}/apps/{APP_NAME}/eval_sets/{eval_set_id}/evals/{eval_case_id}",
            headers={"Content-Type": "application/json"}
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Failed to get eval case details: {response.text}")
            return None
    except Exception as e:
        st.error(f"Error getting eval case details: {e}")
        return None


def delete_eval_case(eval_set_id: str, eval_case_id: str):
    """Delete a specific eval case from an eval set."""
    try:
        response = requests.delete(
            f"{API_BASE_URL}/apps/{APP_NAME}/eval_sets/{eval_set_id}/evals/{eval_case_id}",
            headers={"Content-Type": "application/json"}
        )
        if response.status_code == 200:
            return True
        else:
            st.error(f"Failed to delete eval case: {response.text}")
            return False
    except Exception as e:
        st.error(f"Error deleting eval case: {e}")
        return False


def load_session(session_id):
    """Load a session and its message history."""
    try:
        # Fetch session details
        response = requests.get(
            f"{API_BASE_URL}/apps/{APP_NAME}/users/{st.session_state.user_id}/sessions/{session_id}",
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            session_data = response.json()
            
            # Set current session
            st.session_state.session_id = session_id
            
            # Load messages from events (not messages)
            messages = []
            artifacts = []
            
            for event in session_data.get('events', []):
                content = event.get('content', {})
                author = event.get('author', 'agent')
                #print('event_content:', content)
                #print('event_author:', author)
                # Determine role (user or agent)
                role = 'user' if author == 'user' else 'agent'
                
                msg_entry = {'role': role}
                
                # Extract text from parts
                parts = content.get('parts', [{}])
                if not parts:
                    continue

                # Initialize paths
                structure_path = None
                plot_path = None
                artifact_path = None
                text = ""
                thought_text = ""
                function_calls = []
                function_responses = []

                # Process all parts, separating thought parts from response parts
                for part in parts:
                    if part.get("thought"):
                        thought_text = merge_replayed_text(
                            thought_text,
                            part.get("text", ""),
                        )
                        continue

                    # Extract text content
                    if "text" in part:
                        text = merge_replayed_text(text, part.get("text", ""))

                    # Extract function calls
                    if "functionCall" in part:
                        fc = part["functionCall"]
                        function_calls.append({"id": fc.get("id"), "name": fc.get("name", "Unknown"), "args": fc.get("args", {})})

                    # Extract paths from functionResponse
                    if "functionResponse" in part:
                        fr = part["functionResponse"]
                        resp = fr.get("response", {})
                        function_responses.append({"id": fr.get("id"), "name": fr.get("name", "Unknown"), "response": resp})

                        # Check for plot path
                        p_path = resp.get("plot_path")
                        if p_path:
                            plot_path = p_path
                            artifacts.append({"name": os.path.basename(p_path), "url": p_path})
                            
                        # Check for structure path
                        s_path = resp.get("structure_path")
                        if s_path:
                            structure_path = s_path
                            artifacts.append({"name": os.path.basename(s_path), "url": s_path})
                            
                        a_path = resp.get("artifact_path")
                        if a_path:
                            artifact_path=a_path
                            artifacts.append({"name": os.path.basename(a_path), "url": a_path})
                            
                if thought_text:
                    msg_entry["thought"] = compact_repeated_prefix_snapshots(thought_text)
                if function_calls:
                    msg_entry["function_calls"] = function_calls
                if function_responses:
                    msg_entry["function_responses"] = function_responses
                if text:
                    msg_entry["content"] = compact_repeated_prefix_snapshots(text)
                # Add extracted paths to message entry
                if structure_path:
                    msg_entry["structure_path"] = structure_path
                if plot_path:
                    msg_entry["plot_path"] = plot_path
                if artifact_path:
                    msg_entry["artifact_path"] = artifact_path
                messages.append(msg_entry)

            # Post-process: embed each functionResponse into its matching functionCall by ID.
            # Then remove messages that contained only those now-merged responses.
            fr_by_id = {
                fr["id"]: fr
                for msg in messages
                for fr in msg.get("function_responses", [])
                if fr.get("id")
            }
            merged_fr_ids = set()
            for msg in messages:
                for fc in msg.get("function_calls", []):
                    fc_id = fc.get("id")
                    if fc_id and fc_id in fr_by_id:
                        fc["response"] = fr_by_id[fc_id]["response"]
                        merged_fr_ids.add(fc_id)

            def _is_merged_response_only(msg):
                frs = msg.get("function_responses", [])
                return (
                    frs
                    and not msg.get("function_calls")
                    and not msg.get("content")
                    and not msg.get("thought")
                    and not msg.get("structure_path")
                    and not msg.get("plot_path")
                    and not msg.get("artifact_path")
                    and all(fr.get("id") in merged_fr_ids for fr in frs if fr.get("id"))
                )

            messages = [m for m in messages if not _is_merged_response_only(m)]
            messages = collapse_replayed_messages(messages)
            messages = dedupe_agent_thoughts(messages)

            st.session_state.messages = messages
            st.session_state.artifacts = artifacts
            st.session_state.audio_files = []
            return True
        else:
            st.error(f"Failed to load session: {response.text}")
            return False
    except Exception as e:
        st.error(f"Error loading session: {e}")
        return False


def create_session():
    """
    Create a new session with the speaker agent.
    
    This function:
    1. Generates a unique session ID based on timestamp
    2. Sends a POST request to the ADK API to create a session
    3. Updates the session state variables if successful
    4. Refreshes the session list
    
    Returns:
        bool: True if session was created successfully, False otherwise
    
    API Endpoint:
        POST /apps/{app_name}/users/{user_id}/sessions/{session_id}
    """
    session_id = f"session-{int(time.time())}"
    response = requests.post(
        f"{API_BASE_URL}/apps/{APP_NAME}/users/{st.session_state.user_id}/sessions/{session_id}",
        headers={"Content-Type": "application/json"},
        data=json.dumps({})
    )
    
    if response.status_code == 200:
        st.session_state.session_id = session_id
        st.session_state.messages = []
        st.session_state.audio_files = []
        # Refresh session list
        list_sessions()
        return True
    else:
        st.error(f"Failed to create session: {response.text}")
        return False

def send_message_sse(message, attachments=None):
    """Stream agent response in-place so thought/tool/text parts render immediately."""
    if not st.session_state.session_id:
        st.error("No active session. Please create a session first.")
        return False

    attachments_payload = attachments if attachments else []

    with st.chat_message("user"):
        st.write(message)
        for att_path in attachments_payload:
            st.caption(f"📎 {os.path.basename(att_path)}")

    message_with_attachments = message
    if attachments_payload:
        attachment_paths = "\n".join(attachments_payload)
        message_with_attachments = f"{message}\n\nAttached files:\n{attachment_paths}"

    try:
        with requests.post(
            f"{API_BASE_URL}/run_sse",
            headers={
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
            data=json.dumps({
                "app_name": APP_NAME,
                "user_id": st.session_state.user_id,
                "session_id": st.session_state.session_id,
                "new_message": {
                    "role": "user",
                    "parts": [{"text": message_with_attachments}],
                }
            }),
            stream=True,
            timeout=(10, None),
        ) as response:
            if response.status_code != 200:
                st.error(f"Error: {response.text}")
                return False

            timeline = []
            accumulated_content = ""
            with st.chat_message("agent"):
                timeline_holder = st.empty()

                for line in response.iter_lines(chunk_size=1, decode_unicode=True):
                    if not line or not line.startswith("data: "):
                        continue
                    data_str = line[len("data: "):]
                    if data_str == "[DONE]":
                        break
                    try:
                        event = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    for part in event.get("content", {}).get("parts", []):
                        if part.get("thought"):
                            upsert_timeline_thought(timeline, part.get("text", ""))
                            render_stream_timeline(timeline_holder, timeline)

                        elif "functionCall" in part:
                            fc = part["functionCall"]
                            upsert_timeline_event(timeline, {
                                "type": "function_call",
                                "id": fc.get("id"),
                                "name": fc.get("name", "Unknown"),
                                "args": fc.get("args", {}),
                            })
                            render_stream_timeline(timeline_holder, timeline)

                        elif "functionResponse" in part:
                            fr = part["functionResponse"]
                            resp = fr.get("response", {})
                            for path in _extract_response_paths(resp).values():
                                resolved = resolve_path(path)
                                if os.path.exists(resolved) and is_structure_file(resolved):
                                    st.session_state.selected_structure_path = resolved
                            upsert_timeline_event(timeline, {
                                "type": "function_response",
                                "id": fr.get("id"),
                                "name": fr.get("name", "Unknown"),
                                "response": resp,
                            })
                            render_stream_timeline(timeline_holder, timeline)

                        elif "text" in part:
                            accumulated_content = merge_replayed_text(
                                accumulated_content,
                                part["text"],
                            )
                            upsert_timeline_text(
                                timeline,
                                compact_repeated_prefix_snapshots(accumulated_content),
                            )
                            render_stream_timeline(timeline_holder, timeline)

    except requests.RequestException as exc:
        st.error(f"Streaming error: {exc}")
        return False

    load_session(st.session_state.session_id)
    st.session_state.uploader_key += 1
    return True


def render_message(msg, msg_idx) -> None:
    if not msg or (len(msg) == 1 and "role" in msg):
        return

    if msg["role"] == "user":
        with st.chat_message("user"):
            if "content" in msg:
                st.markdown(compact_repeated_prefix_snapshots(msg["content"]))
            if "attachments" in msg and msg["attachments"]:
                for att_path in msg["attachments"]:
                    st.caption(f"📎 {os.path.basename(att_path)}")
        return

    with st.chat_message("agent"):
        if msg.get("thought"):
            with st.expander("🤔 Thinking...", expanded=False):
                st.markdown(msg["thought"])
        for fc in msg.get("function_calls", []):
            with st.expander(fc["name"], expanded=False, icon="🔧"):
                st.json(fc["args"])
            if "response" in fc:
                with st.expander(fc["name"], expanded=False, icon="📥"):
                    st.json(fc["response"])
        for fr in msg.get("function_responses", []):
            with st.expander(fr["name"], expanded=False, icon="📥"):
                st.json(fr["response"])
        if msg.get("content"):
            st.markdown(compact_repeated_prefix_snapshots(msg["content"]))

        if "structure_path" in msg and os.path.exists(resolve_path(msg["structure_path"])):
            if st.button(f"🔬 View Structure: {os.path.basename(msg['structure_path'])}", key=f"view_struct_{msg_idx}"):
                st.session_state.selected_structure_path = msg["structure_path"]
                st.rerun()
            with open(resolve_path(msg["structure_path"]), "rb") as f:
                st.download_button(
                    label=f"⬇️ Download {os.path.basename(msg['structure_path'])}",
                    data=f.read(),
                    file_name=os.path.basename(msg["structure_path"]),
                    key=f"download_struct_{msg_idx}_{os.path.basename(msg['structure_path'])}"
                )
        if "plot_path" in msg and os.path.exists(resolve_path(msg["plot_path"])):
            st.image(resolve_path(msg["plot_path"]), use_container_width=True)
            with open(resolve_path(msg["plot_path"]), "rb") as f:
                st.download_button(
                    label=f"⬇️ Download {os.path.basename(msg['plot_path'])}",
                    data=f.read(),
                    file_name=os.path.basename(msg["plot_path"]),
                    key=f"download_plot_{msg_idx}_{os.path.basename(msg['plot_path'])}"
                )
        if "artifact_path" in msg and os.path.exists(resolve_path(msg["artifact_path"])):
            st.divider()
            st.markdown("**Model File:**")
            st.info(f"📦 {os.path.basename(msg['artifact_path'])}")
            with open(resolve_path(msg["artifact_path"]), "rb") as f:
                st.download_button(
                    label=f"⬇️ Download {os.path.basename(msg['artifact_path'])}",
                    data=f.read(),
                    file_name=os.path.basename(msg["artifact_path"]),
                    key=f"download_model_{msg_idx}_{os.path.basename(msg['artifact_path'])}"
                )

# UI Components
st.title("MatCreator")

with st.sidebar:
    st.header("User")
    if st.session_state.user_id:
        st.success(f"Signed in as `{st.session_state.user_id}`")
        if st.button("Switch User", width='stretch', key="switch_user"):
            _logout_user()
            st.rerun()
    else:
        st.text_input(
            "Username",
            key="login_username",
            placeholder="Enter your username",
        )
        if st.button("Enter System", width='stretch', key="login_user"):
            if _login_user(st.session_state.login_username):
                list_sessions()
                st.rerun()

    st.divider()

if not st.session_state.user_id:
    st.info("Enter your username in the sidebar to access your own conversation history.")
    st.stop()

# Sidebar - Mode selector at the top
with st.sidebar:
    st.header("Navigation")
    view_mode = st.radio(
        "Select Mode:",
        ["💬 Session & Conversation", "📊 Evaluation & Results"],
        index=0 if st.session_state.view_mode == "session" else 1,
        key="mode_selector"
    )
    
    # Update session state based on selection
    if view_mode == "💬 Session & Conversation":
        st.session_state.view_mode = "session"
    else:
        st.session_state.view_mode = "evaluation"
    
    st.divider()

# ==================== SESSION MODE: Sidebar & Main Content ====================
if st.session_state.view_mode == "session":
    # Sidebar for session management
    with st.sidebar:
        st.header("Session Management")
        
        # New session button
        if st.button("➕ New Session", width='stretch', key="new_session_tab1"):
            if create_session():
                st.rerun()
        
        # Refresh sessions button
        if st.button("🔄 Refresh Sessions", width='stretch', key="refresh_sessions_tab1"):
            list_sessions()
            st.rerun()
        
        st.divider()
        
        # Display current session
        if st.session_state.session_id:
            st.success(f"**Active:** {st.session_state.session_id}")
        else:
            st.warning("No active session")
        
        st.divider()
        
        # Session history
        st.subheader("Session History")
        
        # Load sessions if not already loaded
        if not st.session_state.available_sessions:
            list_sessions()
        
        if st.session_state.available_sessions:
            for idx, session in enumerate(st.session_state.available_sessions):
                session_id = session.get('id', None)
                
                # Skip invalid sessions
                if not session_id:
                    continue
                
                is_current = session_id == st.session_state.session_id
                
                # Parse timestamp from session_id (format: session-TIMESTAMP)
                try:
                    timestamp = int(session_id.split('-')[1])
                    date_str = time.strftime('%Y-%m-%d %H:%M', time.localtime(timestamp))
                except:
                    date_str = session_id[:20]  # Fallback to truncated session_id
                
                # Session button with indicator - use idx to ensure unique keys
                label = f"{'🔵' if is_current else '⚪'} {date_str}"
                if st.button(label, key=f"session_btn_{idx}_{session_id}", width='stretch', disabled=is_current):
                    if load_session(session_id):
                        st.rerun()
        else:
            st.caption("No sessions available")
        
        st.divider()
        st.caption("This app interacts with the MatCreator Agent via the ADK API Server.")
        st.caption("Make sure the ADK API Server is running on port 8000.")

        st.divider()
        st.subheader("Artifacts")
        if st.session_state.artifacts:
            for idx, art in enumerate(st.session_state.artifacts):
                name = art.get("name", os.path.basename(art.get("url", "artifact")))
                url = art.get("url", "")
                
                # Show artifact name and download button
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.caption(f"📄 {name}")
                with col2:
                    if os.path.exists(resolve_path(url)):
                        try:
                            with open(resolve_path(url), "rb") as f:
                                st.download_button(
                                    label="⬇️",
                                    data=f.read(),
                                    file_name=name,
                                    key=f"download_artifact_{idx}_{name}",
                                    width='stretch'
                                )
                        except Exception as e:
                            st.caption("❌")
        else:
            st.caption("No artifacts yet")
    
    # Main content - Chat interface with right-side plot panel
    float_init()
    chat_col, plot_col = st.columns([3, 2], gap="medium")

    with chat_col:
        st.subheader("Conversation")

        # Display messages
        for msg_idx, msg in enumerate(st.session_state.messages):
            render_message(msg, msg_idx)

        # Input for new messages
        if st.session_state.session_id:  # Only show input if session exists
            # File upload in chat area
            uploaded_files = st.file_uploader(
                "📎 Attach files (optional)",
                type=["extxyz", "xyz", "cif", "vasp", "txt"],
                accept_multiple_files=True,
                key=f"uploader_{st.session_state.uploader_key}"
            )
            
            user_input = st.chat_input("Type your message...")
            if user_input:
                # Process attachments if present
                attachments = []
                if uploaded_files:
                    os.makedirs("/tmp/streamlit_uploads", exist_ok=True)
                    for uf in uploaded_files:
                        save_path = os.path.join("/tmp/streamlit_uploads", uf.name)
                        with open(save_path, "wb") as f:
                            f.write(uf.getbuffer())
                        attachments.append(os.path.abspath(save_path))
                
                send_message_sse(user_input, attachments)
                st.rerun()  # Rerun to update the UI with new messages
        else:
            st.info("👈 Create a session to start chatting")

    with plot_col:
        float_parent(css="top: 3.5rem; overflow-y: auto; max-height: calc(100vh - 4rem);")

        if st.session_state.selected_structure_path and os.path.exists(resolve_path(st.session_state.selected_structure_path)):
            st.subheader("Structure Viewer")
            selected_path = Path(resolve_path(st.session_state.selected_structure_path))
            try:
                selected_label = selected_path.relative_to(WORKSPACE_ROOT)
            except ValueError:
                selected_label = selected_path
            st.caption(f"🔬 `{selected_label}`")
            visualize_structure(resolve_path(st.session_state.selected_structure_path), height=280, width=400)
            if st.button("✖ Clear", key="clear_structure"):
                st.session_state.selected_structure_path = None
                st.rerun()
            st.divider()

        st.subheader("Session Files")
        if st.session_state.session_id:
            session_workdir = _session_workdir()
            if st.button("🔄 Refresh", key="refresh_session_files"):
                st.rerun()
            if session_workdir and session_workdir.exists():
                all_files = sorted(session_workdir.rglob("*"))
                files_only = [f for f in all_files if f.is_file()]
                if files_only:
                    for idx, file_path in enumerate(files_only):
                        rel = file_path.relative_to(session_workdir)
                        depth = len(rel.parts) - 1
                        indent = "\u00a0" * (depth * 4)
                        size = file_path.stat().st_size
                        size_str = f"{size} B" if size < 1024 else f"{size // 1024} KB"
                        if is_structure_file(file_path):
                            col1, col2, col3 = st.columns([4, 1, 1])
                        else:
                            col1, col2 = st.columns([4, 1])
                            col3 = None
                        with col1:
                            st.markdown(
                                f"{indent}📄 `{rel}` "
                                f"<span style='color:gray;font-size:0.8em'>{size_str}</span>",
                                unsafe_allow_html=True,
                            )
                        with col2:
                            with open(file_path, "rb") as f:
                                st.download_button(
                                    "⬇️",
                                    data=f.read(),
                                    file_name=file_path.name,
                                    key=f"dl_session_{idx}_{file_path.name}",
                                )
                        if col3 is not None:
                            with col3:
                                if st.button(
                                    "🔬",
                                    key=f"preview_session_{idx}_{file_path.name}",
                                    help=f"Preview structure: {rel}",
                                ):
                                    st.session_state.selected_structure_path = str(file_path)
                                    st.rerun()
                else:
                    st.info("No files yet in this session's working directory.")
            else:
                st.info("Session working directory not created yet.")
        else:
            st.info("No active session.")

# ==================== EVALUATION MODE: Sidebar & Main Content ====================
elif st.session_state.view_mode == "evaluation":
    # Sidebar for evaluation management
    with st.sidebar:
        st.header("Evaluation Management")
        
        # Refresh eval sets button
        if st.button("🔄 Refresh Eval Sets", width='stretch', key="refresh_eval_tab2"):
            list_eval_sets()
            list_eval_results()
            st.rerun()
        
        # Metric Configuration
        with st.expander("⚙️ Configure Evaluation Metrics", expanded=False):
            st.write("Select metrics to use for evaluation:")
            
            # Available prebuilt metrics organized by type
            st.markdown("**🟢 Local Metrics** (No API key needed)")
            local_metric_options = {
                "Response Match Score (ROUGE)": "response_match_score",
                "Tool Trajectory Average Score": "tool_trajectory_avg_score",
            }
            
            st.markdown("**🔴 LLM-based Metrics** (Require Google AI API key)")
            llm_metric_options = {
                "Response Evaluation Score": "response_evaluation_score",
                "Safety V1": "safety_v1",
                "Final Response Match V2": "final_response_match_v2",
                "Rubric-based Final Response Quality V1": "rubric_based_final_response_quality_v1",
                "Hallucinations V1": "hallucinations_v1",
                "Rubric-based Tool Use Quality V1": "rubric_based_tool_use_quality_v1",
                "Per-turn User Simulator Quality V1": "per_turn_user_simulator_quality_v1"
            }
            
            # Combine all metrics for selection
            all_metric_options = {**local_metric_options, **llm_metric_options}
            
            selected_metric_names = st.multiselect(
                "Choose metrics:",
                list(all_metric_options.keys()),
                key="metric_selector"
            )
            
            configured_metrics = []
            
            if selected_metric_names:
                st.divider()
                st.write("**Configure selected metrics:**")
                
                for metric_display_name in selected_metric_names:
                    metric_name = all_metric_options[metric_display_name]
                    
                    with st.container():
                        # Show if metric requires API key
                        is_llm_metric = metric_name in [
                            "response_evaluation_score",
                            "safety_v1",
                            "final_response_match_v2",
                            "rubric_based_final_response_quality_v1",
                            "hallucinations_v1",
                            "rubric_based_tool_use_quality_v1",
                            "per_turn_user_simulator_quality_v1"
                        ]
                        
                        if is_llm_metric:
                            st.markdown(f"**{metric_display_name}** 🔴 _Requires API key_")
                        else:
                            st.markdown(f"**{metric_display_name}** 🟢 _Local_")
                        
                        # Common configuration: Threshold
                        threshold = st.slider(
                            "Threshold",
                            min_value=0.0,
                            max_value=1.0,
                            value=0.7,
                            step=0.05,
                            key=f"threshold_{metric_name}"
                        )
                        
                        # LLM-based metrics configuration
                        llm_metrics = [
                            "response_evaluation_score",
                            "safety_v1",
                            "final_response_match_v2",
                            "rubric_based_final_response_quality_v1",
                            "hallucinations_v1",
                            "rubric_based_tool_use_quality_v1",
                            "per_turn_user_simulator_quality_v1"
                        ]
                        
                        judge_model = "gemini-2.5-flash"
                        num_samples = 5
                        
                        if metric_name in llm_metrics:
                            col1, col2 = st.columns(2)
                            with col1:
                                judge_model = st.text_input(
                                    "Judge Model",
                                    value="gemini-2.5-flash",
                                    key=f"judge_{metric_name}"
                                )
                            with col2:
                                num_samples = st.number_input(
                                    "Samples",
                                    min_value=1,
                                    max_value=10,
                                    value=5,
                                    key=f"samples_{metric_name}"
                                )
                        
                        # Metric-specific options
                        kwargs = {}
                        
                        if metric_name == "tool_trajectory_avg_score":
                            match_type = st.selectbox(
                                "Match Type",
                                ["EXACT", "IN_ORDER", "ANY_ORDER"],
                                key=f"match_type_{metric_name}"
                            )
                            kwargs["match_type"] = match_type
                        
                        if metric_name == "hallucinations_v1":
                            evaluate_intermediate = st.checkbox(
                                "Evaluate intermediate responses",
                                value=False,
                                key=f"eval_intermediate_{metric_name}"
                            )
                            kwargs["evaluate_intermediate"] = evaluate_intermediate
                        
                        # Create metric config
                        metric_config = create_metric_config(
                            metric_name,
                            threshold,
                            judge_model,
                            num_samples,
                            **kwargs
                        )
                        configured_metrics.append(metric_config)
                        
                        st.divider()
                
                # Save configuration button
                if st.button("💾 Save Metric Configuration", key="save_metrics"):
                    st.session_state.configured_metrics = configured_metrics
                    st.success(f"Saved {len(configured_metrics)} metric(s)")
            
            # Display current configuration
            if st.session_state.configured_metrics:
                st.info(f"✓ {len(st.session_state.configured_metrics)} metric(s) configured")
                with st.expander("View configured metrics"):
                    for i, metric in enumerate(st.session_state.configured_metrics, 1):
                        st.write(f"{i}. {metric['metricName']} (threshold: {metric['criterion']['threshold']})")
        
        st.divider()
        
        # Create new eval set
        with st.expander("➕ Create New Eval Set"):
            new_eval_name = st.text_input("Eval Set Name", key="new_eval_name_tab2")
            new_eval_desc = st.text_area("Description (optional)", key="new_eval_desc_tab2", height=80)
            if st.button("Create Eval Set", width='stretch', key="create_eval_tab2"):
                if new_eval_name:
                    if create_eval_set(new_eval_name, new_eval_desc):
                        st.success(f"Created eval set: {new_eval_name}")
                        st.rerun()
                else:
                    st.warning("Please enter a name for the eval set")
        
        st.divider()
        
        # Main eval set selector and operations
        if not st.session_state.eval_sets:
            list_eval_sets()
        
        if st.session_state.eval_sets:
            # Dropdown to select eval set
            selected_eval = st.selectbox(
                "Select Eval Set",
                st.session_state.eval_sets,
                key="eval_set_dropdown_tab2"
            )
            st.session_state.selected_eval_set = selected_eval
            
            # Add current session to selected eval set
            if st.session_state.session_id:
                if st.button("📝 Add Current Session", width='stretch', key="add_session_tab2"):
                    if add_session_to_eval_set(selected_eval, st.session_state.session_id):
                        st.success(f"Added session to {selected_eval}")
                        st.rerun()
            
            # Run evaluation button for selected eval set
            if st.button("▶️ Run Evaluation", width='stretch', key="run_eval_tab2"):
                if not st.session_state.configured_metrics:
                    st.warning("⚠️ No metrics configured. Please configure metrics first.")
                else:
                    # Determine which cases to run
                    eval_case_ids_to_run = st.session_state.selected_eval_cases if st.session_state.selected_eval_cases else None
                    
                    with st.spinner(f"Running evaluation on {selected_eval}..."):
                        result = run_evaluation(
                            selected_eval,
                            eval_case_ids=eval_case_ids_to_run,
                            eval_metrics=st.session_state.configured_metrics
                        )
                        if result:
                            st.success("✅ Evaluation completed!")
                            list_eval_results()
                            st.rerun()
            
            st.divider()
            
            # Display eval cases in the selected eval set
            # Header with delete button
            col1, col2 = st.columns([3, 1])
            with col1:
                st.subheader(f"Eval Cases")
            with col2:
                if st.session_state.selected_eval_cases:
                    if st.button("🗑️ Delete Selected", width='stretch', key="delete_cases_tab2"):
                        deleted_count = 0
                        for case_id in st.session_state.selected_eval_cases:
                            if delete_eval_case(selected_eval, case_id):
                                deleted_count += 1
                        
                        if deleted_count > 0:
                            st.success(f"✅ Deleted {deleted_count} eval case(s)")
                            st.session_state.selected_eval_cases = []
                            st.rerun()
            
            eval_case_ids = get_eval_case_list(selected_eval)
            
            if eval_case_ids:
                # Display each case with a checkbox
                for idx, case_id in enumerate(eval_case_ids):
                    is_selected = case_id in st.session_state.selected_eval_cases
                    # Create checkbox for selection
                    if st.checkbox(f"📋 {case_id}", value=is_selected, key=f"case_checkbox_tab2_{idx}_{case_id}"):
                        if case_id not in st.session_state.selected_eval_cases:
                            st.session_state.selected_eval_cases.append(case_id)
                    else:
                        if case_id in st.session_state.selected_eval_cases:
                            st.session_state.selected_eval_cases.remove(case_id)
            else:
                st.info("No eval cases in this eval set. Add sessions to create eval cases.")
        else:
            st.caption("No eval sets available")
        
        st.divider()
        st.caption("This app interacts with the MatCreator Agent via the ADK API Server.")
        st.caption("Make sure the ADK API Server is running on port 8000.")
    
    # Main content - Eval Results Display
    st.subheader("Evaluation Results")
    
    # Refresh eval results
    if st.button("🔄 Refresh Results", key="refresh_results_tab2"):
        list_eval_results()
        st.rerun()
    
    # Load results if not already loaded
    if not st.session_state.eval_results:
        list_eval_results()
    
    # Filter eval results based on selected eval set
    filtered_eval_results = []
    if st.session_state.eval_results and st.session_state.selected_eval_set:
        for result_id in st.session_state.eval_results:
            # Parse the result_id: format is "{APP_NAME}_{eval_set_id}_{timestamp}"
            # Find the last underscore to separate timestamp from the rest
            parts = result_id.rsplit('_', 1)
            if len(parts) == 2:
                prefix, timestamp_str = parts
                # Extract eval set name by removing APP_NAME prefix
                # Format: "MatCreator_evalsetf6633f" -> "evalsetf6633f"
                if prefix.startswith(f"{APP_NAME}_"):
                    eval_set_name = prefix[len(APP_NAME) + 1:]  # +1 for underscore
                    if eval_set_name == st.session_state.selected_eval_set:
                        try:
                            timestamp = float(timestamp_str)
                            filtered_eval_results.append((result_id, timestamp))
                        except ValueError:
                            # If timestamp parsing fails, still include it with timestamp 0
                            filtered_eval_results.append((result_id, 0))
        
        # Sort by timestamp in descending order (newest first)
        filtered_eval_results.sort(key=lambda x: x[1], reverse=True)
        # Extract just the result IDs
        filtered_eval_results = [r[0] for r in filtered_eval_results]
    
    if filtered_eval_results:
        st.write(f"**Total Results for {st.session_state.selected_eval_set}:** {len(filtered_eval_results)}")
        
        # Dropdown to select eval result
        selected_result_id = st.selectbox(
            "Select Eval Result:",
            filtered_eval_results,
            key="selected_eval_result"
        )
        
        if selected_result_id:
            result_data = get_eval_result(selected_result_id)
            
            if result_data:
                st.divider()
                
                # Display metadata
                st.markdown(f"**Eval Set Result ID:** `{result_data.get('evalSetResultId', 'N/A')}`")
                st.markdown(f"**Eval Set ID:** `{result_data.get('evalSetId', 'N/A')}`")
                
                st.divider()
                
                # Get case results
                case_results = result_data.get('evalCaseResults', [])
                
                if case_results:
                    st.markdown(f"**Total Cases:** {len(case_results)}")
                    
                    # Create list of case IDs for dropdown
                    case_options = [f"Case {idx + 1}: {case.get('evalId', 'N/A')}" for idx, case in enumerate(case_results)]
                    
                    # Dropdown to select eval case
                    selected_case_option = st.selectbox(
                        "Select Eval Case:",
                        case_options,
                        key="selected_eval_case"
                    )
                    
                    # Find selected case index
                    selected_case_idx = case_options.index(selected_case_option)
                    case_result = case_results[selected_case_idx]
                    
                    st.divider()
                    
                    # Display case details
                    st.markdown(f"### {selected_case_option}")
                    
                    # Display overall metrics
                    st.markdown("**Overall Metrics:**")
                    metric_results = case_result.get('overallEvalMetricResults', [])
                    if metric_results:
                        cols = st.columns(len(metric_results))
                        for col_idx, metric in enumerate(metric_results):
                            with cols[col_idx]:
                                metric_name = metric.get('metricName', 'Unknown')
                                metric_score = metric.get('score', 'N/A')
                                st.metric(metric_name, f"{metric_score:.2f}" if isinstance(metric_score, (int, float)) else metric_score)
                    else:
                        st.info("No overall metrics available")
                    
                    st.divider()
                    
                    # Display per-invocation results
                    invocation_results = case_result.get('evalMetricResultPerInvocation', [])
                    if invocation_results:
                        st.markdown(f"**Per-Invocation Results:** ({len(invocation_results)} invocations)")
                        
                        # Create tabs for each invocation
                        invocation_tabs = st.tabs([f"Invocation {i+1}" for i in range(len(invocation_results))])
                        
                        for inv_idx, (tab, inv_result) in enumerate(zip(invocation_tabs, invocation_results)):
                            with tab:
                                # Display metrics for this invocation
                                inv_metrics = inv_result.get('metricResults', [])
                                if inv_metrics:
                                    st.markdown("**Metrics:**")
                                    cols = st.columns(len(inv_metrics))
                                    for col_idx, metric in enumerate(inv_metrics):
                                        with cols[col_idx]:
                                            metric_name = metric.get('metricName', 'Unknown')
                                            metric_score = metric.get('score', 'N/A')
                                            st.metric(metric_name, f"{metric_score:.2f}" if isinstance(metric_score, (int, float)) else metric_score)
                                    
                                    st.divider()
                                
                                # Render the actual invocation (user content and response)
                                actual_invocation = inv_result.get('actualInvocation', {})
                                if actual_invocation:
                                    st.markdown("**Conversation:**")
                                    render_invocation(actual_invocation)
                                else:
                                    st.info("No invocation data available")
                    else:
                        st.info("No per-invocation results available")
                else:
                    st.info("No case results available")
            else:
                st.error("Failed to load eval result details")
    elif st.session_state.selected_eval_set:
        st.info(f"No evaluation results available for {st.session_state.selected_eval_set}. Run an evaluation to generate results.")
    else:
        st.info("Please select an eval set from the sidebar to view results.")
