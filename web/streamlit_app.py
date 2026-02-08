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
import streamlit.components.v1 as components
from ase.io import read
import py3Dmol

# Set page config
st.set_page_config(
    page_title="FFPilot",
    page_icon="🔊",
    layout="centered"
)

# Constants
API_BASE_URL = "http://localhost:8000"
APP_NAME = "MatCreator"

# Initialize session state variables
if "user_id" not in st.session_state:
    st.session_state.user_id = f"user-{uuid.uuid4()}"
    
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

def visualize_structure(structure_path, height=400, width=600):
    """Visualize atomic structure using ASE and py3Dmol."""
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
            sessions = response.json()
            # Sort by session_id (timestamp-based) descending
            st.session_state.available_sessions = sorted(
                sessions, 
                key=lambda s: s.get('id', ''), 
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
            print("response:", response.json())
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
            st.session_state.eval_results = result.get("eval_result_ids", [])
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
                print('event_content:', content)
                print('event_author:', author)
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
                model_path = None
                text = ""
                
                
                # Process each part
                part = parts[0]
                
                
                # Extract text content
                if "text" in part:
                    text = part.get("text", "")
                    
                    # Extract paths from functionResponse (same logic as send_message_sse)
                if "functionResponse" in part:
                    fr = part["functionResponse"]
                    resp = fr.get("response", {})
                        
                    # Check for plot path
                    p_path = resp.get("plot_path")
                    if p_path:
                        plot_path = p_path
                        artifacts.append({"name": os.path.basename(p_path), "url": p_path})
                        
                    # Check for structure/model paths in content
                    contents = resp.get("content", [])
                    for c in contents:
                        if c.get("type") == "text" and "text" in c:
                            try:
                                payload = json.loads(c["text"])
                                    # Get structure path
                                struct_path = payload.get("structure_path") or payload.get("path")
                                if struct_path:
                                    structure_path = struct_path
                                    artifacts.append({"name": os.path.basename(struct_path), "url": struct_path})
                                    
                                    # Get model path
                                model_p = payload.get("model")
                                if isinstance(model_p, list) and len(model_p) > 0:
                                    model_path = model_p[0]
                                    artifacts.append({"name": os.path.basename(model_p[0]), "url": model_p[0]})
                            except json.JSONDecodeError:
                                continue
                
                if text:
                    msg_entry ["content"] = text
                # Add extracted paths to message entry
                if structure_path:
                    msg_entry["structure_path"] = structure_path
                    msg_entry["content"]="**Structure Visualization**"
                if plot_path:
                    msg_entry["plot_path"] = plot_path
                    msg_entry["content"]="**Plot**"
                    
                if model_path:
                    msg_entry["model_path"] = model_path
                messages.append(msg_entry)
            
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
    """Stream agent response incrementally via /run_sse."""
    if not st.session_state.session_id:
        st.error("No active session. Please create a session first.")
        return False

    # Backend only needs file paths; send as a list of strings
    attachments_payload = attachments if attachments else []

    # Store message with its attachments for display
    msg_data = {"role": "user", "content": message}
    if attachments_payload:
        msg_data["attachments"] = attachments_payload
    
    st.session_state.messages.append(msg_data)

    # Display user message immediately
    with st.chat_message("user"):
        st.write(message)
        if attachments_payload:
            for att_path in attachments_payload:
                st.caption(f"📎 {os.path.basename(att_path)}")

    # Build message text with attachment paths included
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
                    "parts": [{
                        "text": message_with_attachments
                        }],
                }
            }),
            stream=True,
        ) as response:
            if response.status_code != 200:
                st.error(f"Error: {response.text}")
                return False

            for line in response.iter_lines(decode_unicode=True, chunk_size=1):
                if not line or not line.startswith("data: "):
                    continue
                data_str = line[len("data: "):]
                if data_str == "[DONE]":
                    break
                try:
                    event = json.loads(data_str)
                    role = event.get("author", "agent")
                    part = event.get("content", {}).get("parts", [{}])[0]
                    
                    content = ""
                    structure_path = None
                    plot_path = None
                    model_path = None
                    # Extract text content
                    if "text" in part:
                        content = part["text"]
                    
                    # Extract function response content
                    if "functionResponse" in part:
                        fr = part["functionResponse"]
                        resp = fr.get("response", {})
                        
                        # Check for plot path
                        p_path = resp.get("plot_path")
                        if p_path:
                            st.session_state.artifacts.append({"name": os.path.basename(p_path), "url": p_path})
                            content += f"\n\nplot: [{os.path.basename(p_path)}]({p_path})\n\n"
                            plot_path = p_path
                        
                            
                        # Check for structure path in content
                        contents = resp.get("content", [])
                        for c in contents:
                            if c.get("type") == "text" and "text" in c:
                                try:
                                    payload = json.loads(c["text"])
                                    # get structure path
                                    struct_path = payload.get("structure_path") or payload.get("path")
                                    if struct_path:
                                        st.session_state.artifacts.append({"name": os.path.basename(struct_path), "url": struct_path})
                                        content += f"\n\nstructure path: [{os.path.basename(struct_path)}]({struct_path})\n\n"
                                        structure_path = struct_path
                                    # get model path
                                    model_p = payload.get("model")
                                    if isinstance(model_p,list) and len(model_p) > 0:
                                        st.session_state.artifacts.append({"name": os.path.basename(model_p[0]), "url": model_p[0]})
                                        content += f"\n\nmodel: [{os.path.basename(model_p[0])}]({model_p[0]})\n\n"
                                        model_path = model_p[0]
                                        
                                except json.JSONDecodeError:
                                    continue
                    
                    # Display each event in its own chat message
                    if content:
                        with st.chat_message(role):
                            st.markdown(content)
                            
                            # Visualize structure if present
                            if structure_path and os.path.exists(structure_path):
                                st.divider()
                                st.markdown("**Structure Visualization:**")
                                visualize_structure(structure_path)
                            
                            # Display plot if present
                            if plot_path and os.path.exists(plot_path):
                                st.divider()
                                st.markdown("**Plot:**")
                                st.image(plot_path, 
                                         width='content',
                                         #use_container_width=True
                                         )
                            
                            # Display model if present
                            if model_path and os.path.exists(model_path):
                                st.divider()
                                st.markdown("**Model File:**")
                                st.info(f"📦 {os.path.basename(model_path)}")
                                # Add download button for model
                                with open(model_path, "rb") as f:
                                    st.download_button(
                                        label=f"⬇️ Download {os.path.basename(model_path)}",
                                        data=f.read(),
                                        file_name=os.path.basename(model_path),
                                        key=f"download_model_stream_{model_path}"
                                    )
                        
                        # Store message in session state
                        msg_to_store = {"role": role, "content": content}
                        if structure_path:
                            msg_to_store["structure_path"] = structure_path
                        if plot_path:
                            msg_to_store["plot_path"] = plot_path
                        if model_path:
                            msg_to_store["model_path"] = model_path
                        st.session_state.messages.append(msg_to_store)
                        
                except json.JSONDecodeError:
                    continue
            
            # Clear file uploader by incrementing key
            st.session_state.uploader_key += 1
            return True
    except requests.RequestException as exc:
        st.error(f"Streaming error: {exc}")
        return False

# UI Components
st.title("FFPilot")

# Sidebar for session management
with st.sidebar:
    st.header("Session Management")
    
    # New session button
    if st.button("➕ New Session", width='stretch'):
        if create_session():
            st.rerun()
    
    # Refresh sessions button
    if st.button("🔄 Refresh Sessions", width='stretch'):
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
    
    # Evaluation Management Section
    st.header("Evaluation Management")
    
    # Refresh eval sets button
    if st.button("🔄 Refresh Eval Sets", width='stretch'):
        list_eval_sets()
        list_eval_results()
        st.rerun()
    
    # Create new eval set
    with st.expander("➕ Create New Eval Set"):
        new_eval_name = st.text_input("Eval Set Name", key="new_eval_name")
        new_eval_desc = st.text_area("Description (optional)", key="new_eval_desc", height=80)
        if st.button("Create Eval Set", width='stretch'):
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
            key="eval_set_dropdown"
        )
        st.session_state.selected_eval_set = selected_eval
        
        # Add current session to selected eval set
        if st.session_state.session_id:
            if st.button("📝 Add Current Session", width='stretch'):
                if add_session_to_eval_set(selected_eval, st.session_state.session_id):
                    st.success(f"Added session to {selected_eval}")
                    st.rerun()
        
        # Run evaluation button for selected eval set
        if st.button("▶️ Run Evaluation", width='stretch', key="run_selected_eval"):
            with st.spinner(f"Running evaluation on {selected_eval}..."):
                result = run_evaluation(selected_eval)
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
                if st.button("🗑️ Delete Selected", width='stretch', key="delete_selected_cases"):
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
                print("details for case_id:", get_eval_case_details(selected_eval, case_id))
                # Create checkbox for selection
                if st.checkbox(f"📋 {case_id}", value=is_selected, key=f"case_checkbox_{idx}_{case_id}"):
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
                if os.path.exists(url):
                    try:
                        with open(url, "rb") as f:
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

# Main area - tabs for conversation and eval results
tab1, tab2 = st.tabs(["💬 Conversation", "📊 Eval Results"])

with tab1:
    # Chat interface
    st.subheader("Conversation")

    # Display messages
    for msg_idx, msg in enumerate(st.session_state.messages):
        # Skip messages that only have "role" key
        if len(msg) == 1 and "role" in msg:
            continue
        
        if msg["role"] == "user":
            with st.chat_message("user"):
                if "content" in msg:
                    st.write(msg["content"])
                # Show attachments if present
                if "attachments" in msg and msg["attachments"]:
                    for att_path in msg["attachments"]:
                        st.caption(f"📎 {os.path.basename(att_path)}")
        else:
            with st.chat_message("agent"):
                if "content" in msg:
                    st.write(msg["content"])
                
                # Show structure visualization if present
                if "structure_path" in msg and os.path.exists(msg["structure_path"]):
                    #st.divider()
                    #st.markdown("**Structure Visualization:**")
                    visualize_structure(msg["structure_path"])
                    # Add download button for structure
                    with open(msg["structure_path"], "rb") as f:
                        st.download_button(
                            label=f"⬇️ Download {os.path.basename(msg['structure_path'])}",
                            data=f.read(),
                            file_name=os.path.basename(msg["structure_path"]),
                            key=f"download_struct_{msg_idx}_{os.path.basename(msg['structure_path'])}"
                        )
                # Show plot if present
                if "plot_path" in msg and os.path.exists(msg["plot_path"]):
                    #st.divider()
                    #st.markdown("**Plot:**")
                    st.image(msg["plot_path"], 
                             width='content',
                             #use_container_width=True
                             )
                    # Add download button for plot
                    with open(msg["plot_path"], "rb") as f:
                        st.download_button(
                            label=f"⬇️ Download {os.path.basename(msg['plot_path'])}",
                            data=f.read(),
                            file_name=os.path.basename(msg["plot_path"]),
                            key=f"download_plot_{msg_idx}_{os.path.basename(msg['plot_path'])}"
                        )
                # Show model if present
                if "model_path" in msg and os.path.exists(msg["model_path"]):
                    st.divider()
                    st.markdown("**Model File:**")
                    st.info(f"📦 {os.path.basename(msg['model_path'])}")
                    # Add download button for model
                    with open(msg["model_path"], "rb") as f:
                        st.download_button(
                            label=f"⬇️ Download {os.path.basename(msg['model_path'])}",
                            data=f.read(),
                            file_name=os.path.basename(msg["model_path"]),
                            key=f"download_model_{msg_idx}_{os.path.basename(msg['model_path'])}"
                        )
                

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

with tab2:
    # Eval Results Display
    st.subheader("Evaluation Results")
    
    # Refresh eval results
    if st.button("🔄 Refresh Results"):
        list_eval_results()
        st.rerun()
    
    # Load results if not already loaded
    if not st.session_state.eval_results:
        list_eval_results()
    
    if st.session_state.eval_results:
        st.write(f"**Total Results:** {len(st.session_state.eval_results)}")
        
        # Display each eval result
        for idx, eval_result_id in enumerate(st.session_state.eval_results):
            with st.expander(f"📊 {eval_result_id}", expanded=(idx == 0)):
                result_data = get_eval_result(eval_result_id)
                
                if result_data:
                    # Display metadata
                    st.markdown(f"**Eval Result ID:** `{result_data.get('id', 'N/A')}`")
                    st.markdown(f"**Eval Set ID:** `{result_data.get('eval_set_id', 'N/A')}`")
                    
                    # Display case results
                    case_results = result_data.get('case_results', [])
                    if case_results:
                        st.markdown(f"**Case Results:** ({len(case_results)} cases)")
                        
                        for case_idx, case_result in enumerate(case_results):
                            with st.container():
                                st.markdown(f"**Case {case_idx + 1}:** `{case_result.get('eval_case_id', 'N/A')}`")
                                
                                # Display metrics
                                metric_results = case_result.get('metric_results', [])
                                if metric_results:
                                    cols = st.columns(len(metric_results))
                                    for col_idx, metric in enumerate(metric_results):
                                        with cols[col_idx]:
                                            metric_name = metric.get('metric_name', 'Unknown')
                                            metric_score = metric.get('score', 'N/A')
                                            st.metric(metric_name, f"{metric_score:.2f}" if isinstance(metric_score, (int, float)) else metric_score)
                                
                                st.divider()
                    else:
                        st.info("No case results available")
                else:
                    st.error("Failed to load eval result details")
    else:
        st.info("No evaluation results available. Run an evaluation to generate results.")