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
    page_title="MLCreator",
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
            print("Fetched sessions:", sessions)
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
            
            print("Session data:", session_data)
            # Set current session
            st.session_state.session_id = session_id
            
            # Load messages from events (not messages)
            messages = []
            artifacts = []
            
            for event in session_data.get('events', []):
                content = event.get('content', {})
                author = event.get('author', 'agent')
                
                # Determine role (user or agent)
                role = content.get('role') or ('user' if author == 'user' else 'agent')
                
                # Extract text from parts
                parts = content.get('parts', [])
                if not parts:
                    continue
                
                text = parts[0].get('text', '')
                if not text:
                    continue
                
                msg_entry = {"role": role, "content": text}
                
                # Try to extract structure/plot/model paths from content
                if 'structure path:' in text or 'plot:' in text or 'model:' in text:
                    import re
                    # Extract paths from markdown links
                    struct_match = re.search(r'structure path: \[.*?\]\((.*?)\)', text)
                    plot_match = re.search(r'plot: \[.*?\]\((.*?)\)', text)
                    model_match = re.search(r'model: \[.*?\]\((.*?)\)', text)
                    
                    if struct_match:
                        struct_path = struct_match.group(1)
                        msg_entry["structure_path"] = struct_path
                        artifacts.append({"name": os.path.basename(struct_path), "url": struct_path})
                    
                    if plot_match:
                        plot_path = plot_match.group(1)
                        msg_entry["plot_path"] = plot_path
                        artifacts.append({"name": os.path.basename(plot_path), "url": plot_path})
                    
                    if model_match:
                        model_path = model_match.group(1)
                        msg_entry["model_path"] = model_path
                        artifacts.append({"name": os.path.basename(model_path), "url": model_path})
                
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
                #print(f"Received line: {line}")
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
                    print(part)
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
                                    print(model_p)
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
                                st.image(plot_path, use_container_width=True)
                            
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
st.title("MatCreator Agent Chat")

# Sidebar for session management
with st.sidebar:
    st.header("Session Management")
    
    # New session button
    if st.button("➕ New Session", use_container_width=True):
        if create_session():
            st.rerun()
    
    # Refresh sessions button
    if st.button("🔄 Refresh Sessions", use_container_width=True):
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
            if st.button(label, key=f"session_btn_{idx}_{session_id}", use_container_width=True, disabled=is_current):
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
                if os.path.exists(url):
                    try:
                        with open(url, "rb") as f:
                            st.download_button(
                                label="⬇️",
                                data=f.read(),
                                file_name=name,
                                key=f"download_artifact_{idx}_{name}",
                                use_container_width=True
                            )
                    except Exception as e:
                        st.caption("❌")
    else:
        st.caption("No artifacts yet")

# Chat interface
st.subheader("Conversation")

# Display messages
for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.write(msg["content"])
            # Show attachments if present
            if "attachments" in msg and msg["attachments"]:
                for att_path in msg["attachments"]:
                    st.caption(f"📎 {os.path.basename(att_path)}")
    else:
        with st.chat_message("agent"):
            st.write(msg["content"])
            # Show structure visualization if present
            if "structure_path" in msg and os.path.exists(msg["structure_path"]):
                st.divider()
                st.markdown("**Structure Visualization:**")
                visualize_structure(msg["structure_path"])
                # Add download button for structure
                with open(msg["structure_path"], "rb") as f:
                    st.download_button(
                        label=f"⬇️ Download {os.path.basename(msg['structure_path'])}",
                        data=f.read(),
                        file_name=os.path.basename(msg["structure_path"]),
                        key=f"download_struct_{msg['structure_path']}"
                    )
            # Show plot if present
            if "plot_path" in msg and os.path.exists(msg["plot_path"]):
                st.divider()
                st.markdown("**Plot:**")
                st.image(msg["plot_path"], use_container_width=True)
                # Add download button for plot
                with open(msg["plot_path"], "rb") as f:
                    st.download_button(
                        label=f"⬇️ Download {os.path.basename(msg['plot_path'])}",
                        data=f.read(),
                        file_name=os.path.basename(msg["plot_path"]),
                        key=f"download_plot_{msg['plot_path']}"
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
                        key=f"download_model_{msg['model_path']}"
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