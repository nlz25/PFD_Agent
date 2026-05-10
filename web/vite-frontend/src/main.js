import { marked } from "marked";
import { Network, DataSet } from "vis-network/standalone";
import * as $3Dmol from "3dmol";
import "./style.css";

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

const APP_NAME = "MatCreator";

const state = {
  sessionId: `session-${Math.floor(Date.now() / 1000)}`,
  userId: localStorage.getItem("mat_userId") || "",
  sessionReady: false,
  artifacts: [],
  structures: [],
};

// ---------------------------------------------------------------------------
// DOM refs
// ---------------------------------------------------------------------------

const chatArea = document.getElementById("chat-area");
const textInput = document.getElementById("text-input");
const sendBtn = document.getElementById("send-btn");
const artifactList = document.getElementById("artifact-list");
const structureList = document.getElementById("structure-list");
const sessionIdEl = document.getElementById("session-id");
const sessionListEl = document.getElementById("session-list");
const resetBtn = document.getElementById("reset-session");
const refreshSessionsBtn = document.getElementById("refresh-sessions");
const structureViewer = document.getElementById("structure-viewer");
const svCanvas = document.getElementById("sv-canvas");
const svMeta = document.getElementById("sv-meta");
const svClose = document.getElementById("sv-close");
const graphStatusEl = document.getElementById("graph-status");
const loginModal = document.getElementById("login-modal");
const loginInput = document.getElementById("login-input");
const loginSubmit = document.getElementById("login-submit");
const userDisplay = document.getElementById("user-display");
const editUserBtn = document.getElementById("edit-user");

sessionIdEl.textContent = state.sessionId;
if (state.userId) userDisplay.textContent = state.userId;

// ---------------------------------------------------------------------------
// Agent Graph Visualization
// ---------------------------------------------------------------------------

const NODE_COLORS = {
  orchestrator: { bg: "#7C3AED", border: "#6D28D9", font: "#fff" },
  planning:     { bg: "#3B82F6", border: "#2563EB", font: "#fff" },
  execution:    { bg: "#10B981", border: "#059669", font: "#fff" },
  tester:       { bg: "#F59E0B", border: "#D97706", font: "#1a1a1a" },
  step:         { bg: "#374151", border: "#4B5563", font: "#e5e7eb" },
};

const STATUS_COLORS = {
  running:          { bg: "#FBBF24", border: "#F59E0B", font: "#1a1a1a" },
  success:          null,
  failed:           { bg: "#EF4444", border: "#DC2626", font: "#fff" },
  needs_replanning: { bg: "#F97316", border: "#EA580C", font: "#fff" },
  idle:             { bg: "#374151", border: "#4B5563", font: "#9CA3AF" },
};

class AgentGraphView {
  constructor(containerId) {
    this._container = document.getElementById(containerId);
    this._nodes = new DataSet([]);
    this._edges = new DataSet([]);
    this._network = null;
    this._pollInterval = null;
    this._detailEl = document.getElementById("graph-detail");
    this._detailLabel = document.getElementById("detail-label");
    this._detailStatus = document.getElementById("detail-status");
    this._detailSummary = document.getElementById("detail-summary");
    this._detailArtifacts = document.getElementById("detail-artifacts");
    this._detailTiming = document.getElementById("detail-timing");
    this._detailInput = document.getElementById("detail-input");
    this._detailToolcalls = document.getElementById("detail-toolcalls");
    this._detailToolcallsCount = document.getElementById("detail-toolcalls-count");
    this._detailConversation = document.getElementById("detail-conversation");
    this._nodeData = {};
    this._init();
  }

  _init() {
    const options = {
      layout: {
        hierarchical: {
          direction: "UD",
          sortMethod: "directed",
          nodeSpacing: 120,
          levelSeparation: 90,
          blockShifting: true,
          edgeMinimization: true,
        },
      },
      physics: { enabled: false },
      edges: {
        arrows: { to: { enabled: true, scaleFactor: 0.6 } },
        color: { color: "#4B5563", highlight: "#9CA3AF" },
        width: 1.5,
        smooth: { type: "cubicBezier", forceDirection: "vertical" },
      },
      nodes: {
        shape: "box",
        borderWidth: 2,
        borderWidthSelected: 3,
        font: { size: 13, face: "Manrope, sans-serif" },
        margin: { top: 8, bottom: 8, left: 12, right: 12 },
      },
      interaction: {
        hover: true,
        tooltipDelay: 200,
        dragNodes: true,
        zoomView: true,
      },
    };

    this._network = new Network(
      this._container,
      { nodes: this._nodes, edges: this._edges },
      options
    );

    this._network.on("selectNode", (params) => {
      if (params.nodes.length) this._showDetail(params.nodes[0]);
    });
    this._network.on("deselectNode", () => this._hideDetail());
  }

  _visNode(raw) {
    const typeColors = NODE_COLORS[raw.type] || NODE_COLORS.step;
    const statusOverride = STATUS_COLORS[raw.status];
    const colors = statusOverride || typeColors;
    const isRunning = raw.status === "running";
    return {
      id: raw.id,
      label: raw.label,
      color: {
        background: colors.bg,
        border: colors.border,
        highlight: { background: colors.border, border: colors.border },
      },
      font: { color: colors.font },
      shapeProperties: isRunning ? { borderDashes: [4, 3] } : {},
      borderWidth: isRunning ? 2.5 : 2,
      title: raw.summary || raw.label,
    };
  }

  _computeLevels(rawNodes, edges) {
    const nodeMap = Object.fromEntries(rawNodes.map(n => [n.id, n]));
    const children = {};
    const hasParent = new Set();
    edges.forEach(e => {
      (children[e.from] = children[e.from] || []).push(e.to);
      hasParent.add(e.to);
    });
    const roots = rawNodes.map(n => n.id).filter(id => !hasParent.has(id));
    const levels = {};

    // Recursively assign levels. Among siblings, sort by start_time and
    // increment the level each time a child starts after the previous group ends
    // (sequential). Children whose time windows overlap stay at the same level
    // (parallel).
    function assign(id, minLevel) {
      if (levels[id] !== undefined && levels[id] >= minLevel) return;
      levels[id] = minLevel;

      const kids = (children[id] || []).slice().sort((a, b) => {
        const ta = nodeMap[a]?.start_time ? new Date(nodeMap[a].start_time).getTime() : Infinity;
        const tb = nodeMap[b]?.start_time ? new Date(nodeMap[b].start_time).getTime() : Infinity;
        return ta - tb;
      });

      let nextLevel = minLevel + 1;
      let groupEndTime = null; // latest end_time seen in the current parallel group

      for (const kid of kids) {
        const kidStart = nodeMap[kid]?.start_time ? new Date(nodeMap[kid].start_time).getTime() : null;
        const kidEnd   = nodeMap[kid]?.end_time   ? new Date(nodeMap[kid].end_time).getTime()   : null;

        if (groupEndTime !== null && kidStart !== null && kidStart >= groupEndTime) {
          // This child starts after the previous group ended — sequential, new level
          nextLevel++;
          groupEndTime = kidEnd;
        } else {
          // Concurrent with previous group (or no timing info) — extend group window
          if (groupEndTime === null || (kidEnd !== null && kidEnd > groupEndTime)) {
            groupEndTime = kidEnd;
          }
        }

        assign(kid, nextLevel);
      }
    }

    roots.forEach(r => assign(r, 0));
    return levels;
  }

  update(graphData) {
    if (!graphData || typeof graphData.nodes !== "object") return;

    const rawNodes = Object.values(graphData.nodes);
    this._nodeData = graphData.nodes;
    const levels = this._computeLevels(rawNodes, graphData.edges || []);

    rawNodes.forEach((raw) => {
      const vis = this._visNode(raw);
      vis.level = levels[raw.id] ?? 0;
      if (this._nodes.get(raw.id)) {
        this._nodes.update(vis);
      } else {
        this._nodes.add(vis);
      }
    });

    const existingEdgeIds = new Set(this._edges.getIds());
    (graphData.edges || []).forEach((e) => {
      const edgeId = `${e.from}__${e.to}`;
      if (!existingEdgeIds.has(edgeId)) {
        this._edges.add({ id: edgeId, from: e.from, to: e.to });
      }
    });

    if (rawNodes.length > 0) {
      this._network.fit({ animation: { duration: 300, easingFunction: "easeInOutQuad" } });
    }
  }

  startPolling(sessionId) {
    this._setStatus("polling");
    this._poll(sessionId);
    this._pollInterval = setInterval(() => this._poll(sessionId), 2000);
  }

  stopPolling() {
    if (this._pollInterval) {
      clearInterval(this._pollInterval);
      this._pollInterval = null;
    }
    this._setStatus("idle");
  }

  async _poll(sessionId) {
    try {
      const resp = await fetch(`/api/agent-graph/${sessionId}`);
      if (!resp.ok) return;
      const data = await resp.json();
      this.update(data);
    } catch (_) {
      // silently ignore network errors during polling
    }
  }

  reset() {
    this._nodes.clear();
    this._edges.clear();
    this._nodeData = {};
    this._hideDetail();
    this._setStatus("idle");
    this.stopPolling();
  }

  _setStatus(s) {
    if (graphStatusEl) {
      graphStatusEl.textContent = s;
      graphStatusEl.className = `graph-status status-${s}`;
    }
  }

  _showDetail(nodeId) {
    const raw = this._nodeData[nodeId];
    if (!raw) return;
    this._detailLabel.textContent = raw.label;
    this._detailStatus.textContent = raw.status;
    this._detailStatus.className = `badge badge-${raw.status}`;
    this._detailSummary.textContent = raw.summary || "—";

    // Timing
    if (raw.start_time) {
      const start = new Date(raw.start_time);
      if (raw.end_time) {
        const end = new Date(raw.end_time);
        const secs = ((end - start) / 1000).toFixed(1);
        this._detailTiming.textContent = `${secs}s`;
      } else {
        this._detailTiming.textContent = "running…";
      }
    } else {
      this._detailTiming.textContent = "—";
    }

    // Artifacts
    this._detailArtifacts.innerHTML = "";
    const arts = raw.artifacts || [];
    if (arts.length) {
      arts.forEach((a) => {
        const li = document.createElement("li");
        li.textContent = a.split("/").pop();
        li.title = a;
        this._detailArtifacts.appendChild(li);
      });
    } else {
      const li = document.createElement("li");
      li.textContent = "none";
      this._detailArtifacts.appendChild(li);
    }

    // Input parameters
    if (raw.input && Object.keys(raw.input).length) {
      this._detailInput.textContent = JSON.stringify(raw.input, null, 2);
      document.getElementById("detail-input-row").style.display = "";
    } else {
      document.getElementById("detail-input-row").style.display = "none";
    }

    // Tool calls
    const toolCalls = raw.tool_calls || [];
    this._detailToolcallsCount.textContent = toolCalls.length;
    this._detailToolcalls.innerHTML = "";
    if (toolCalls.length) {
      toolCalls.forEach((tc) => {
        const d = document.createElement("details");
        d.className = "timeline-function-call";
        const dur = tc.start_time && tc.end_time
          ? ` (${((new Date(tc.end_time) - new Date(tc.start_time)) / 1000).toFixed(1)}s)`
          : "";
        const s = document.createElement("summary");
        s.textContent = `🔧 ${tc.name}${dur}`;
        d.appendChild(s);
        if (tc.args_summary) {
          const pre = document.createElement("pre");
          pre.className = "json-block";
          pre.textContent = tc.args_summary;
          d.appendChild(pre);
        }
        if (tc.result_summary) {
          const pre = document.createElement("pre");
          pre.className = "json-block";
          pre.style.borderTop = "1px solid rgba(255,255,255,0.06)";
          pre.textContent = `→ ${tc.result_summary}`;
          d.appendChild(pre);
        }
        this._detailToolcalls.appendChild(d);
      });
      document.getElementById("detail-toolcalls-row").style.display = "";
    } else {
      document.getElementById("detail-toolcalls-row").style.display = "none";
    }

    // Conversation transcript
    const conversation = raw.conversation || [];
    this._detailConversation.innerHTML = "";
    if (conversation.length) {
      conversation.forEach((evt) => {
        const d = document.createElement("details");
        d.className = `timeline-${evt.type}`;
        const s = document.createElement("summary");
        const icon = evt.type === "thought" ? "💭" : evt.type === "text" ? "💬" : evt.type === "function_call" ? "🔧" : "↩";
        s.textContent = `${icon} [${evt.author}] ${evt.type}`;
        d.appendChild(s);
        const pre = document.createElement("pre");
        pre.className = "json-block";
        pre.textContent = evt.content;
        d.appendChild(pre);
        this._detailConversation.appendChild(d);
      });
      document.getElementById("detail-conversation-row").style.display = "";
    } else {
      document.getElementById("detail-conversation-row").style.display = "none";
    }

    this._detailEl.classList.remove("hidden");
  }

  _hideDetail() {
    this._detailEl.classList.add("hidden");
  }
}

const agentGraph = new AgentGraphView("agent-graph");

// ---------------------------------------------------------------------------
// Login / username management
// ---------------------------------------------------------------------------

function showLoginModal() {
  loginModal.classList.remove("hidden");
  loginInput.value = state.userId;
  loginInput.focus();
}

function hideLoginModal() {
  loginModal.classList.add("hidden");
}

function applyUsername(name) {
  state.userId = name;
  localStorage.setItem("mat_userId", name);
  userDisplay.textContent = name;
  hideLoginModal();
  loadSessions();
}

loginSubmit.addEventListener("click", () => {
  const name = loginInput.value.trim();
  if (name) applyUsername(name);
});

loginInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") {
    const name = loginInput.value.trim();
    if (name) applyUsername(name);
  }
});

editUserBtn.addEventListener("click", () => showLoginModal());

// Show login modal on load if no saved username
if (!state.userId) {
  showLoginModal();
} else {
  loadSessions();
}

// ---------------------------------------------------------------------------
// Session list management
// ---------------------------------------------------------------------------

async function loadSessions() {
  if (!state.userId) return;
  try {
    const resp = await fetch(`/apps/${APP_NAME}/users/${state.userId}/sessions`);
    if (!resp.ok) return;
    const sessions = await resp.json();
    renderSessionList(sessions);
  } catch (_) {
    // silently ignore — server may not be running yet
  }
}

function renderSessionList(sessions) {
  sessionListEl.innerHTML = "";
  if (!Array.isArray(sessions) || !sessions.length) {
    sessionListEl.innerHTML = '<li class="empty">No sessions yet</li>';
    return;
  }
  sessions
    .slice()
    .sort((a, b) => (b.lastUpdateTime || 0) - (a.lastUpdateTime || 0))
    .forEach((s) => {
      const li = document.createElement("li");
      li.className = "session-item" + (s.id === state.sessionId ? " active" : "");
      li.textContent = s.id;
      li.title = s.id;
      li.addEventListener("click", () => switchSession(s.id));
      sessionListEl.appendChild(li);
    });
}

async function switchSession(sessionId) {
  state.sessionId = sessionId;
  state.sessionReady = true;
  sessionIdEl.textContent = sessionId;
  state.artifacts = [];
  state.structures = [];
  setArtifacts([], artifactList, "No artifacts yet");
  setArtifacts([], structureList, "No structures yet");
  agentGraph.reset();
  await loadSession(sessionId);
  await loadSessions();
  agentGraph.startPolling(sessionId);
}

refreshSessionsBtn.addEventListener("click", (e) => { e.stopPropagation(); loadSessions(); });

// ---------------------------------------------------------------------------
// File path → API URL conversion
// ---------------------------------------------------------------------------

function pathToApiUrl(path) {
  return `/api/workspace/files?path=${encodeURIComponent(path)}`;
}

// ---------------------------------------------------------------------------
// Chat helpers
// ---------------------------------------------------------------------------

function scrollToBottom() {
  chatArea.scrollTop = chatArea.scrollHeight;
}

function addMessage(role, content) {
  const div = document.createElement("div");
  div.className = `message ${role}-message`;
  const inner = document.createElement("div");
  inner.className = "markdown-content";
  inner.innerHTML = marked.parse(content || "");
  div.appendChild(inner);
  chatArea.appendChild(div);
  scrollToBottom();
  return div;
}

// Render a typed timeline array into a container element, mirroring
// Streamlit's render_stream_timeline: thoughts and tool calls go into
// collapsible <details> blocks; text parts render as markdown;
// plot_path responses render as inline images.
function renderTimeline(container, timeline) {
  container.innerHTML = "";
  for (const item of timeline) {
    if (item.type === "thought") {
      const details = document.createElement("details");
      details.className = "timeline-thought";
      const summary = document.createElement("summary");
      summary.textContent = "🤔 Thinking...";
      details.appendChild(summary);
      const body = document.createElement("div");
      body.className = "markdown-content";
      body.innerHTML = marked.parse(item.text || "");
      details.appendChild(body);
      container.appendChild(details);
    } else if (item.type === "function_call") {
      const details = document.createElement("details");
      details.className = "timeline-function-call";
      const summary = document.createElement("summary");
      summary.textContent = `🔧 ${item.name}`;
      details.appendChild(summary);
      const pre = document.createElement("pre");
      pre.className = "json-block";
      pre.textContent = JSON.stringify(item.args, null, 2);
      details.appendChild(pre);
      container.appendChild(details);
    } else if (item.type === "function_response") {
      const details = document.createElement("details");
      details.className = "timeline-function-response";
      const summary = document.createElement("summary");
      summary.textContent = `📥 ${item.name}`;
      details.appendChild(summary);
      const pre = document.createElement("pre");
      pre.className = "json-block";
      pre.textContent = JSON.stringify(item.response, null, 2);
      details.appendChild(pre);
      container.appendChild(details);
      // Render inline image if this response produced a plot
      if (item.response && item.response.plot_path) {
        const img = document.createElement("img");
        img.src = pathToApiUrl(item.response.plot_path);
        img.className = "timeline-image";
        img.alt = item.response.plot_path.split("/").pop();
        container.appendChild(img);
      }
      // Render inline "View Structure" button for structure tool responses
      if (item.response && item.response.structure_path) {
        const btn = document.createElement("button");
        btn.className = "ghost structure-view-btn";
        btn.textContent = `🔬 View: ${item.response.structure_path.split("/").pop()}`;
        btn.addEventListener("click", () => openViewer({
          path: item.response.structure_path,
          url: pathToApiUrl(item.response.structure_path),
        }));
        container.appendChild(btn);
      }
    } else if (item.type === "text") {
      const div = document.createElement("div");
      div.className = "markdown-content";
      div.innerHTML = marked.parse(item.text || "");
      container.appendChild(div);
    }
  }
  scrollToBottom();
}

// Create an agent message div with an inner timeline container, append to
// chatArea, and return the inner container for live updates.
function addAgentTimelineMessage(timeline) {
  const outer = document.createElement("div");
  outer.className = "message agent-message";
  const inner = document.createElement("div");
  inner.className = "timeline-container";
  outer.appendChild(inner);
  chatArea.appendChild(outer);
  renderTimeline(inner, timeline);
  return inner;
}

function setArtifacts(list, target, emptyText) {
  target.innerHTML = "";
  if (!list.length) {
    const li = document.createElement("li");
    li.className = "empty";
    li.textContent = emptyText;
    target.appendChild(li);
    return;
  }
  list.forEach((item) => {
    const li = document.createElement("li");
    if (item.path) {
      // Structure file — always open in 3D viewer, never as a raw link
      li.textContent = item.name || item.path.split("/").pop();
      li.classList.add("structure-item");
      li.addEventListener("click", () => openViewer(item));
    } else if (item.url) {
      if (item.isImage) {
        // Show thumbnail + filename link for plot images
        const wrap = document.createElement("div");
        wrap.className = "artifact-image-wrap";
        const img = document.createElement("img");
        img.src = item.url;
        img.className = "artifact-thumb";
        img.alt = item.name;
        const a = document.createElement("a");
        a.href = item.url;
        a.textContent = item.name;
        a.target = "_blank";
        wrap.appendChild(img);
        wrap.appendChild(a);
        li.appendChild(wrap);
      } else {
        const a = document.createElement("a");
        a.href = item.url;
        a.textContent = item.name || item.url;
        a.target = "_blank";
        li.appendChild(a);
      }
    }
    target.appendChild(li);
  });
}

// Classify a file path as "structure", "image", or "artifact" by extension/name.
const STRUCTURE_EXTS = new Set([".cif", ".xyz", ".extxyz", ".vasp"]);
const STRUCTURE_NAMES = new Set(["poscar", "contcar"]);
const IMAGE_EXTS = new Set([".png", ".jpg", ".jpeg", ".gif", ".svg"]);

function classifyPath(p) {
  const name = p.split("/").pop();
  const dotIdx = name.lastIndexOf(".");
  const ext = dotIdx >= 0 ? name.slice(dotIdx).toLowerCase() : "";
  if (STRUCTURE_EXTS.has(ext) || STRUCTURE_NAMES.has(name.toLowerCase())) return "structure";
  if (IMAGE_EXTS.has(ext)) return "image";
  return "artifact";
}

// Extract artifacts/structures from a functionResponse.response object,
// converting filesystem paths to /api/workspace/files URLs.
function harvestFromResponse(response) {
  if (!response || typeof response !== "object") return;
  if (response.plot_path) {
    state.artifacts.push({
      name: response.plot_path.split("/").pop(),
      url: pathToApiUrl(response.plot_path),
      isImage: true,
    });
  }
  if (response.artifact_path) {
    state.artifacts.push({
      name: response.artifact_path.split("/").pop(),
      url: pathToApiUrl(response.artifact_path),
    });
  }
  if (response.structure_path) {
    state.structures.push({
      name: response.structure_path.split("/").pop(),
      path: response.structure_path,
      url: pathToApiUrl(response.structure_path),
    });
  }
  // Handle artifacts array from submit_step_result — classify each path by extension
  const list = response.artifacts || response.artifact_paths || [];
  for (const item of list) {
    const p = typeof item === "string" ? item : (item.path || item.url || "");
    if (!p) continue;
    const name = p.split("/").pop();
    const kind = classifyPath(p);
    if (kind === "structure") {
      state.structures.push({ name, path: p, url: pathToApiUrl(p) });
    } else if (kind === "image") {
      state.artifacts.push({ name, url: pathToApiUrl(p), isImage: true });
    } else {
      state.artifacts.push({ name, url: pathToApiUrl(p) });
    }
  }
  setArtifacts(state.artifacts, artifactList, "No artifacts yet");
  setArtifacts(state.structures, structureList, "No structures yet");
}

// Harvest top-level artifact/structure arrays from an SSE event envelope,
// and also check functionResponse.response for file paths.
function harvest(evt) {
  const artifacts = evt.artifacts || evt.artifact_paths || [];
  artifacts.forEach((art) => {
    if (typeof art === "string") {
      state.artifacts.push({ name: art.split("/").pop(), url: pathToApiUrl(art) });
    } else if (art && typeof art === "object") {
      const rawUrl = art.url || art.path;
      state.artifacts.push({
        name: art.name || art.path || "artifact",
        url: rawUrl ? pathToApiUrl(rawUrl) : rawUrl,
      });
    }
  });
  const structures = evt.structures || evt.structure_paths || evt.trajectory_files || [];
  structures.forEach((s) => {
    if (typeof s === "string") {
      state.structures.push({ name: s.split("/").pop(), path: s, url: pathToApiUrl(s) });
    } else if (s && typeof s === "object") {
      const rawPath = s.path || s.url || "";
      state.structures.push({
        name: s.name || s.path || "structure",
        path: rawPath,
        url: rawPath ? pathToApiUrl(rawPath) : "",
      });
    }
  });
  // Extract from every functionResponse.response inside content.parts
  const parts = evt?.content?.parts || [];
  for (const p of parts) {
    if (p.functionResponse?.response) {
      harvestFromResponse(p.functionResponse.response);
    }
  }
  setArtifacts(state.artifacts, artifactList, "No artifacts yet");
  setArtifacts(state.structures, structureList, "No structures yet");
}

// ---------------------------------------------------------------------------
// Session management
// ---------------------------------------------------------------------------

async function createSession() {
  const url = `/apps/${APP_NAME}/users/${state.userId}/sessions/${state.sessionId}`;
  try {
    const resp = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({}),
    });
    if (!resp.ok) {
      console.error(`Failed to create session: HTTP ${resp.status}`, await resp.text());
      return;
    }
    state.sessionReady = true;
    await loadSessions();
  } catch (err) {
    console.error("Failed to create session:", err);
  }
}

// Reload full session history from the ADK server and re-render the chat,
// mirroring Streamlit's load_session() called after send_message_sse().
async function loadSession(sessionId) {
  try {
    const resp = await fetch(
      `/apps/${APP_NAME}/users/${state.userId}/sessions/${sessionId}`,
      { headers: { "Content-Type": "application/json" } }
    );
    if (!resp.ok) return;
    const sessionData = await resp.json();
    const events = sessionData.events || [];

    // Rebuild chat from server-canonical state
    chatArea.innerHTML = "";
    state.artifacts = [];
    state.structures = [];

    // First pass: collect all functionResponses keyed by ID for cross-event matching
    const frById = {};
    for (const event of events) {
      for (const p of (event.content?.parts || [])) {
        if (p.functionResponse?.id) {
          frById[p.functionResponse.id] = p.functionResponse;
        }
      }
    }

    for (const event of events) {
      const role = event.author === "user" ? "user" : "agent";
      const parts = event.content?.parts || [];

      if (role === "user") {
        const text = parts.map((p) => p.text || "").join("");
        if (text) addMessage("user", text);
        continue;
      }

      const timeline = [];
      let accText = "";

      for (const p of parts) {
        if (p.thought) {
          timeline.push({ type: "thought", text: p.text || "" });
        } else if (p.functionCall) {
          const fc = p.functionCall;
          const matchedFr = frById[fc.id];
          timeline.push({
            type: "function_call",
            id: fc.id,
            name: fc.name || "Unknown",
            args: fc.args || {},
          });
          if (matchedFr) {
            timeline.push({
              type: "function_response",
              id: matchedFr.id,
              name: matchedFr.name || "Unknown",
              response: matchedFr.response || {},
            });
            harvestFromResponse(matchedFr.response);
          }
        } else if (p.functionResponse) {
          const fr = p.functionResponse;
          const alreadyMatched = timeline.some(
            (t) => t.type === "function_response" && t.id === fr.id
          );
          if (!alreadyMatched) {
            timeline.push({
              type: "function_response",
              id: fr.id,
              name: fr.name || "Unknown",
              response: fr.response || {},
            });
            harvestFromResponse(fr.response);
          }
        } else if (p.text && !p.thought) {
          accText += p.text;
          const last = timeline[timeline.length - 1];
          if (last?.type === "text") {
            last.text = accText;
          } else {
            timeline.push({ type: "text", text: accText });
          }
        }
      }

      if (timeline.length > 0) {
        addAgentTimelineMessage(timeline);
      }
    }

    setArtifacts(state.artifacts, artifactList, "No artifacts yet");
    setArtifacts(state.structures, structureList, "No structures yet");
  } catch (err) {
    console.error("Failed to load session:", err);
  }
}

// ---------------------------------------------------------------------------
// Message sending + SSE streaming
// ---------------------------------------------------------------------------

async function sendMessage(message) {
  if (!message.trim()) return;
  if (!state.userId) { showLoginModal(); return; }

  addMessage("user", message);
  textInput.value = "";

  if (!state.sessionReady) await createSession();

  agentGraph.startPolling(state.sessionId);

  const controller = new AbortController();
  const payload = {
    app_name: APP_NAME,
    user_id: state.userId,
    session_id: state.sessionId,
    new_message: {
      role: "user",
      parts: [{ text: message }],
    },
  };

  const timeline = [];
  let timelineContainer = null;
  let accText = "";

  try {
    const resp = await fetch("/run_sse", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
      },
      body: JSON.stringify(payload),
      signal: controller.signal,
    });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      const chunk = decoder.decode(value);
      const lines = chunk.split("\n");
      for (const line of lines) {
        if (!line.startsWith("data: ")) continue;
        const dataStr = line.slice(6);
        if (dataStr === "[DONE]") continue;
        try {
          const evt = JSON.parse(dataStr);
          harvest(evt);
          const parts = evt?.content?.parts || [];
          for (const p of parts) {
            if (p.thought) {
              timeline.push({ type: "thought", text: p.text || "" });
            } else if (p.functionCall) {
              const fc = p.functionCall;
              timeline.push({
                type: "function_call",
                id: fc.id,
                name: fc.name || "Unknown",
                args: fc.args || {},
              });
            } else if (p.functionResponse) {
              const fr = p.functionResponse;
              timeline.push({
                type: "function_response",
                id: fr.id,
                name: fr.name || "Unknown",
                response: fr.response || {},
              });
            } else if (p.text) {
              accText += p.text;
              const last = timeline[timeline.length - 1];
              if (last?.type === "text") {
                last.text = accText;
              } else {
                timeline.push({ type: "text", text: accText });
              }
            }

            if (timeline.length > 0 && !timelineContainer) {
              timelineContainer = addAgentTimelineMessage(timeline);
            } else if (timelineContainer) {
              renderTimeline(timelineContainer, timeline);
            }
          }
        } catch (_) {
          // ignore malformed lines
        }
      }
    }
  } catch (err) {
    addMessage("agent", `Backend error: ${err}`);
  } finally {
    await agentGraph._poll(state.sessionId);
    agentGraph.stopPolling();
    await loadSession(state.sessionId);
  }
}

// ---------------------------------------------------------------------------
// Structure viewer
// ---------------------------------------------------------------------------

async function openViewer(item) {
  structureViewer.classList.remove("hidden");
  svCanvas.innerHTML = '<div style="color:var(--muted);padding:16px;font-size:13px">Loading…</div>';
  svMeta.textContent = "";

  try {
    const resp = await fetch(`/api/structure/view?path=${encodeURIComponent(item.path)}`);
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const data = await resp.json();

    svCanvas.innerHTML = "";

    const viewer = $3Dmol.createViewer(svCanvas, { backgroundColor: "0x06080f" });
    viewer.addModel(data.xyz, "xyz");
    viewer.setStyle({}, { sphere: { scale: 0.3 }, stick: { radius: 0.15 } });

    if (data.periodic && data.cell) {
      const [a, b, c] = data.cell;
      const add = (u, v) => [u[0] + v[0], u[1] + v[1], u[2] + v[2]];
      const corners = [
        [0, 0, 0], a, b, c,
        add(a, b), add(a, c), add(b, c), add(add(a, b), c),
      ];
      const edges = [[0,1],[0,2],[0,3],[1,4],[1,5],[2,4],[2,6],[3,5],[3,6],[4,7],[5,7],[6,7]];
      for (const [i, j] of edges) {
        viewer.addLine({
          start: { x: corners[i][0], y: corners[i][1], z: corners[i][2] },
          end:   { x: corners[j][0], y: corners[j][1], z: corners[j][2] },
          color: "0x7dd3fc",
          linewidth: 2,
        });
      }
    }

    viewer.zoomTo();
    viewer.render();

    svMeta.textContent =
      `${data.formula}  ·  ${data.n_atoms} atoms${data.periodic ? "  ·  periodic" : ""}`;
  } catch (err) {
    svCanvas.innerHTML =
      `<div style="color:#f87171;padding:16px;font-size:13px">Failed to load structure: ${err}</div>`;
  }
}

svClose.addEventListener("click", () => {
  structureViewer.classList.add("hidden");
  svCanvas.innerHTML = "";
});

// ---------------------------------------------------------------------------
// Event listeners
// ---------------------------------------------------------------------------

sendBtn.addEventListener("click", () => sendMessage(textInput.value));
textInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendMessage(textInput.value);
  }
});

Array.from(document.querySelectorAll("[data-quick]"))
  .forEach((btn) => btn.addEventListener("click", () => sendMessage(btn.dataset.quick || "")));

resetBtn.addEventListener("click", () => {
  state.sessionId = `session-${Math.floor(Date.now() / 1000)}`;
  state.sessionReady = false;
  sessionIdEl.textContent = state.sessionId;
  chatArea.innerHTML = "";
  state.artifacts = [];
  state.structures = [];
  setArtifacts([], artifactList, "No artifacts yet");
  setArtifacts([], structureList, "No structures yet");
  agentGraph.reset();
  loadSessions();
});
