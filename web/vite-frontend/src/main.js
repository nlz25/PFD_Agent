import { marked } from "marked";
import { Network, DataSet } from "vis-network/standalone";
import { Terminal } from "@xterm/xterm";
import { FitAddon } from "@xterm/addon-fit";
import "@xterm/xterm/css/xterm.css";
import * as $3Dmol from "3dmol";
import "./style.css";

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

const APP_NAME = "MatCreator";

const AGENT_MODE_KEY = "mat_agentMode";
const THEME_KEY = "mat_theme";

const state = {
  sessionId: localStorage.getItem("mat_sessionId") || `session-${Math.floor(Date.now() / 1000)}`,
  userId: localStorage.getItem("mat_userId") || "",
  displayName: localStorage.getItem("mat_displayName") || localStorage.getItem("mat_userId") || "",
  activeSessionUserId: localStorage.getItem("mat_userId") || "",
  isAdmin: false,
  deploymentMode: localStorage.getItem("mat_deploymentMode") || "local",
  sessionReady: false,
  structure3dViewer: null,
  activeCenterTabId: "chat",
  currentUploads: [],
  isSending: false,
  sendController: null,
  agentMode: localStorage.getItem(AGENT_MODE_KEY) || "normal",
  theme: localStorage.getItem(THEME_KEY) || "dark",
  customWorkdir: "",
  sessionSummaries: {},   // { sessionId: "summary text" }
  summaryGeneratedFor: new Set(),  // sessionIds that have triggered summary generation
};

// ---------------------------------------------------------------------------
// DOM refs
// ---------------------------------------------------------------------------

const chatArea = document.getElementById("chat-area");
const textInput = document.getElementById("text-input");
const inputContainer = document.querySelector(".input-container");
const sendBtn = document.getElementById("send-btn");
const fileUploadBtn = document.getElementById("file-upload-btn");
const fileUploadInput = document.getElementById("file-upload-input");
const uploadStatus = document.getElementById("upload-status");
const sessionIdEl = document.getElementById("session-id");
const sessionListEl = document.getElementById("session-list");
const resetBtn = document.getElementById("reset-session");
const workspaceCliToggle = document.getElementById("workspace-cli-toggle");
const themeToggle = document.getElementById("theme-toggle");
const refreshSessionsBtn = document.getElementById("refresh-sessions");
const graphViewport = document.getElementById("graph-viewport");
const graphDetail = document.getElementById("graph-detail");
const centerTabs = document.getElementById("center-tabs");
const centerTabsScroll = document.getElementById("center-tabs-scroll");
const centerTabPanels = document.getElementById("center-tab-panels");
const graphResizer = document.getElementById("graph-resizer");
const graphStatusEl = document.getElementById("graph-status");
const loginModal = document.getElementById("login-modal");
const loginInput = document.getElementById("login-input");
const loginPassword = document.getElementById("login-password");
const loginError = document.getElementById("login-error");
const loginUuidDisplay = document.getElementById("login-uuid-display");
const loginSubmit = document.getElementById("login-submit");
const loginView = document.getElementById("login-view");
const registerView = document.getElementById("register-view");
const regInput = document.getElementById("reg-input");
const regPassword = document.getElementById("reg-password");
const regConfirm = document.getElementById("reg-confirm");
const regError = document.getElementById("reg-error");
const regSubmit = document.getElementById("reg-submit");
const switchToRegister = document.getElementById("switch-to-register");
const switchToLogin = document.getElementById("switch-to-login");
const userDisplay = document.getElementById("user-display");
const editUserBtn = document.getElementById("edit-user");
const logoutBtn = document.getElementById("logout-btn");
const settingsLogoutBtn = document.getElementById("settings-logout-btn");
const benchToggle = null; // removed — replaced by mode-selector
const modeSelector = document.getElementById("mode-selector");
const graphColumn       = document.getElementById("graph-column");
const sidePanel         = document.getElementById("side-panel");
const fileExplorerCol   = document.getElementById("file-explorer-col");
const colResizerGraph   = document.getElementById("col-resizer-graph");
const colResizerSide    = document.getElementById("col-resizer-side");
const colResizerFiles   = document.getElementById("col-resizer-files");
const sessionSummaryHeader = document.getElementById("session-summary-header");
const sessionSummaryText = document.getElementById("session-summary-text");
const sessionSummaryEdit = document.getElementById("session-summary-edit");
const filesColToggleBtn = document.getElementById("files-col-toggle");
const knowledgeReviewBanner = document.getElementById("knowledge-review-banner");
const knowledgeReviewText = document.getElementById("knowledge-review-text");
const knowledgeReviewSpinner = document.getElementById("knowledge-review-spinner");
const workspaceCli = document.getElementById("workspace-cli");
const workspaceTerminalEl = document.getElementById("workspace-terminal");
let knowledgeReviewPoll = null;
let workspaceTerminal = null;
let workspaceTerminalFit = null;
let workspaceTerminalSocket = null;
const structureTabs = new Map();

function autoResizeTextInput() {
  if (!textInput) return;
  textInput.style.height = "auto";
  const computed = window.getComputedStyle(textInput);
  const lineHeight = parseFloat(computed.lineHeight) || 24;
  const maxHeight = lineHeight * 3;
  const nextHeight = Math.min(textInput.scrollHeight, maxHeight);
  textInput.style.height = `${nextHeight}px`;
  textInput.style.overflowY = textInput.scrollHeight > maxHeight ? "auto" : "hidden";
}

autoResizeTextInput();
textInput?.addEventListener("input", autoResizeTextInput);

function applyTheme(theme) {
  const nextTheme = theme === "light" ? "light" : "dark";
  state.theme = nextTheme;
  document.body.dataset.theme = nextTheme;
  themeToggle?.setAttribute("aria-pressed", String(nextTheme === "light"));
  themeToggle?.setAttribute("title", nextTheme === "light" ? "Toggle dark mode" : "Toggle light mode");
  themeToggle?.setAttribute("aria-label", nextTheme === "light" ? "Toggle dark mode" : "Toggle light mode");
}

applyTheme(state.theme);
themeToggle?.addEventListener("click", () => {
  const nextTheme = state.theme === "light" ? "dark" : "light";
  localStorage.setItem(THEME_KEY, nextTheme);
  applyTheme(nextTheme);
});

sessionIdEl.textContent = state.sessionId;
if (state.userId) userDisplay.textContent = state.displayName || state.userId;
refreshKnowledgeReviewStatus();

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

const MOBILE_LAYOUT_QUERY = window.matchMedia("(max-width: 900px)");
const PANEL_HEIGHT_DEFAULTS = {};
const PANEL_HEIGHT_BOUNDS = {};

const COL_WIDTH_DEFAULTS = {
  "graph-column": 360,
  "side-panel": 320,
};
const COL_WIDTH_BOUNDS = {
  "graph-column":      { min: 240, max: 600 },
  "side-panel":        { min: 240, max: 520 },
};

class AgentGraphView {
  constructor(containerId) {
    this._container = document.getElementById(containerId);
    this._surfaceEl = document.getElementById("graph-surface");
    this._nodes = new DataSet([]);
    this._edges = new DataSet([]);
    this._network = null;
    this._pollInterval = null;
    this._didInitialFit = false;
    this._pendingFit = true;
    this._detailEl = document.getElementById("graph-detail");
    this._detailClose = document.getElementById("graph-detail-close");
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
    this._activeDetailNodeId = null;
    this._init();
  }

  _captureOpenToolCallKeys() {
    const openKeys = new Set();
    this._detailToolcalls?.querySelectorAll("details[data-toolcall-key][open]")?.forEach((el) => {
      openKeys.add(el.getAttribute("data-toolcall-key"));
    });
    return openKeys;
  }

  _restoreOpenToolCallKeys(openKeys) {
    if (!openKeys?.size) return;
    this._detailToolcalls?.querySelectorAll("details[data-toolcall-key]")?.forEach((el) => {
      el.open = openKeys.has(el.getAttribute("data-toolcall-key"));
    });
  }

  _init() {
    const options = {
      layout: {
        hierarchical: {
          direction: "UD",
          sortMethod: "directed",
          nodeSpacing: 76,
          levelSeparation: 86,
          blockShifting: true,
          edgeMinimization: true,
        },
      },
      physics: { enabled: false },
      edges: {
        arrows: { to: { enabled: true, scaleFactor: 0.72 } },
        color: { color: "#4B5563", highlight: "#9CA3AF" },
        width: 2.4,
        smooth: { type: "cubicBezier", forceDirection: "vertical" },
      },
      nodes: {
        shape: "custom",
        borderWidth: 2,
        borderWidthSelected: 3,
      },
      interaction: {
        hover: true,
        tooltipDelay: 200,
        dragNodes: true,
        dragView: true,
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
    this._detailClose?.addEventListener("click", () => {
      this._network.unselectAll();
      this._hideDetail();
    });
  }

  _nodeTooltip(raw) {
    const lines = [
      raw.label || raw.id,
      `Status: ${raw.status || "unknown"}`,
      `Type: ${raw.type || "step"}`,
    ];
    if (raw.summary) lines.push(`Summary: ${raw.summary}`);
    if (raw.start_time) {
      if (raw.end_time) {
        const secs = ((new Date(raw.end_time) - new Date(raw.start_time)) / 1000).toFixed(1);
        lines.push(`Duration: ${secs}s`);
      } else {
        lines.push("Duration: running");
      }
    }
    return lines.join("\n");
  }

  _nodeBadge(raw) {
    const stepNumber = raw.input && raw.input.step_number;
    if (raw.type === "step" && stepNumber !== undefined && stepNumber !== null) {
      return String(stepNumber).slice(0, 2);
    }
    const typeInitials = {
      orchestrator: "O",
      planning: "P",
      execution: "E",
      tester: "T",
    };
    if (typeInitials[raw.type]) return typeInitials[raw.type];
    return String(raw.label || raw.id || "?").trim().charAt(0).toUpperCase() || "?";
  }

  _nodeRadius(raw) {
    if (raw.type === "orchestrator") return 17;
    if (raw.type === "planning") return 15;
    return 13;
  }

  _nodeRenderer(raw, colors, badge, radius, isRunning) {
    return ({ ctx, x, y, state }) => {
      const selected = Boolean(state?.selected);
      const hover = Boolean(state?.hover);
      const drawRadius = radius + (selected ? 2 : hover ? 1 : 0);
      const borderWidth = isRunning ? 2.5 : selected ? 3 : 2;

      return {
        drawNode: () => {
          ctx.save();
          ctx.beginPath();
          ctx.arc(x, y, drawRadius, 0, Math.PI * 2);
          ctx.fillStyle = hover || selected ? colors.border : colors.bg;
          ctx.fill();
          ctx.lineWidth = borderWidth;
          ctx.strokeStyle = colors.border;
          if (isRunning) ctx.setLineDash([4, 3]);
          ctx.stroke();
          ctx.setLineDash([]);

          ctx.fillStyle = colors.font;
          ctx.font = `800 ${badge.length > 1 ? 10.5 : 12}px Manrope, system-ui, sans-serif`;
          ctx.textAlign = "center";
          ctx.textBaseline = "middle";
          const metrics = ctx.measureText(badge);
          const opticalOffset = metrics.actualBoundingBoxLeft !== undefined
            ? (metrics.actualBoundingBoxLeft - metrics.actualBoundingBoxRight) / 2
            : 0;
          ctx.fillText(badge, x + opticalOffset, y);
          ctx.restore();
        },
        nodeDimensions: {
          width: (drawRadius + borderWidth) * 2,
          height: (drawRadius + borderWidth) * 2,
        },
      };
    };
  }

  _visNode(raw) {
    const typeColors = NODE_COLORS[raw.type] || NODE_COLORS.step;
    const statusOverride = STATUS_COLORS[raw.status];
    const colors = statusOverride || typeColors;
    const isRunning = raw.status === "running";
    const badge = this._nodeBadge(raw);
    const radius = this._nodeRadius(raw);
    return {
      id: raw.id,
      label: "",
      shape: "custom",
      color: {
        background: colors.bg,
        border: colors.border,
        highlight: { background: colors.border, border: colors.border },
      },
      ctxRenderer: this._nodeRenderer(raw, colors, badge, radius, isRunning),
      title: this._nodeTooltip(raw),
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

      const currentType = nodeMap[id]?.type;
      if (currentType === "orchestrator") {
        (children[id] || []).forEach((kid) => assign(kid, 1));
        return;
      }
      if (currentType === "planning") {
        (children[id] || []).forEach((kid) => assign(kid, 2));
        return;
      }

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

  _buildDisplayEdges(rawNodes, edges) {
    const nodeMap = Object.fromEntries(rawNodes.map((n) => [n.id, n]));
    const phaseTypes = new Set(["planning", "execution", "tester"]);
    const displayEdges = [];
    const phaseNodes = rawNodes
      .filter((n) => n.parent_id === "orchestrator" && phaseTypes.has(n.type))
      .sort((a, b) => {
        const ta = a.start_time ? new Date(a.start_time).getTime() : Infinity;
        const tb = b.start_time ? new Date(b.start_time).getTime() : Infinity;
        return ta - tb;
      });

    const planningNodes = phaseNodes.filter((n) => n.type === "planning");
    const childPhaseNodes = phaseNodes.filter((n) => n.type !== "planning");

    planningNodes.forEach((planning) => {
      displayEdges.push({
        id: `phase__orchestrator__${planning.id}`,
        from: "orchestrator",
        to: planning.id,
      });
    });

    childPhaseNodes.forEach((node) => {
      let parentPlanning = null;
      const nodeStart = node.start_time ? new Date(node.start_time).getTime() : Infinity;

      for (const planning of planningNodes) {
        const planningStart = planning.start_time ? new Date(planning.start_time).getTime() : -Infinity;
        if (planningStart <= nodeStart) {
          parentPlanning = planning;
        } else {
          break;
        }
      }

      displayEdges.push({
        id: `phase__${(parentPlanning || { id: "orchestrator" }).id}__${node.id}`,
        from: parentPlanning ? parentPlanning.id : "orchestrator",
        to: node.id,
      });
    });

    (edges || []).forEach((edge) => {
      const fromNode = nodeMap[edge.from];
      const toNode = nodeMap[edge.to];
      if (!fromNode || !toNode) return;

      const isTopLevelPhaseEdge =
        edge.from === "orchestrator" &&
        toNode.parent_id === "orchestrator" &&
        phaseTypes.has(toNode.type);

      if (isTopLevelPhaseEdge) return;

      displayEdges.push({
        id: edge.id || `${edge.from}__${edge.to}`,
        from: edge.from,
        to: edge.to,
      });
    });

    return displayEdges;
  }

  _resizeSurface() {
    if (!this._surfaceEl || !graphViewport) return;

    // Match the canvas to the visible viewport exactly; larger off-screen
    // surfaces make fit() center against hidden space instead of the panel.
    const targetWidth = Math.max(1, Math.round(graphViewport.clientWidth || 1));
    const targetHeight = Math.max(1, Math.round(graphViewport.clientHeight || 1));
    this._surfaceEl.style.width = `${targetWidth}px`;
    this._surfaceEl.style.height = `${targetHeight}px`;
  }

  _fitGraph() {
    if (!this._network || this._nodes.length === 0) return;
    requestAnimationFrame(() => {
      if (!this._network || this._nodes.length === 0) return;
      this._network.redraw();
      this._network.fit({ animation: { duration: 300, easingFunction: "easeInOutQuad" } });
      this._didInitialFit = true;
      this._pendingFit = false;
    });
  }

  update(graphData) {
    if (!graphData || typeof graphData.nodes !== "object") return;

    const prevNodeIds = new Set(this._nodes.getIds());
    const prevEdgeIds = new Set(this._edges.getIds());
    const rawNodes = Object.values(graphData.nodes);
    this._nodeData = graphData.nodes;
    stepExecutionFeed.update(graphData);
    const displayEdges = this._buildDisplayEdges(rawNodes, graphData.edges || []);
    const levels = this._computeLevels(rawNodes, displayEdges);
    this._resizeSurface(levels);
    const nextNodeIds = new Set(rawNodes.map((raw) => raw.id));
    const nextEdgeIds = new Set(displayEdges.map((e) => e.id || `${e.from}__${e.to}`));
    const topologyChanged =
      prevNodeIds.size !== nextNodeIds.size ||
      prevEdgeIds.size !== nextEdgeIds.size ||
      [...nextNodeIds].some((id) => !prevNodeIds.has(id)) ||
      [...nextEdgeIds].some((id) => !prevEdgeIds.has(id));

    this._nodes.getIds().forEach((nodeId) => {
      if (!nextNodeIds.has(nodeId)) this._nodes.remove(nodeId);
    });

    rawNodes.forEach((raw) => {
      const vis = this._visNode(raw);
      vis.level = levels[raw.id] ?? 0;
      if (this._nodes.get(raw.id)) {
        this._nodes.update(vis);
      } else {
        this._nodes.add(vis);
      }
    });

    this._edges.getIds().forEach((edgeId) => {
      if (!nextEdgeIds.has(edgeId)) this._edges.remove(edgeId);
    });

    const existingEdgeIds = new Set(this._edges.getIds());
    displayEdges.forEach((e) => {
      const edgeId = e.id || `${e.from}__${e.to}`;
      if (!existingEdgeIds.has(edgeId)) {
        this._edges.add({
          id: edgeId,
          from: e.from,
          to: e.to,
          hidden: false,
          physics: false,
          width: 2.4,
          smooth: { type: "cubicBezier", forceDirection: "vertical" },
        });
      }
    });

    if (rawNodes.length > 0 && (topologyChanged || !this._didInitialFit || this._pendingFit)) {
      this._fitGraph();
    }

    if (this._activeDetailNodeId) {
      if (this._nodeData[this._activeDetailNodeId]) {
        this._showDetail(this._activeDetailNodeId, { preserveScroll: true, scrollToStep: false });
      } else {
        this._hideDetail();
      }
    }
  }

  startPolling(sessionId) {
    this._currentSessionId = sessionId;
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
      if (sessionId !== this._currentSessionId) return;
      this.update(data);
    } catch (_) {
      // silently ignore network errors during polling
    }
  }

  reset() {
    this._currentSessionId = null;
    this._nodes.clear();
    this._edges.clear();
    this._nodeData = {};
    this._didInitialFit = false;
    this._pendingFit = true;
    this._resizeSurface([], { 0: 1 });
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

  _showDetail(nodeId, options = {}) {
    const raw = this._nodeData[nodeId];
    if (!raw) return;
    this._activeDetailNodeId = nodeId;
    const preserveScroll = Boolean(options.preserveScroll);
    const prevScrollTop = preserveScroll ? this._detailEl.scrollTop : 0;
    const prevOpenToolCallKeys = preserveScroll ? this._captureOpenToolCallKeys() : new Set();
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

    // Stop-step button (only for running step nodes)
    const actionsRow = document.getElementById("detail-actions-row");
    const stopStepBtn = document.getElementById("detail-stop-step-btn");
    const stepNumber = raw.input && raw.input.step_number;
    if (raw.type === "step" && raw.status === "running" && stepNumber !== undefined && stepNumber !== null) {
      stopStepBtn.disabled = false;
      stopStepBtn.textContent = "Stop step";
      stopStepBtn.onclick = async () => {
        stopStepBtn.disabled = true;
        stopStepBtn.textContent = "Stopping…";
        await requestStepCancellation(stepNumber);
      };
      actionsRow.style.display = "";
    } else {
      actionsRow.style.display = "none";
    }
    this._detailArtifacts.innerHTML = "";
    const arts = raw.artifacts || [];
    if (arts.length) {
      arts.forEach((a) => {
        this._detailArtifacts.appendChild(createArtifactListItem(a));
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
        d.setAttribute("data-toolcall-key", tc.id || `${tc.name}:${tc.start_time || ""}`);
        const dur = tc.start_time && tc.end_time
          ? ` (${((new Date(tc.end_time) - new Date(tc.start_time)) / 1000).toFixed(1)}s)`
          : "";
        const s = document.createElement("summary");
        s.textContent = `🔧 ${tc.name}${dur}`;
        d.appendChild(s);
        if (tc.args_summary) {
          d.appendChild(createJsonBlock(tc.args_summary));
        }
        if (tc.result_summary) {
          const pre = createJsonBlock(`→ ${tc.result_summary}`);
          pre.style.borderTop = "1px solid rgba(255,255,255,0.06)";
          d.appendChild(pre);
        }
        getStructurePaths(tc).forEach((path) => {
          d.appendChild(createStructureViewButton(path));
        });
        this._detailToolcalls.appendChild(d);
      });
      document.getElementById("detail-toolcalls-row").style.display = "";
    } else {
      document.getElementById("detail-toolcalls-row").style.display = "none";
    }

    // Conversation transcript is rendered live in the main chat step feed.
    const conversation = raw.type === "step" ? [] : (raw.conversation || []);
    this._detailConversation.innerHTML = "";
    if (conversation.length) {
      conversation.forEach((evt) => {
        const d = document.createElement("details");
        d.className = `timeline-${evt.type}`;
        const s = document.createElement("summary");
        const icon = evt.type === "thought" ? "💭" : evt.type === "text" ? "💬" : evt.type === "function_call" ? "🔧" : "↩";
        s.textContent = `${icon} [${evt.author}] ${evt.type}`;
        d.appendChild(s);
        d.appendChild(createJsonBlock(evt.content));
        this._detailConversation.appendChild(d);
      });
      document.getElementById("detail-conversation-row").style.display = "";
    } else {
      document.getElementById("detail-conversation-row").style.display = "none";
    }

    this._detailEl.classList.remove("hidden");
    if (raw.type === "step" && options.scrollToStep !== false) stepExecutionFeed.highlight(raw.id);
    syncPanelResizerVisibility();
    if (preserveScroll) {
      this._restoreOpenToolCallKeys(prevOpenToolCallKeys);
      this._detailEl.scrollTop = prevScrollTop;
    }
  }

  _hideDetail() {
    this._activeDetailNodeId = null;
    this._detailEl.classList.add("hidden");
    syncPanelResizerVisibility();
  }

  notifyLayoutChanged() {
    if (!this._network) return;
    this._network.redraw();
  }
}

// ---------------------------------------------------------------------------
// Step executor feed in the main chat window
// ---------------------------------------------------------------------------

class StepExecutionFeed {
  constructor() {
    this._cards = new Map();
    this._userOpen = new Map();
    this._nestedOpen = new Map();
    this._highlightedId = null;
    this._liveAnchorEl = null;
    this._liveContainerEl = null;
    this._liveStartedAt = null;
    this._liveToolHostEl = null;
    this._stepById = new Map();
    this._childNodes = new Map();
  }

  reset() {
    this._cards.clear();
    this._userOpen.clear();
    this._nestedOpen.clear();
    this._highlightedId = null;
    this._liveAnchorEl = null;
    this._liveContainerEl = null;
    this._liveStartedAt = null;
    this._liveToolHostEl = null;
    this._stepById = new Map();
    this._childNodes = new Map();
  }

  startLiveTurn(anchorEl, startedAt = Date.now()) {
    this._liveAnchorEl = anchorEl || null;
    this._liveStartedAt = startedAt;
    this._liveContainerEl = document.createElement("div");
    this._liveContainerEl.className = "step-feed-live-region";
    this._liveContainerEl.dataset.stepLiveRegion = "true";
    this._liveToolHostEl = null;

    if (anchorEl && anchorEl.parentNode === chatArea) {
      chatArea.insertBefore(this._liveContainerEl, anchorEl.nextSibling);
    } else {
      chatArea.appendChild(this._liveContainerEl);
    }

    return this._liveContainerEl;
  }

  attachLiveToolHost(hostEl) {
    if (!hostEl || this._liveToolHostEl === hostEl) return;
    this._liveToolHostEl = hostEl;
    for (const [nodeId, card] of this._cards.entries()) {
      const node = this._stepById.get(nodeId);
      if (node && !this.isRootStep(node)) continue;
      if (card.dataset.stepStartTime !== undefined) {
        hostEl.appendChild(card);
      }
    }
  }

  finishLiveTurn() {
    this._liveAnchorEl = null;
    this._liveContainerEl = null;
    this._liveStartedAt = null;
    this._liveToolHostEl = null;
  }

  update(graphData) {
    if (!graphData || typeof graphData.nodes !== "object") return;
    const liveContainer = this._activeLiveContainer();
    const steps = Object.values(graphData.nodes)
      .filter((node) => node.type === "step")
      .filter((node) => !liveContainer || this._isLiveStep(node))
      .sort((a, b) => {
        const ta = a.start_time ? new Date(a.start_time).getTime() : Infinity;
        const tb = b.start_time ? new Date(b.start_time).getTime() : Infinity;
        return ta - tb;
      });
    this.setHierarchy(steps);
    const rootSteps = steps.filter((node) => this.isRootStep(node));

    const seen = new Set(steps.map((node) => node.id));
    for (const nodeId of this._cards.keys()) {
      if (!seen.has(nodeId)) {
        this._cards.delete(nodeId);
        this._nestedOpen.delete(nodeId);
      }
    }

    const shouldStick = isChatNearBottom();
    rootSteps.forEach((node) => this._upsert(node));
    if (shouldStick) scrollToBottom();
  }

  setHierarchy(stepNodes) {
    const steps = Array.isArray(stepNodes) ? stepNodes : [];
    this._stepById = new Map(steps.map((node) => [node.id, node]));
    this._childNodes = new Map();

    steps.forEach((node) => {
      if (!this._stepById.has(node.parent_id)) return;
      const children = this._childNodes.get(node.parent_id) || [];
      children.push(node);
      this._childNodes.set(node.parent_id, children);
    });

    for (const children of this._childNodes.values()) {
      children.sort((a, b) => this._stepSortTime(a) - this._stepSortTime(b));
    }
  }

  isRootStep(node) {
    return !this._stepById.has(node?.parent_id);
  }

  _activeLiveContainer() {
    if (this._liveToolHostEl && document.body.contains(this._liveToolHostEl)) {
      return this._liveToolHostEl;
    }
    return this._liveContainerEl && this._liveContainerEl.isConnected
      ? this._liveContainerEl
      : null;
  }

  _isLiveStep(node) {
    if (!this._liveStartedAt) return true;
    if (!node.start_time) return node.status === "running";
    const startedAt = new Date(node.start_time).getTime();
    return Number.isFinite(startedAt) && startedAt >= this._liveStartedAt - 2000;
  }

  highlight(nodeId) {
    this._highlightedId = nodeId;
    for (const [id, card] of this._cards.entries()) {
      card.classList.toggle("step-feed-highlight", id === nodeId);
    }
    const card = this._cards.get(nodeId);
    if (card) {
      card.scrollIntoView({ behavior: "smooth", block: "nearest" });
      setTimeout(() => card.classList.remove("step-feed-highlight"), 1600);
    }
  }

  _upsert(node) {
    let outer = this._cards.get(node.id);
    const nextSortTime = this._stepSortTime(node);
    if (!outer || !chatArea.contains(outer)) {
      outer = this._createCard(node);
      this._cards.set(node.id, outer);
      this._placeCard(outer, node);
    } else if (state.isSending || outer.dataset.stepStartTime !== String(nextSortTime)) {
      this._placeCard(outer, node);
    }
    this._renderCard(outer, node);
  }

  appendStatic(node, container = chatArea) {
    let outer = this._cards.get(node.id);
    if (!outer || !container.contains(outer)) {
      outer = this._createCard(node);
      this._cards.set(node.id, outer);
    }
    outer.dataset.stepStartTime = String(this._stepSortTime(node));
    container.appendChild(outer);
    this._renderCard(outer, node);
    return outer;
  }

  _placeCard(outer, node) {
    outer.classList.remove("step-feed-child-message");
    const liveContainer = this._activeLiveContainer();
    if (liveContainer) {
      this._insertIntoLiveContainer(liveContainer, outer, node);
      return;
    }
    if (state.isSending && this._liveContainerEl) {
      this._insertIntoLiveContainer(this._liveContainerEl, outer, node);
      return;
    }
    this._insertSorted(outer, node);
  }

  _stepSortTime(node) {
    return node?.start_time ? new Date(node.start_time).getTime() : Infinity;
  }

  _upsertNested(node, container, ancestors) {
    let outer = this._cards.get(node.id);
    if (!outer) {
      outer = this._createCard(node);
      this._cards.set(node.id, outer);
    }
    outer.classList.add("step-feed-child-message");
    this._insertIntoLiveContainer(container, outer, node);
    this._renderCard(outer, node, ancestors);
    return outer;
  }

  _insertIntoLiveContainer(container, outer, node) {
    const newTime = this._stepSortTime(node);
    outer.dataset.stepStartTime = String(newTime);

    for (const el of [...container.children]) {
      if (el === outer) continue;
      if (!el.dataset.stepStartTime) continue;
      if (newTime < Number(el.dataset.stepStartTime)) {
        container.insertBefore(outer, el);
        return;
      }
    }
    container.appendChild(outer);
  }

  _insertSorted(outer, node) {
    const newTime = this._stepSortTime(node);
    outer.dataset.stepStartTime = String(newTime);

    // Walk all chat children to find the right insertion point.
    // - Step cards: compare by start_time, insert before the first later one.
    // - User/agent messages: track as anchor, but reset when a step card is
    //   found after them (so we insert after the most recent step card too).
    const children = [...chatArea.children];
    const liveAnchor = this._liveAnchorEl && chatArea.contains(this._liveAnchorEl)
      ? this._liveAnchorEl
      : null;
    let insertAfter = liveAnchor; // element to insert after (null = before first child)
    let passedLiveAnchor = !liveAnchor;

    for (const el of children) {
      if (el === outer) continue;
      if (liveAnchor && !passedLiveAnchor) {
        if (el === liveAnchor) passedLiveAnchor = true;
        continue;
      }

      if (el.dataset.stepStartTime) {
        const elTime = Number(el.dataset.stepStartTime);
        if (newTime < elTime) {
          // Found a later step card — insert before it
          if (insertAfter) {
            chatArea.insertBefore(outer, insertAfter.nextElementSibling);
          } else {
            chatArea.insertBefore(outer, el);
          }
          return;
        }
        // This step card is earlier — update anchor
        insertAfter = el;
      } else if (el.dataset.msgIndex !== undefined) {
        // User/agent message — update anchor
        insertAfter = el;
      } else if (el.classList.contains("user-message")) {
        // Live messages have no msgIndex until the session is reloaded.
        insertAfter = el;
      } else if (
        insertAfter &&
        el.classList.contains("agent-message") &&
        !el.classList.contains("step-feed-message")
      ) {
        chatArea.insertBefore(outer, el);
        return;
      }
    }

    // No later step card found — insert after the last tracked element.
    if (insertAfter) {
      chatArea.insertBefore(outer, insertAfter.nextElementSibling);
    } else {
      chatArea.appendChild(outer);
    }
  }

  _createCard(node) {
    const outer = document.createElement("div");
    outer.className = "message agent-message step-feed-message";
    outer.dataset.stepNodeId = node.id;
    outer.dataset.stepStartTime = node.start_time ? String(new Date(node.start_time).getTime()) : "";
    outer.appendChild(createAgentAvatarEl());

    const bubble = document.createElement("div");
    bubble.className = "message-bubble step-feed-bubble";
    const details = document.createElement("details");
    details.className = "step-feed-details";
    details.addEventListener("toggle", () => {
      this._userOpen.set(node.id, details.open);
    });
    bubble.appendChild(details);
    outer.appendChild(bubble);
    return outer;
  }

  _wireNested(nodeId, key, details) {
    let nodeState = this._nestedOpen.get(nodeId);
    if (!nodeState) {
      nodeState = new Map();
      this._nestedOpen.set(nodeId, nodeState);
    }
    if (nodeState.has(key)) {
      details.open = nodeState.get(key);
    }
    details.dataset.stepNestedKey = key;
    details.addEventListener("toggle", (event) => {
      if (event.target !== details) return;
      nodeState.set(key, details.open);
    });
    return details;
  }

  _renderCard(outer, node, ancestors = new Set([node.id])) {
    outer.dataset.stepNodeId = node.id;
    outer.classList.toggle("step-feed-highlight", this._highlightedId === node.id);

    const bubble = outer.querySelector(".step-feed-bubble");
    bubble?.querySelector(":scope > .step-feed-child-section")?.remove();

    const details = outer.querySelector(".step-feed-details");
    const userChoice = this._userOpen.get(node.id);
    details.open = userChoice === undefined ? node.status === "running" : userChoice;
    details.innerHTML = "";

    const summary = document.createElement("summary");
    summary.className = "step-feed-summary";
    const title = document.createElement("span");
    title.className = "step-feed-title";
    title.textContent = stepFeedTitle(node);
    const badge = document.createElement("span");
    badge.className = `badge badge-${node.status || "idle"}`;
    badge.textContent = node.status || "idle";
    const meta = document.createElement("span");
    meta.className = "step-feed-meta";
    meta.textContent = formatStepDuration(node);
    summary.append(title, badge, meta);

    const stepNumber = node.input && node.input.step_number;
    if (node.status === "running" && stepNumber !== undefined && stepNumber !== null) {
      const stopBtn = document.createElement("button");
      stopBtn.type = "button";
      stopBtn.className = "step-feed-stop-btn";
      stopBtn.textContent = "Stop";
      stopBtn.title = `Stop step ${stepNumber}`;
      stopBtn.addEventListener("click", async (event) => {
        event.preventDefault();
        event.stopPropagation();
        stopBtn.disabled = true;
        stopBtn.textContent = "Stopping…";
        await requestStepCancellation(stepNumber);
      });
      summary.appendChild(stopBtn);
    }
    details.appendChild(summary);

    const body = document.createElement("div");
    body.className = "step-feed-body";

    if (node.summary) {
      const p = document.createElement("div");
      p.className = "step-feed-node-summary";
      p.textContent = node.summary;
      body.appendChild(p);
    }

    if (node.input && Object.keys(node.input).length) {
      body.appendChild(this._wireNested(node.id, "input", renderStepInput(node.input)));
    }

    const childNodes = (this._childNodes.get(node.id) || [])
      .filter((child) => !ancestors.has(child.id));
    if (childNodes.length) {
      const section = document.createElement("div");
      section.className = "step-feed-section step-feed-child-section";
      const label = document.createElement("div");
      label.className = "step-feed-section-title";
      label.textContent = `Sub-executors (${childNodes.length})`;
      const childHost = document.createElement("div");
      childHost.className = "step-feed-child-list";
      section.append(label, childHost);

      childNodes.forEach((child) => {
        const nextAncestors = new Set(ancestors);
        nextAncestors.add(child.id);
        this._upsertNested(child, childHost, nextAncestors);
      });

      bubble?.appendChild(section);
    }

    const conversation = node.conversation || [];
    if (conversation.length) {
      const section = document.createElement("div");
      section.className = "step-feed-section";
      const sectionDetails = document.createElement("details");
      sectionDetails.className = "step-feed-section-details";
      const sectionSummary = document.createElement("summary");
      sectionSummary.className = "step-feed-section-title";
      sectionSummary.textContent = `Conversations (${conversation.length})`;
      sectionDetails.appendChild(sectionSummary);
      conversation.forEach((evt, idx) => {
        const key = `conversation:${idx}:${evt.timestamp || ""}:${evt.type || ""}:${evt.author || ""}`;
        sectionDetails.appendChild(this._wireNested(node.id, key, renderStepConversationEvent(evt)));
      });
      sectionDetails.addEventListener("toggle", () => {
        sectionSummary.classList.toggle("open", sectionDetails.open);
      });
      section.appendChild(this._wireNested(node.id, "section:conversation", sectionDetails));
      body.appendChild(section);
    }

    const toolCalls = node.tool_calls || [];
    if (toolCalls.length) {
      const section = document.createElement("div");
      section.className = "step-feed-section";
      const sectionDetails = document.createElement("details");
      sectionDetails.className = "step-feed-section-details";
      const sectionSummary = document.createElement("summary");
      sectionSummary.className = "step-feed-section-title";
      sectionSummary.textContent = `Tool calls (${toolCalls.length})`;
      sectionDetails.appendChild(sectionSummary);
      toolCalls.forEach((tc, idx) => {
        const key = `tool:${idx}:${tc.name || ""}:${tc.start_time || ""}`;
        sectionDetails.appendChild(this._wireNested(node.id, key, renderStepToolCall(tc)));
      });
      sectionDetails.addEventListener("toggle", () => {
        sectionSummary.classList.toggle("open", sectionDetails.open);
      });
      section.appendChild(this._wireNested(node.id, "section:toolcalls", sectionDetails));
      body.appendChild(section);
    }

    const artifacts = node.artifacts || [];
    if (artifacts.length) {
      const section = document.createElement("div");
      section.className = "step-feed-section";
      const label = document.createElement("div");
      label.className = "step-feed-section-title";
      label.textContent = "Artifacts";
      const list = document.createElement("ul");
      list.className = "detail-artifacts step-feed-artifacts";
      artifacts.forEach((artifact) => {
        list.appendChild(createArtifactListItem(artifact));
      });
      section.append(label, list);
      body.appendChild(section);
    }

    if (!body.childElementCount) {
      const empty = document.createElement("div");
      empty.className = "step-feed-empty";
      empty.textContent = "Waiting for step executor events…";
      body.appendChild(empty);
    }

    details.appendChild(body);
  }
}

// ---------------------------------------------------------------------------
// Execution Plan Graph (floating popup in chat column)
// ---------------------------------------------------------------------------

const PLAN_NODE_STATUS_COLORS = {
  pending:   { bg: "#374151", border: "#6B7280", font: "#9CA3AF" },
  running:   { bg: "#FBBF24", border: "#F59E0B", font: "#1a1a1a" },
  success:   { bg: "#10B981", border: "#059669", font: "#043F2E" },
  failed:    { bg: "#EF4444", border: "#DC2626", font: "#fff" },
  blocked:   { bg: "#1F2937", border: "#374151", font: "#4B5563" },
};

class ExecutionPlanView {
  constructor(containerId) {
    this._container = document.getElementById(containerId);
    this._network = null;
    this._pollInterval = null;
    this._didInitialFit = false;
    this._structureKey = null;
    this._subgraphs = [];
    this._currentIndex = 0;
    this._hierarchicalMode = true;
    this._init();
  }

  _init() {
    if (!this._container) return;
    const options = {
      layout: {
        hierarchical: {
          direction: "LR",
          sortMethod: "directed",
          nodeSpacing: 120,
          levelSeparation: 170,
          blockShifting: true,
          edgeMinimization: true,
        },
      },
      physics: { enabled: false },
      edges: {
        arrows: { to: { enabled: true, scaleFactor: 0.6 } },
        color: { color: "#4B5563", highlight: "#9CA3AF" },
        width: 1.5,
        smooth: { type: "cubicBezier", forceDirection: "horizontal" },
      },
      nodes: {
        shape: "box",
        borderWidth: 2,
        borderWidthSelected: 3,
        font: { size: 14, face: "Manrope, sans-serif", bold: true },
        margin: { top: 8, bottom: 8, left: 12, right: 12 },
      },
      interaction: {
        hover: true,
        tooltipDelay: 200,
        dragNodes: true,
        dragView: true,
        zoomView: true,
      },
    };
    this._network = new Network(
      this._container,
      { nodes: new DataSet([]), edges: new DataSet([]) },
      options
    );
  }

  _computeLevels(nodeIds, rawEdges) {
    const inDeg = Object.fromEntries(nodeIds.map((id) => [id, 0]));
    const adj   = Object.fromEntries(nodeIds.map((id) => [id, []]));
    rawEdges.forEach((e) => {
      const from = Array.isArray(e) ? e[0] : e.from;
      const to   = Array.isArray(e) ? e[1] : e.to;
      if (adj[from]) adj[from].push(to);
      if (to in inDeg) inDeg[to]++;
    });
    const levels = {};
    const queue = nodeIds.filter((id) => inDeg[id] === 0);
    queue.forEach((id) => { levels[id] = 0; });
    while (queue.length) {
      const curr = queue.shift();
      (adj[curr] || []).forEach((nxt) => {
        levels[nxt] = Math.max(levels[nxt] ?? 0, (levels[curr] ?? 0) + 1);
        if (--inDeg[nxt] === 0) queue.push(nxt);
      });
    }
    nodeIds.forEach((id) => { if (!(id in levels)) levels[id] = 0; });
    return levels;
  }

  _visNode(nodeId, node, level) {
    const status = node.status || "pending";
    const isRunning = status === "running";
    const colors = PLAN_NODE_STATUS_COLORS[status] || PLAN_NODE_STATUS_COLORS.pending;
    return {
      id: nodeId,
      label: node.label || nodeId,
      title: node.action || nodeId,
      level,
      color: {
        background: colors.bg,
        border: colors.border,
        highlight: { background: colors.border, border: colors.border },
      },
      font: { color: colors.font, size: 14, bold: true, face: "Manrope, sans-serif" },
      shapeProperties: isRunning ? { borderDashes: [4, 3] } : {},
      borderWidth: isRunning ? 2.5 : 2,
    };
  }

  _extractConnectedSubgraphs(graphData) {
    const nodes = graphData.nodes || {};
    const rawEdges = graphData.edges || [];
    const nodeIds = Object.keys(nodes);
    if (nodeIds.length === 0) return [graphData];

    // No edges at all → treat as one graph
    if (rawEdges.length === 0) return [graphData];

    const adj = {};
    nodeIds.forEach((id) => { adj[id] = []; });
    rawEdges.forEach((e) => {
      const from = Array.isArray(e) ? e[0] : e.from;
      const to = Array.isArray(e) ? e[1] : e.to;
      if (adj[from]) adj[from].push(to);
      if (adj[to]) adj[to].push(from);
    });

    const visited = new Set();
    const components = [];
    for (const id of nodeIds) {
      if (visited.has(id)) continue;
      const compIds = new Set();
      const queue = [id];
      visited.add(id);
      while (queue.length) {
        const cur = queue.shift();
        compIds.add(cur);
        for (const nb of (adj[cur] || [])) {
          if (!visited.has(nb)) {
            visited.add(nb);
            queue.push(nb);
          }
        }
      }
      const compNodes = {};
      const compEdges = [];
      for (const cid of compIds) {
        if (nodes[cid]) compNodes[cid] = nodes[cid];
      }
      rawEdges.forEach((e) => {
        const from = Array.isArray(e) ? e[0] : e.from;
        const to = Array.isArray(e) ? e[1] : e.to;
        if (compIds.has(from) || compIds.has(to)) {
          compEdges.push(e);
        }
      });
      components.push({ nodes: compNodes, edges: compEdges });
    }

    // Merge all isolated (single-node) components into one group
    const multi = components.filter((c) => Object.keys(c.nodes).length > 1);
    const singles = components.filter((c) => Object.keys(c.nodes).length === 1);
    if (singles.length > 0) {
      const mergedNodes = {};
      singles.forEach((c) => Object.assign(mergedNodes, c.nodes));
      multi.push({ nodes: mergedNodes, edges: [] });
    }

    return multi.length > 0 ? multi : [graphData];
  }

  update(graphData) {
    if (!graphData || typeof graphData.nodes !== "object") return;
    const nodeEntries = Object.entries(graphData.nodes);
    if (nodeEntries.length === 0) return;

    // Auto-show popup on first data
    if (!this._didInitialFit && planGraphPopup?.classList.contains("hidden")) {
      showPlanGraph();
    }

    const rawEdges = graphData.edges || [];
    const nodeIds = nodeEntries.map(([id]) => id);

    // Detect structural changes
    const structureKey = JSON.stringify({ ids: [...nodeIds].sort(), edges: rawEdges });
    const structureChanged = structureKey !== this._structureKey;
    if (structureChanged) {
      this._structureKey = structureKey;
      this._subgraphs = this._extractConnectedSubgraphs(graphData);
      this._currentIndex = 0;
    }

    if (this._subgraphs.length === 0) return;
    this._renderCurrentSubgraph();
    this._updateNavUI();
  }

  _renderCurrentSubgraph() {
    const sub = this._subgraphs[this._currentIndex];
    if (!sub) return;

    const nodeEntries = Object.entries(sub.nodes);
    const rawEdges = sub.edges || [];
    const nodeIds = nodeEntries.map(([id]) => id);
    const levels = this._computeLevels(nodeIds, rawEdges);
    const maxLevel = Math.max(...Object.values(levels), 0);
    const noHierarchy = maxLevel === 0 && nodeIds.length > 1;

    let visNodes;
    if (noHierarchy) {
      if (this._hierarchicalMode !== false) {
        this._network.setOptions({ layout: { hierarchical: false, randomSeed: 42 } });
        this._hierarchicalMode = false;
      }
      const cols = Math.ceil(Math.sqrt(nodeIds.length));
      const gapX = 260;
      const gapY = 55;
      visNodes = nodeEntries.map(([id, n], i) => {
        const col = i % cols;
        const row = Math.floor(i / cols);
        return {
          ...this._visNode(id, n, 0),
          x: col * gapX,
          y: row * gapY,
          fixed: { x: true, y: true },
        };
      });
    } else {
      if (this._hierarchicalMode !== true) {
        this._network.setOptions({
          layout: {
            hierarchical: {
              direction: "LR",
              sortMethod: "directed",
              nodeSpacing: 120,
              levelSeparation: 170,
              blockShifting: true,
              edgeMinimization: true,
            },
          },
        });
        this._hierarchicalMode = true;
      }
      visNodes = nodeEntries.map(([id, n]) => this._visNode(id, n, levels[id] ?? 0));
    }

    const visEdges = rawEdges.map((e) => {
      const from = Array.isArray(e) ? e[0] : e.from;
      const to = Array.isArray(e) ? e[1] : e.to;
      return {
        id: `e__${from}__${to}`,
        from,
        to,
        physics: false,
        hidden: false,
        smooth: { type: "cubicBezier", forceDirection: "horizontal" },
      };
    });

    // Save camera before setData resets the layout
    let savedCamera = null;
    if (this._didInitialFit) {
      try {
        savedCamera = {
          position: this._network.getViewPosition(),
          scale: this._network.getScale(),
        };
      } catch (_) {}
    }

    this._network.setData({ nodes: new DataSet(visNodes), edges: new DataSet(visEdges) });

    if (!this._didInitialFit) {
      this._network.fit({ animation: { duration: 300, easingFunction: "easeInOutQuad" } });
      this._didInitialFit = true;
    } else {
      if (savedCamera) {
        this._network.moveTo({
          position: savedCamera.position,
          scale: savedCamera.scale,
          animation: false,
        });
      }
    }
  }

  _updateNavUI() {
    const counter = document.getElementById("plan-graph-counter");
    const prevBtn = document.getElementById("plan-graph-prev");
    const nextBtn = document.getElementById("plan-graph-next");
    if (this._subgraphs.length <= 1) {
      if (counter) counter.textContent = "";
      if (prevBtn) prevBtn.style.display = "none";
      if (nextBtn) nextBtn.style.display = "none";
    } else {
      if (counter) counter.textContent = `Plan Graph ${this._currentIndex + 1} / ${this._subgraphs.length}`;
      if (prevBtn) { prevBtn.style.display = ""; prevBtn.disabled = this._currentIndex === 0; }
      if (nextBtn) { nextBtn.style.display = ""; nextBtn.disabled = this._currentIndex >= this._subgraphs.length - 1; }
    }
  }

  goPrev() {
    if (this._currentIndex > 0) {
      this._currentIndex--;
      this._didInitialFit = false;
      this._renderCurrentSubgraph();
      this._updateNavUI();
    }
  }

  goNext() {
    if (this._currentIndex < this._subgraphs.length - 1) {
      this._currentIndex++;
      this._didInitialFit = false;
      this._renderCurrentSubgraph();
      this._updateNavUI();
    }
  }

  startPolling(sessionId) {
    this._currentSessionId = sessionId;
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
      const resp = await fetch(`/api/execution-graph/${sessionId}`);
      if (!resp.ok) return;
      const data = await resp.json();
      if (sessionId !== this._currentSessionId) return;
      this.update(data);
    } catch (_) {}
  }

  reset() {
    this._currentSessionId = null;
    this._hierarchicalMode = true;
    this._network?.setOptions({
      layout: {
        hierarchical: {
          direction: "LR",
          sortMethod: "directed",
          nodeSpacing: 120,
          levelSeparation: 170,
          blockShifting: true,
          edgeMinimization: true,
        },
      },
    });
    this._network?.setData({ nodes: new DataSet([]), edges: new DataSet([]) });
    this._didInitialFit = false;
    this._structureKey = null;
    this._subgraphs = [];
    this._currentIndex = 0;
    this._updateNavUI();
    this.stopPolling();
  }

  notifyLayoutChanged() {
    this._network?.redraw();
    if (!this._didInitialFit) {
      this._network?.fit({ animation: false });
    }
  }

  zoomIn() {
    if (!this._network) return;
    const scale = this._network.getScale() * 1.3;
    this._network.moveTo({ scale, animation: { duration: 200, easingFunction: "easeInOutQuad" } });
  }

  zoomOut() {
    if (!this._network) return;
    const scale = this._network.getScale() / 1.3;
    this._network.moveTo({ scale, animation: { duration: 200, easingFunction: "easeInOutQuad" } });
  }

  fitToView() {
    if (!this._network) return;
    this._network.fit({ animation: { duration: 300, easingFunction: "easeInOutQuad" } });
  }

  _setStatus(s) {
    const el = document.getElementById("plan-graph-status");
    if (el) { el.textContent = s; el.className = `graph-status status-${s}`; }
  }
}

const stepExecutionFeed = new StepExecutionFeed();
const agentGraph = new AgentGraphView("agent-graph");
const planGraph = new ExecutionPlanView("plan-graph-canvas");

async function requestStepCancellation(stepNumber) {
  if (stepNumber === undefined || stepNumber === null) return false;
  try {
    const resp = await fetch(`/api/sessions/${state.sessionId}/cancel-step/${stepNumber}`, { method: "POST" });
    return resp.ok;
  } catch (_) {
    return false;
  }
}

// ---------------------------------------------------------------------------
// Plan graph popup toggle
// ---------------------------------------------------------------------------

const planGraphPopup = document.getElementById("plan-graph-popup");
const planGraphToggleBtn = document.getElementById("plan-graph-toggle");
const planGraphCloseBtn = document.getElementById("plan-graph-close");

function showPlanGraph() {
  planGraphPopup?.classList.remove("hidden");
  requestAnimationFrame(() => {
    planGraph.notifyLayoutChanged();
    planGraph.fitToView();
  });
}

function hidePlanGraph() {
  planGraphPopup?.classList.add("hidden");
}

planGraphToggleBtn?.addEventListener("click", () => {
  if (planGraphPopup?.classList.contains("hidden")) {
    showPlanGraph();
  } else {
    hidePlanGraph();
  }
});

planGraphCloseBtn?.addEventListener("click", hidePlanGraph);
document.getElementById("plan-graph-zoom-in")?.addEventListener("click", () => planGraph.zoomIn());
document.getElementById("plan-graph-zoom-out")?.addEventListener("click", () => planGraph.zoomOut());
document.getElementById("plan-graph-fit")?.addEventListener("click", () => planGraph.fitToView());
document.getElementById("plan-graph-prev")?.addEventListener("click", () => planGraph.goPrev());
document.getElementById("plan-graph-next")?.addEventListener("click", () => planGraph.goNext());
// ---------------------------------------------------------------------------

function isMobileLayout() {
  return MOBILE_LAYOUT_QUERY.matches;
}

function panelStorageKey(targetId) {
  const user = state.userId || "anon";
  return `mat_panel_height_${user}_${targetId}`;
}

function clamp(n, min, max) {
  return Math.max(min, Math.min(max, n));
}

function getTargetHeight(targetEl) {
  return Math.round(targetEl.getBoundingClientRect().height);
}

function applyTargetHeight(targetEl, heightPx) {
  if (!targetEl) return;
  const bounds = PANEL_HEIGHT_BOUNDS[targetEl.id];
  if (!bounds) return;
  targetEl.style.height = `${clamp(heightPx, bounds.min, bounds.max)}px`;
}

function persistTargetHeight(targetEl) {
  if (!targetEl) return;
  localStorage.setItem(panelStorageKey(targetEl.id), String(getTargetHeight(targetEl)));
}

function applyStoredPanelHeights() {
  for (const [targetId, fallback] of Object.entries(PANEL_HEIGHT_DEFAULTS)) {
    const el = document.getElementById(targetId);
    if (!el) continue;
    if (isMobileLayout()) {
      el.style.removeProperty("height");
      continue;
    }

    const raw = localStorage.getItem(panelStorageKey(targetId));
    const parsed = raw ? Number(raw) : fallback;
    const nextHeight = Number.isFinite(parsed) ? parsed : fallback;
    applyTargetHeight(el, nextHeight);
  }
}

function refreshGraphAndStructureLayout() {
  agentGraph.notifyLayoutChanged();
  for (const tab of structureTabs.values()) {
    if (!tab.viewer) continue;
    try {
      tab.viewer.resize();
      tab.viewer.render();
    } catch (_) {
      // ignore transient resize/render issues
    }
  }
}

function syncPanelResizerVisibility() {
  graphResizer?.classList.add("hidden");
}

function initPanelResizer(handleEl, targetEl) {
  if (!handleEl || !targetEl) return;

  const keyStep = 16;

  const commit = () => {
    persistTargetHeight(targetEl);
    refreshGraphAndStructureLayout();
  };

  const resizeBy = (delta) => {
    const curr = getTargetHeight(targetEl);
    applyTargetHeight(targetEl, curr + delta);
    refreshGraphAndStructureLayout();
  };

  handleEl.addEventListener("pointerdown", (e) => {
    if (isMobileLayout() || handleEl.classList.contains("hidden")) return;
    e.preventDefault();

    const startY = e.clientY;
    const startHeight = getTargetHeight(targetEl);
    handleEl.classList.add("resizing");
    handleEl.setPointerCapture(e.pointerId);

    const onMove = (moveEvt) => {
      const dy = moveEvt.clientY - startY;
      applyTargetHeight(targetEl, startHeight + dy);
      refreshGraphAndStructureLayout();
    };

    const onUp = () => {
      handleEl.classList.remove("resizing");
      handleEl.removeEventListener("pointermove", onMove);
      handleEl.removeEventListener("pointerup", onUp);
      handleEl.removeEventListener("pointercancel", onUp);
      commit();
    };

    handleEl.addEventListener("pointermove", onMove);
    handleEl.addEventListener("pointerup", onUp);
    handleEl.addEventListener("pointercancel", onUp);
  });

  handleEl.addEventListener("keydown", (e) => {
    if (isMobileLayout() || handleEl.classList.contains("hidden")) return;
    if (e.key === "ArrowUp") {
      e.preventDefault();
      resizeBy(-keyStep);
      commit();
    }
    if (e.key === "ArrowDown") {
      e.preventDefault();
      resizeBy(keyStep);
      commit();
    }
  });
}

// ---------------------------------------------------------------------------
// Column (horizontal) resizing — mirrors the panel (vertical) resizer pattern
// ---------------------------------------------------------------------------

function colStorageKey(colId) {
  return `mat_col_width_${state.userId || "anon"}_${colId}`;
}
function getColWidth(colEl) {
  return Math.round(colEl.getBoundingClientRect().width);
}
function applyColWidth(colEl, widthPx) {
  const bounds = COL_WIDTH_BOUNDS[colEl.id];
  if (!bounds) return;
  colEl.style.width = `${clamp(widthPx, bounds.min, bounds.max)}px`;
}
function persistColWidth(colEl) {
  localStorage.setItem(colStorageKey(colEl.id), String(getColWidth(colEl)));
}
function applyStoredColWidths() {
  for (const colId of ["graph-column", "side-panel"]) {
    const el = document.getElementById(colId);
    if (!el) continue;
    if (isMobileLayout()) { el.style.removeProperty("width"); continue; }
    const raw = localStorage.getItem(colStorageKey(colId));
    const w = raw ? Number(raw) : COL_WIDTH_DEFAULTS[colId];
    applyColWidth(el, Number.isFinite(w) ? w : COL_WIDTH_DEFAULTS[colId]);
  }
}
function syncColResizerVisibility() {
  const mobile = isMobileLayout();
  colResizerGraph?.classList.toggle("hidden", mobile);
  colResizerSide?.classList.toggle("hidden", mobile);
  colResizerFiles?.classList.add("hidden");
}

/**
 * initColResizer — horizontal mirror of initPanelResizer.
 * direction: +1 = drag-right widens targetEl (left col), -1 = drag-right narrows it (right col).
 */
function initColResizer(handleEl, targetEl, direction = 1) {
  if (!handleEl || !targetEl) return;
  const keyStep = 16;
  const commit = () => {
    persistColWidth(targetEl);
    refreshGraphAndStructureLayout();
  };
  handleEl.addEventListener("pointerdown", (e) => {
    if (isMobileLayout() || handleEl.classList.contains("hidden")) return;
    e.preventDefault();
    const startX = e.clientX;
    const startWidth = getColWidth(targetEl);
    handleEl.classList.add("resizing");
    handleEl.setPointerCapture(e.pointerId);
    const onMove = (moveEvt) => {
      applyColWidth(targetEl, startWidth + direction * (moveEvt.clientX - startX));
      refreshGraphAndStructureLayout();
    };
    const onUp = () => {
      handleEl.classList.remove("resizing");
      handleEl.removeEventListener("pointermove", onMove);
      handleEl.removeEventListener("pointerup", onUp);
      handleEl.removeEventListener("pointercancel", onUp);
      commit();
    };
    handleEl.addEventListener("pointermove", onMove);
    handleEl.addEventListener("pointerup", onUp);
    handleEl.addEventListener("pointercancel", onUp);
  });
  handleEl.addEventListener("keydown", (e) => {
    if (isMobileLayout() || handleEl.classList.contains("hidden")) return;
    if (e.key === "ArrowLeft")  { e.preventDefault(); applyColWidth(targetEl, getColWidth(targetEl) + direction * -keyStep); commit(); }
    if (e.key === "ArrowRight") { e.preventDefault(); applyColWidth(targetEl, getColWidth(targetEl) + direction *  keyStep); commit(); }
  });
}

function initColResizers() {
  applyStoredColWidths();
  fileExplorerCol?.classList.add("is-open");
  syncColResizerVisibility();
  initColResizer(colResizerGraph, graphColumn, 1);
  initColResizer(colResizerSide, sidePanel, -1);
  MOBILE_LAYOUT_QUERY.addEventListener("change", () => {
    applyStoredColWidths();
    fileExplorerCol?.classList.add("is-open");
    syncColResizerVisibility();
    refreshGraphAndStructureLayout();
  });
}

function initPanelResizers() {
  applyStoredPanelHeights();
  syncPanelResizerVisibility();

  MOBILE_LAYOUT_QUERY.addEventListener("change", () => {
    applyStoredPanelHeights();
    syncPanelResizerVisibility();
    refreshGraphAndStructureLayout();
  });
}

// ---------------------------------------------------------------------------
// Login / username management
// ---------------------------------------------------------------------------

function _isUuid(s) {
  return /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(s);
}

function _isValidIdentity(s) {
  return s === "user" || _isUuid(s);
}

function showLoginModal() {
  if (state.deploymentMode !== "server") {
    hideLoginModal();
    return;
  }
  loginModal.classList.remove("hidden");
  loginView.classList.remove("hidden");
  registerView.classList.add("hidden");
  loginInput.value = state.displayName || "";
  loginPassword.value = "";
  loginError.textContent = "";
  loginUuidDisplay.textContent = state.userId ? `UUID: ${state.userId}` : "";
  // Hide register link when already logged in — log out first to register a new account.
  const registerLink = document.getElementById("switch-to-register")?.parentElement;
  if (registerLink) registerLink.style.display = state.userId ? "none" : "";
  loginInput.focus();
}

async function logout() {
  const userId = state.userId;
  const deploymentMode = state.deploymentMode;
  if (deploymentMode === "server" && userId) {
    try {
      await fetch("/api/auth/logout", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ user_id: userId }),
      });
    } catch (_) { /* best-effort worker shutdown */ }
  }
  state.userId = "";
  state.displayName = "";
  state.activeSessionUserId = "";
  state.isAdmin = false;
  state.sessionReady = false;
  localStorage.removeItem("mat_userId");
  localStorage.removeItem("mat_displayName");
  localStorage.removeItem("mat_sessionId");
  localStorage.removeItem("mat_deploymentMode");
  userDisplay.textContent = "—";
  chatArea.innerHTML = "";
  stepExecutionFeed.reset();
  sessionListEl.innerHTML = '<li class="empty">Sign in to see sessions</li>';
  renderSessionFilesTree([]);
  clearCurrentUploads();
  agentGraph.reset();
  planGraph.reset();
  hidePlanGraph();
  closeSettingsModal();
  showLoginModal();
}

function showRegisterModal() {
  loginModal.classList.remove("hidden");
  loginView.classList.add("hidden");
  registerView.classList.remove("hidden");
  regInput.value = "";
  regPassword.value = "";
  regConfirm.value = "";
  regError.textContent = "";
  regInput.focus();
}

function hideLoginModal() {
  loginModal.classList.add("hidden");
}

function renderUserDisplay() {
  const label = state.displayName || state.userId;
  userDisplay.textContent = state.isAdmin ? `${label} (admin)` : label;
}

function canWriteActiveSession() {
  return state.deploymentMode === "local" || !state.activeSessionUserId || state.activeSessionUserId === state.userId;
}

function activeSessionBackendUserId() {
  return state.deploymentMode === "local"
    ? (state.activeSessionUserId || state.userId)
    : state.userId;
}

async function refreshAccess() {
  state.isAdmin = false;
  if (!state.userId) return;
  try {
    const resp = await fetch(`/api/session-access/${encodeURIComponent(state.userId)}`);
    if (!resp.ok) return;
    const access = await resp.json();
    state.isAdmin = Boolean(access.is_admin);
  } catch (_) {
    state.isAdmin = false;
  }
}

function _applySession(result) {
  state.userId = result.user_id;
  state.displayName = result.display_name;
  state.activeSessionUserId = result.user_id;
  state.sessionId = `session-${Math.floor(Date.now() / 1000)}`;
  state.sessionReady = false;
  state.isAdmin = Boolean(result.is_admin);
  loginUuidDisplay.textContent = `UUID: ${result.user_id}`;
  localStorage.setItem("mat_deploymentMode", state.deploymentMode);
  localStorage.setItem("mat_userId", result.user_id);
  localStorage.setItem("mat_displayName", result.display_name);
  localStorage.setItem("mat_sessionId", state.sessionId);
  sessionIdEl.textContent = state.sessionId;
  chatArea.innerHTML = "";
  stepExecutionFeed.reset();
  renderSessionFilesTree([]);
  clearCurrentUploads();
  agentGraph.reset();
  planGraph.reset();
  hidePlanGraph();
  renderUserDisplay();
  hideLoginModal();
  applyStoredPanelHeights();
  applyStoredColWidths();
  fileExplorerCol?.classList.add("is-open");
  syncColResizerVisibility();
  loadSessions();
}

async function applyLogin(displayName, password = null) {
  loginError.textContent = "";
  let result;
  try {
    const resp = await fetch("/api/auth/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ display_name: displayName, password }),
    });
    const body = await resp.json().catch(() => ({}));
    if (!resp.ok) {
      loginError.textContent = body.detail || `Login failed (${resp.status})`;
      return;
    }
    result = body;
  } catch (err) {
    loginError.textContent = `Login failed: ${err.message}`;
    return;
  }
  _applySession(result);
}

async function applyRegister(displayName, password, confirm) {
  regError.textContent = "";
  if (password !== confirm) {
    regError.textContent = "Passwords do not match.";
    return;
  }
  let result;
  try {
    const resp = await fetch("/api/auth/register", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ display_name: displayName, password }),
    });
    const body = await resp.json().catch(() => ({}));
    if (!resp.ok) {
      regError.textContent = body.detail || `Registration failed (${resp.status})`;
      return;
    }
    result = body;
  } catch (err) {
    regError.textContent = `Registration failed: ${err.message}`;
    return;
  }
  _applySession(result);
}

loginSubmit.addEventListener("click", () => {
  const name = loginInput.value.trim();
  if (name) applyLogin(name, loginPassword.value || null);
});

loginInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") loginPassword.focus();
});

loginPassword.addEventListener("keydown", (e) => {
  if (e.key === "Enter") {
    const name = loginInput.value.trim();
    if (name) applyLogin(name, loginPassword.value || null);
  }
});

regSubmit.addEventListener("click", () => {
  const name = regInput.value.trim();
  if (name) applyRegister(name, regPassword.value, regConfirm.value);
});

regInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") regPassword.focus();
});

regPassword.addEventListener("keydown", (e) => {
  if (e.key === "Enter") regConfirm.focus();
});

regConfirm.addEventListener("keydown", (e) => {
  if (e.key === "Enter") {
    const name = regInput.value.trim();
    if (name) applyRegister(name, regPassword.value, regConfirm.value);
  }
});

switchToRegister.addEventListener("click", () => showRegisterModal());
switchToLogin.addEventListener("click", () => showLoginModal());

editUserBtn.addEventListener("click", () => showLoginModal());
logoutBtn.addEventListener("click", () => logout());
settingsLogoutBtn.addEventListener("click", () => logout());

const savePasswordBtn = document.getElementById("settings-save-password-btn");
const passwordMsg = document.getElementById("settings-password-msg");
const settingsPasswordSection = savePasswordBtn?.parentElement;

async function savePassword() {
  const oldPw = document.getElementById("settings-current-password").value || null;
  const newPw = document.getElementById("settings-new-password").value;
  const confirmPw = document.getElementById("settings-confirm-password").value;
  passwordMsg.style.color = "#f87171";
  if (!newPw) { passwordMsg.textContent = "New password cannot be empty."; return; }
  if (newPw !== confirmPw) { passwordMsg.textContent = "Passwords do not match."; return; }
  try {
    const res = await fetch("/api/auth/set-password", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ user_id: state.userId, old_password: oldPw, new_password: newPw }),
    });
    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      passwordMsg.textContent = data.detail || "Failed to update password.";
      return;
    }
    passwordMsg.style.color = "#4ade80";
    passwordMsg.textContent = "Password updated.";
    document.getElementById("settings-current-password").value = "";
    document.getElementById("settings-new-password").value = "";
    document.getElementById("settings-confirm-password").value = "";
    setTimeout(() => { passwordMsg.textContent = ""; }, 3000);
  } catch (e) {
    passwordMsg.textContent = "Network error.";
  }
}

savePasswordBtn.addEventListener("click", savePassword);

function clearStoredIdentity() {
  localStorage.removeItem("mat_userId");
  localStorage.removeItem("mat_displayName");
  localStorage.removeItem("mat_sessionId");
}

function applyLocalIdentity(resetSession = false) {
  state.deploymentMode = "local";
  state.userId = "user";
  state.displayName = "user";
  state.activeSessionUserId = "user";
  state.isAdmin = false;
  state.sessionReady = false;
  if (resetSession) {
    state.sessionId = `session-${Math.floor(Date.now() / 1000)}`;
    localStorage.setItem("mat_sessionId", state.sessionId);
  }
  localStorage.setItem("mat_deploymentMode", "local");
  localStorage.setItem("mat_userId", "user");
  localStorage.setItem("mat_displayName", "user");
}

function hideLocalAuthControls() {
  if (editUserBtn) editUserBtn.style.display = "none";
  if (logoutBtn) logoutBtn.style.display = "none";
  if (settingsLogoutBtn) settingsLogoutBtn.style.display = "none";
  if (settingsPasswordSection) settingsPasswordSection.style.display = "none";
}

// On load: in local mode force passwordless "user"; in server mode require server auth.
(async () => {
  let serverMode = "local";
  try {
    const healthResp = await fetch("/api/health");
    if (healthResp.ok) {
      const health = await healthResp.json();
      serverMode = health.mode || "local";
    }
  } catch (_) { /* server not up yet — assume local */ }

  state.deploymentMode = serverMode === "server" ? "server" : "local";
  const storedMode = localStorage.getItem("mat_deploymentMode") || "";
  const storedId = localStorage.getItem("mat_userId") || "";

  if (state.deploymentMode === "local") {
    hideLocalAuthControls();
    applyLocalIdentity(storedMode === "server" || (storedId && storedId !== "user"));
    hideLoginModal();
  } else if ((storedMode && storedMode !== "server") || (!storedMode && storedId === "user")) {
    clearStoredIdentity();
    showLoginModal();
    return;
  } else if (!storedId) {
    showLoginModal();
    return;
  } else if (!_isValidIdentity(storedId)) {
    // Legacy: localStorage contains a raw display name (non-"user"). Show login modal.
    showLoginModal();
    return;
  }

  sessionIdEl.textContent = state.sessionId;
  await refreshAccess();
  renderUserDisplay();
  await loadSessions();
  // Don't auto-restore previous session on page load — start fresh
  // User can click a session in the sidebar to switch to it
  localStorage.removeItem("mat_sessionId");
})();

// ---------------------------------------------------------------------------
// Session list management
// ---------------------------------------------------------------------------

async function loadSessions() {
  if (!state.userId) return;
  try {
    const resp = state.isAdmin
      ? await fetch(`/api/admin/sessions?user_id=${encodeURIComponent(state.userId)}`)
      : await fetch(`/api/users/${encodeURIComponent(state.userId)}/sessions`);
    if (!resp.ok) return;
    const sessions = await resp.json();
    renderSessionList(sessions);
  } catch (_) {
    // silently ignore — server may not be running yet
  }
}

function renderSessionList(sessions) {
  renderSessionList._lastSessions = sessions;
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
      const owner = s.userId || state.userId;
      const isActive = s.id === state.sessionId && owner === state.activeSessionUserId;
      li.className = "session-item" + (isActive ? " active" : "");
      li.dataset.owner = owner;

      const content = document.createElement("div");
      content.className = "session-item-content";

      const idLine = document.createElement("div");
      idLine.className = "session-item-id";
      idLine.textContent = state.isAdmin ? `${owner} / ${s.id}` : s.id;
      content.appendChild(idLine);

      const summary = s.summary || state.sessionSummaries[s.id];
      if (summary) {
        const summaryLine = document.createElement("div");
        summaryLine.className = "session-item-summary";
        summaryLine.textContent = summary;
        content.appendChild(summaryLine);
      }
      li.appendChild(content);

      const logBtn = document.createElement("button");
      logBtn.className = "session-item-log";
      logBtn.textContent = "LOG JSON";
      logBtn.title = "Download full session log";
      logBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        downloadSessionLog(s.id, owner);
      });
      li.appendChild(logBtn);

      const delBtn = document.createElement("button");
      delBtn.className = "session-item-delete";
      delBtn.textContent = "×";
      delBtn.title = "Delete session";
      delBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        deleteSession(s.id);
      });
      li.appendChild(delBtn);

      li.title = state.isAdmin ? `${owner} / ${s.id}` : s.id;
      li.addEventListener("click", () => switchSession(s.id, owner));
      sessionListEl.appendChild(li);
    });
}

async function switchSession(sessionId, owner = state.userId) {
  state.sessionId = sessionId;
  state.activeSessionUserId = owner;
  state.sessionReady = true;
  localStorage.setItem("mat_sessionId", sessionId);
  sessionIdEl.textContent = sessionId;
  renderSessionFilesTree([]);
  clearCurrentUploads();
  agentGraph.reset();
  planGraph.reset();
  hidePlanGraph();
  await loadSession(sessionId);
  await loadSessions();
  agentGraph.startPolling(sessionId);
  planGraph.startPolling(sessionId);
}

// ---------------------------------------------------------------------------
// Confirm dialog & session delete
// ---------------------------------------------------------------------------

function showConfirmDialog(message) {
  const existing = document.querySelector(".confirm-overlay");
  if (existing) existing.remove();

  return new Promise((resolve) => {
    let settled = false;
    const done = (result) => { if (settled) return; settled = true; overlay.remove(); resolve(result); };

    const overlay = document.createElement("div");
    overlay.className = "confirm-overlay";
    const msg = document.createElement("p");
    msg.className = "confirm-message";
    msg.innerHTML = message;

    const actions = document.createElement("div");
    actions.className = "confirm-actions";
    const cancelBtn = document.createElement("button");
    cancelBtn.className = "confirm-cancel";
    cancelBtn.textContent = "Cancel";
    cancelBtn.onclick = () => done(false);
    const okBtn = document.createElement("button");
    okBtn.className = "confirm-ok";
    okBtn.textContent = "Delete";
    okBtn.onclick = () => done(true);
    actions.append(cancelBtn, okBtn);

    const card = document.createElement("div");
    card.className = "confirm-card";
    card.append(msg, actions);
    overlay.appendChild(card);
    document.body.appendChild(overlay);

    overlay.addEventListener("click", (e) => { if (e.target === overlay) done(false); });
    overlay.addEventListener("keydown", (e) => { if (e.key === "Escape") done(false); });
    okBtn.focus();
  });
}

async function deleteSession(sessionId) {
  if (state.isSending) return;
  if (!await showConfirmDialog(`Delete session ${sessionId}? This cannot be undone.`)) return;
  try {
    const resp = await fetch(`/api/sessions/${encodeURIComponent(sessionId)}`, { method: "DELETE" });
    if (!resp.ok) return;
    if (sessionId === state.sessionId) {
      state.sessionId = `session-${Math.floor(Date.now() / 1000)}`;
      state.activeSessionUserId = state.userId;
      state.sessionReady = false;
      localStorage.setItem("mat_sessionId", state.sessionId);
      sessionIdEl.textContent = state.sessionId;
      chatArea.innerHTML = "";
      stepExecutionFeed.reset();
      renderSessionFilesTree([]);
      clearCurrentUploads();
      agentGraph.reset();
      planGraph.reset();
      hidePlanGraph();
    }
    await loadSessions();
  } catch (_) {
    // silently ignore
  }
}

refreshSessionsBtn.addEventListener("click", (e) => { e.stopPropagation(); loadSessions(); });

async function downloadSessionLog(sessionId, owner = state.userId) {
  if (!sessionId) return;
  const userQuery = owner || state.userId;
  const query = userQuery ? `?user_id=${encodeURIComponent(userQuery)}` : "";
  const url = `/api/sessions/${encodeURIComponent(sessionId)}/session-log${query}`;
  try {
    const resp = await fetch(url);
    if (!resp.ok) {
      const msg = await resp.text().catch(() => "");
      throw new Error(msg || `HTTP ${resp.status}`);
    }
    const blob = await resp.blob();
    const objectUrl = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = objectUrl;
    link.download = `matcreator-session-log-${sessionId}.json`;
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(objectUrl);
  } catch (err) {
    console.warn("Failed to download session log", err);
  }
}

document.getElementById("refresh-files").addEventListener("click", (e) => { e.stopPropagation(); refreshSessionFiles(); });

// ---------------------------------------------------------------------------
// File path → API URL conversion
// ---------------------------------------------------------------------------

function pathToApiUrl(path) {
  const sid = state.sessionId ? `&session_id=${encodeURIComponent(state.sessionId)}` : "";
  return `/api/workspace/files?path=${encodeURIComponent(path)}${sid}`;
}

// ---------------------------------------------------------------------------
// Chat helpers
// ---------------------------------------------------------------------------

// Post-process marked output: wrap ASCII art (box-drawing chars) in <pre>.
const BOX_RE = /[┌┐└┘├┤┬┴┼│━─]/;
function renderMarkdown(text) {
  if (!text) return "";
  let html = marked.parse(text);
  html = html.replace(/<pre><code>([\s\S]*?)<\/code><\/pre>/gi, (match, inner) => {
    const decoded = inner.replace(/&amp;/g, "&").replace(/&lt;/g, "<").replace(/&gt;/g, ">").replace(/&quot;/g, '"').replace(/&#39;/g, "'");
    return BOX_RE.test(decoded) ? `<pre class="ascii-art">${decoded}</pre>` : match;
  });
  html = html.replace(/<p>([\s\S]*?)<\/p>/gi, (match, inner) => {
    const decoded = inner.replace(/&amp;/g, "&").replace(/&lt;/g, "<").replace(/&gt;/g, ">").replace(/&quot;/g, '"').replace(/&#39;/g, "'");
    return BOX_RE.test(decoded) ? `<pre class="ascii-art">${decoded}</pre>` : match;
  });
  return html;
}

// Unescape common escape sequences in text content.
// Converts literal \n, \t, \r, \\ to actual characters.
function unescapeText(text) {
  if (!text) return "";
  return text
    .replace(/\\\\/g, "\x00")    // protect literal backslashes
    .replace(/\\n/g, "\n")
    .replace(/\\t/g, "\t")
    .replace(/\\r/g, "\r")
    .replace(/\\"/g, '"')
    .replace(/\x00/g, "\\");     // restore literal backslashes
}

// Cached character widths for wrap calculation (ASCII and CJK)
let _asciiWidth = 0;
let _cjkWidth = 0;
function getCharWidths() {
  if (_asciiWidth) return { ascii: _asciiWidth, cjk: _cjkWidth };
  const s = document.createElement("span");
  s.style.cssText = "position:absolute;visibility:hidden;font:14px 'Courier New',Consolas,monospace;white-space:pre;";
  document.body.appendChild(s);
  s.textContent = "x";
  _asciiWidth = s.getBoundingClientRect().width;
  s.textContent = "中";
  _cjkWidth = s.getBoundingClientRect().width;
  document.body.removeChild(s);
  return { ascii: _asciiWidth, cjk: _cjkWidth };
}

const CJK_RE = /[一-鿿㐀-䶿豈-﫿　-〿＀-￯]/;
function measureLine(line) {
  const { ascii, cjk } = getCharWidths();
  let w = 0;
  for (const ch of line) {
    w += CJK_RE.test(ch) ? cjk : ascii;
  }
  return w;
}

// Create a <pre class="json-block"> with unescaped content.
// Strips leading/trailing { } from JSON-like strings for cleaner display.
// Uses ResizeObserver to adapt wrap markers to container width.
function createJsonBlock(content) {
  const pre = document.createElement("pre");
  pre.className = "json-block";
  let rawText = unescapeText(content);
  rawText = rawText.replace(/^\{\s*/, "").replace(/\s*\}$/, "");
  pre.dataset.raw = rawText;
  applyWrapMarkers(pre);
  const ro = new ResizeObserver(() => applyWrapMarkers(pre));
  ro.observe(pre);
  return pre;
}

function applyWrapMarkers(pre) {
  const raw = pre.dataset.raw;
  if (!raw) return;
  const containerW = pre.clientWidth - 16; // subtract padding
  if (containerW <= 0) return;
  const { ascii, cjk } = getCharWidths();
  const markerW = ascii * 3; // " ↵" approx 3 ascii chars wide
  const lines = raw.split("\n");
  const out = [];
  for (const line of lines) {
    const lineW = measureLine(line);
    if (lineW <= containerW) {
      out.push(line);
    } else {
      // Split line by pixel width
      let w = 0, start = 0;
      for (let i = 0; i < line.length; i++) {
        const chW = CJK_RE.test(line[i]) ? cjk : ascii;
        if (w + chW > containerW - markerW) {
          out.push(line.slice(start, i) + " ↵");
          start = i;
          w = chW;
        } else {
          w += chW;
        }
      }
      if (start < line.length) out.push(line.slice(start));
    }
  }
  pre.textContent = out.join("\n");
}

const AGENT_AVATAR_SVG = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="rgba(125,211,252,0.9)" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
  <rect x="3" y="8" width="18" height="11" rx="2"/>
  <path d="M8 8V6a4 4 0 0 1 8 0v2"/>
  <circle cx="9" cy="14" r="1" fill="rgba(125,211,252,0.9)" stroke="none"/>
  <circle cx="15" cy="14" r="1" fill="rgba(125,211,252,0.9)" stroke="none"/>
  <path d="M7 19v2M17 19v2"/>
</svg>`;

const USER_AVATAR_SVG = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="rgba(168,85,247,0.9)" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
  <circle cx="12" cy="8" r="4"/>
  <path d="M4 20c0-4 3.6-7 8-7s8 3 8 7"/>
</svg>`;

function getUserAvatar() {
  return localStorage.getItem("user-avatar-url") || null;
}

function setUserAvatar(dataUrl) {
  localStorage.setItem("user-avatar-url", dataUrl);
  document.querySelectorAll(".user-avatar").forEach(applyUserAvatarToEl);
}

function applyUserAvatarToEl(el) {
  const url = getUserAvatar();
  el.innerHTML = url ? `<img src="${url}" alt="User">` : USER_AVATAR_SVG;
}

function createAgentAvatarEl() {
  const el = document.createElement("div");
  el.className = "message-avatar agent-avatar";
  el.innerHTML = AGENT_AVATAR_SVG;
  return el;
}

function createUserAvatarEl() {
  const el = document.createElement("div");
  el.className = "message-avatar user-avatar";
  applyUserAvatarToEl(el);
  return el;
}

function scrollToBottom() {
  chatArea.scrollTop = chatArea.scrollHeight;
}

function isChatNearBottom() {
  return chatArea.scrollHeight - chatArea.scrollTop - chatArea.clientHeight < 80;
}

function appendLiveTurnChild(container, child) {
  if (container === chatArea || !container?.dataset?.stepLiveRegion) {
    container.appendChild(child);
    return;
  }

  const firstStepCard = [...container.children].find((el) => el.dataset.stepStartTime !== undefined);
  if (firstStepCard) {
    container.insertBefore(child, firstStepCard);
  } else {
    container.appendChild(child);
  }
}

function addMessage(role, content, msgIndex, container = chatArea) {
  const div = document.createElement("div");
  div.className = `message ${role}-message`;
  if (msgIndex !== undefined) div.dataset.msgIndex = String(msgIndex);

  const avatar = role === "agent" ? createAgentAvatarEl() : createUserAvatarEl();
  div.appendChild(avatar);

  const bubble = document.createElement("div");
  bubble.className = "message-bubble";
  const inner = document.createElement("div");
  inner.className = "markdown-content";
  inner.innerHTML = renderMarkdown(content || "");
  bubble.appendChild(inner);
  div.appendChild(bubble);

  appendLiveTurnChild(container, div);
  scrollToBottom();
  return div;
}

function getFunctionCall(part) {
  return part?.functionCall || part?.function_call || null;
}

function getFunctionResponse(part) {
  return part?.functionResponse || part?.function_response || null;
}

function getPlotPaths(response) {
  const paths = [];
  const add = (path) => {
    if (typeof path === "string" && path && !paths.includes(path)) paths.push(path);
  };
  add(response?.plot_path);
  if (Array.isArray(response?.plot_paths)) {
    response.plot_paths.forEach(add);
  }
  return paths;
}

function getStructurePaths(payload) {
  const paths = [];
  const add = (path) => {
    if (typeof path === "string" && path && !paths.includes(path)) paths.push(path);
  };
  const visit = (value, key = "") => {
    if (!value) return;
    if (key === "structure_path") {
      add(value);
      return;
    }
    if (key === "structure_paths" && Array.isArray(value)) {
      value.forEach(add);
      return;
    }
    if ((key === "artifacts" || key === "artifact_paths") && Array.isArray(value)) {
      value.forEach((path) => {
        if (classifyPath(String(path)) === "structure") add(path);
      });
      return;
    }
    if (Array.isArray(value)) {
      value.forEach((item) => visit(item, key));
      return;
    }
    if (typeof value === "object") {
      Object.entries(value).forEach(([childKey, childValue]) => visit(childValue, childKey));
    }
  };
  visit(payload);
  return paths;
}

function createStructureViewButton(path) {
  const btn = document.createElement("button");
  btn.className = "ghost structure-view-btn";
  btn.textContent = `🔬 View: ${path.split("/").pop()}`;
  btn.addEventListener("click", () => openViewer({ path, url: pathToApiUrl(path) }));
  return btn;
}

function createArtifactListItem(path) {
  const li = document.createElement("li");
  li.title = path;
  if (classifyPath(path) === "structure") {
    li.appendChild(createStructureViewButton(path));
  } else {
    li.textContent = path.split("/").pop();
  }
  return li;
}

function isExecutorLauncherTool(name) {
  return ["run_flash_step", "run_node_executor", "run_sub_agent"].includes(name || "");
}

// Render a typed timeline array into a container element, mirroring
// Streamlit's render_stream_timeline: thoughts and tool calls go into
// collapsible <details> blocks; text parts render as markdown;
// plot_path responses render as inline images.
function renderTimeline(container, timeline, shownPlotPaths = null) {
  container.innerHTML = "";
  const containerPlotPaths = container._plotPaths || new Set();
  const visiblePlotPaths = new Set();
  for (const item of timeline) {
    if (item.type === "thought") {
      const details = document.createElement("details");
      details.className = "timeline-thought";
      const summary = document.createElement("summary");
      summary.textContent = "🤔 Thinking...";
      details.appendChild(summary);
      const body = document.createElement("div");
      body.className = "markdown-content";
      body.innerHTML = renderMarkdown(item.text || "");
      details.appendChild(body);
      container.appendChild(details);
    } else if (item.type === "function_call") {
      const details = document.createElement("details");
      details.className = "timeline-function-call";
      if (isExecutorLauncherTool(item.name)) details.open = true;
      const summary = document.createElement("summary");
      summary.innerHTML = `<span class="timeline-badge badge-in">IN</span> ${item.name}`;
      details.appendChild(summary);
      details.appendChild(createJsonBlock(JSON.stringify(item.args, null, 2)));
      container.appendChild(details);
      if (isExecutorLauncherTool(item.name)) {
        const inlineHost = document.createElement("div");
        inlineHost.className = "step-feed-inline-region";
        inlineHost.dataset.stepInlineHost = item.name;
        container.appendChild(inlineHost);
        if (Array.isArray(item.stepNodes) && item.stepNodes.length) {
          item.stepNodes.forEach((node) => stepExecutionFeed.appendStatic(node, inlineHost));
        } else if (state.isSending) {
          stepExecutionFeed.attachLiveToolHost(inlineHost);
        }
      }
    } else if (item.type === "function_response") {
      const details = document.createElement("details");
      details.className = "timeline-function-response";
      const summary = document.createElement("summary");
      summary.innerHTML = `<span class="timeline-badge badge-out">OUT</span> ${item.name}`;
      details.appendChild(summary);
      details.appendChild(createJsonBlock(JSON.stringify(item.response, null, 2)));
      container.appendChild(details);
      for (const plotPath of getPlotPaths(item.response)) {
        if (
          visiblePlotPaths.has(plotPath) ||
          (shownPlotPaths && shownPlotPaths.has(plotPath) && !containerPlotPaths.has(plotPath))
        ) {
          continue;
        }
        visiblePlotPaths.add(plotPath);
        const img = document.createElement("img");
        img.src = pathToApiUrl(plotPath);
        img.className = "timeline-image";
        img.alt = plotPath.split("/").pop();
        img.style.cursor = "zoom-in";
        img.addEventListener("click", () => lightbox.open(img.src));
        container.appendChild(img);
      }
      getStructurePaths(item.response).forEach((path) => {
        container.appendChild(createStructureViewButton(path));
      });
    } else if (item.type === "text") {
      const div = document.createElement("div");
      div.className = "markdown-content";
      div.innerHTML = renderMarkdown(item.text || "");
      container.appendChild(div);
    }
  }
  container._plotPaths = visiblePlotPaths;
  visiblePlotPaths.forEach((path) => shownPlotPaths?.add(path));
  scrollToBottom();
}

// Create an agent message div with an inner timeline container, append to
// chatArea, and return the inner container for live updates.
function addAgentTimelineMessage(timeline, shownPlotPaths = null, msgIndex, container = chatArea) {
  const outer = document.createElement("div");
  outer.className = "message agent-message";
  if (msgIndex !== undefined) outer.dataset.msgIndex = String(msgIndex);
  outer.appendChild(createAgentAvatarEl());
  const bubble = document.createElement("div");
  bubble.className = "message-bubble";
  const inner = document.createElement("div");
  inner.className = "timeline-container";
  bubble.appendChild(inner);
  outer.appendChild(bubble);
  appendLiveTurnChild(container, outer);
  renderTimeline(inner, timeline, shownPlotPaths);
  return inner;
}

function formatStepDuration(node) {
  if (!node.start_time) return "—";
  if (!node.end_time) return "running…";
  const secs = ((new Date(node.end_time) - new Date(node.start_time)) / 1000).toFixed(1);
  return `${secs}s`;
}

function stepFeedTitle(node) {
  const input = node.input || {};
  const label = node.label || input.node_id || node.id;
  const action = input.action ? ` — ${input.action}` : "";
  return `${label}${action}`;
}

function renderStepInput(input) {
  const details = document.createElement("details");
  details.className = "step-feed-nested";
  const summary = document.createElement("summary");
  summary.textContent = "Input";
  details.appendChild(summary);
  details.appendChild(createJsonBlock(JSON.stringify(input, null, 2)));
  return details;
}

function renderStepConversationEvent(evt) {
  const details = document.createElement("details");
  details.className = `timeline-${evt.type} step-feed-nested`;
  const summary = document.createElement("summary");
  const icon = evt.type === "thought" ? "💭" : evt.type === "text" ? "💬" : evt.type === "function_call" ? "🔧" : "↩";
  summary.textContent = `${icon} [${evt.author || "step_executor"}] ${evt.type || "event"}`;
  details.appendChild(summary);
  details.appendChild(createJsonBlock(evt.content));
  return details;
}

function renderStepToolCall(tc) {
  const details = document.createElement("details");
  details.className = "timeline-function-call step-feed-nested";
  const dur = tc.start_time && tc.end_time
    ? ` (${((new Date(tc.end_time) - new Date(tc.start_time)) / 1000).toFixed(1)}s)`
    : "";
  const summary = document.createElement("summary");
  summary.textContent = `🔧 ${tc.name || "tool"}${dur}`;
  details.appendChild(summary);
  if (tc.args_summary) {
    details.appendChild(createJsonBlock(tc.args_summary));
  }
  if (tc.result_summary) {
    const pre = createJsonBlock(`→ ${tc.result_summary}`);
    pre.style.borderTop = "1px solid rgba(255,255,255,0.06)";
    details.appendChild(pre);
  }
  getStructurePaths(tc).forEach((path) => {
    details.appendChild(createStructureViewButton(path));
  });
  return details;
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

// ---------------------------------------------------------------------------
// Session files tree
// ---------------------------------------------------------------------------

function formatFileSize(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function _createFileItem(f) {
  const li = document.createElement("li");
  li.className = "tree-file";

  const nameSpan = document.createElement("span");
  nameSpan.className = "tree-filename";
  nameSpan.textContent = f.relname;
  li.appendChild(nameSpan);

  const sizeSpan = document.createElement("span");
  sizeSpan.className = "tree-filesize";
  sizeSpan.textContent = formatFileSize(f.size);
  li.appendChild(sizeSpan);

  const actions = document.createElement("div");
  actions.className = "tree-actions";

  const dlLink = document.createElement("a");
  dlLink.href = `/api/workspace/files?path=${encodeURIComponent(f.path)}`;
  dlLink.download = f.relname;
  dlLink.className = "tree-btn";
  dlLink.title = "Download";
  dlLink.textContent = "↓";
  actions.appendChild(dlLink);

  if (classifyPath(f.path) === "structure") {
    const viewBtn = document.createElement("button");
    viewBtn.className = "tree-btn";
    viewBtn.title = "View 3D";
    viewBtn.textContent = "⬡";
    viewBtn.addEventListener("click", () =>
      openViewer({ path: f.path, name: f.relname, url: pathToApiUrl(f.path) })
    );
    actions.appendChild(viewBtn);
  } else {
    const viewBtn = document.createElement("button");
    viewBtn.className = "tree-btn";
    viewBtn.title = "View";
    viewBtn.textContent = "👁";
    viewBtn.addEventListener("click", () =>
      openFileViewer({ path: f.path, name: f.relname })
    );
    actions.appendChild(viewBtn);
  }

  li.appendChild(actions);
  return li;
}

function _buildFileTree(files, prefix) {
  const root = { children: {}, files: [] };
  for (const file of files) {
    const rel = file.path.slice(prefix.length).replace(/^\//, "");
    const parts = rel.split("/");
    const filename = parts[parts.length - 1];
    const dirs = parts.slice(0, -1);
    let node = root;
    for (const dir of dirs) {
      if (!node.children[dir]) {
        node.children[dir] = { name: dir, children: {}, files: [] };
      }
      node = node.children[dir];
    }
    node.files.push({ ...file, relname: filename, relpath: rel });
  }
  return root;
}

function _renderTreeNode(node, container, depth) {
  const sortedDirs = Object.keys(node.children).sort();
  const sortedFiles = node.files.slice().sort((a, b) => a.relname.localeCompare(b.relname));

  for (const dirName of sortedDirs) {
    const child = node.children[dirName];
    const li = document.createElement("li");
    li.className = "tree-dir-node";

    const details = document.createElement("details");

    const summary = document.createElement("summary");
    summary.className = "tree-dir-summary";
    summary.textContent = dirName + "/";
    details.appendChild(summary);

    const childUl = document.createElement("ul");
    childUl.className = "tree-dir-children";
    _renderTreeNode(child, childUl, depth + 1);
    details.appendChild(childUl);

    li.appendChild(details);
    container.appendChild(li);
  }

  for (const f of sortedFiles) {
    container.appendChild(_createFileItem(f));
  }
}

function renderSessionFilesTree(files) {
  const ul = document.getElementById("session-files-tree");
  ul.innerHTML = "";
  if (!files.length) {
    const li = document.createElement("li");
    li.className = "empty";
    li.textContent = "No files yet";
    ul.appendChild(li);
    return;
  }

  let prefix = "";
  const sidIdx = files[0].path.indexOf(state.sessionId);
  if (sidIdx >= 0) {
    prefix = files[0].path.slice(0, sidIdx + state.sessionId.length);
  } else {
    let common = files[0].path;
    for (const f of files) {
      let i = 0;
      while (i < common.length && i < f.path.length && common[i] === f.path[i]) i++;
      common = common.slice(0, i);
    }
    prefix = common.slice(0, common.lastIndexOf("/") + 1);
  }

  const root = _buildFileTree(files, prefix);
  _renderTreeNode(root, ul, 0);
}

function setWorkspaceCliOpen(open) {
  workspaceCli?.classList.toggle("hidden", !open);
  workspaceCliToggle?.classList.toggle("is-active", open);
  workspaceCliToggle?.setAttribute("aria-expanded", String(open));
  if (open) {
    startWorkspaceTerminal();
  } else {
    stopWorkspaceTerminal();
  }
}

function terminalWebSocketUrl() {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const params = new URLSearchParams();
  if (state.deploymentMode === "server" && state.userId) params.set("user_id", state.userId);
  const qs = params.toString();
  return `${protocol}//${window.location.host}/api/workspace/terminal${qs ? `?${qs}` : ""}`;
}

function resizeWorkspaceTerminal() {
  if (!workspaceTerminal || !workspaceTerminalFit || !workspaceTerminalSocket) return;
  try {
    workspaceTerminalFit.fit();
    if (workspaceTerminalSocket.readyState === WebSocket.OPEN) {
      workspaceTerminalSocket.send(JSON.stringify({
        type: "resize",
        rows: workspaceTerminal.rows,
        cols: workspaceTerminal.cols,
      }));
    }
  } catch (_) { /* terminal may not be visible yet */ }
}

function startWorkspaceTerminal() {
  if (!workspaceTerminalEl) return;
  if (workspaceTerminalSocket && workspaceTerminalSocket.readyState === WebSocket.OPEN) {
    workspaceTerminal?.focus();
    resizeWorkspaceTerminal();
    return;
  }
  workspaceTerminalEl.innerHTML = "";
  workspaceTerminal = new Terminal({
    cursorBlink: true,
    convertEol: true,
    fontFamily: 'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace',
    fontSize: 12,
    theme: {
      background: "#030712",
      foreground: "#d1fae5",
      cursor: "#7dd3fc",
      selectionBackground: "#1e40af88",
    },
  });
  workspaceTerminalFit = new FitAddon();
  workspaceTerminal.loadAddon(workspaceTerminalFit);
  workspaceTerminal.open(workspaceTerminalEl);
  workspaceTerminal.write("\r\nStarting workspace terminal...\r\n");
  workspaceTerminalFit.fit();
  workspaceTerminal.focus();

  workspaceTerminalSocket = new WebSocket(terminalWebSocketUrl());
  workspaceTerminalSocket.addEventListener("open", () => {
    resizeWorkspaceTerminal();
  });
  workspaceTerminalSocket.addEventListener("message", (event) => {
    try {
      const message = JSON.parse(event.data);
      if (message.type === "output") workspaceTerminal.write(message.data || "");
    } catch (_) {
      workspaceTerminal.write(String(event.data || ""));
    }
  });
  workspaceTerminalSocket.addEventListener("close", () => {
    workspaceTerminal?.write("\r\n[terminal closed]\r\n");
  });
  workspaceTerminalSocket.addEventListener("error", () => {
    workspaceTerminal?.write("\r\n[terminal connection error]\r\n");
  });
  workspaceTerminal.onData((data) => {
    if (workspaceTerminalSocket?.readyState === WebSocket.OPEN) {
      workspaceTerminalSocket.send(JSON.stringify({ type: "input", data }));
    }
  });
}

function stopWorkspaceTerminal() {
  if (workspaceTerminalSocket) {
    workspaceTerminalSocket.close();
    workspaceTerminalSocket = null;
  }
  workspaceTerminal?.dispose();
  workspaceTerminal = null;
  workspaceTerminalFit = null;
  if (workspaceTerminalEl) workspaceTerminalEl.innerHTML = "";
}

workspaceCliToggle?.addEventListener("click", () => {
  setWorkspaceCliOpen(workspaceCli?.classList.contains("hidden"));
});

window.addEventListener("resize", resizeWorkspaceTerminal);

async function refreshSessionFiles() {
  if (!state.sessionId || !state.sessionReady) return;
  try {
    const resp = await fetch(`/api/sessions/${state.sessionId}/files`);
    if (!resp.ok) return;
    const data = await resp.json();
    renderSessionFilesTree(data.files || []);
  } catch (_) {}
}

function setUploadStatus(message, tone = "idle") {
  if (!uploadStatus) return;
  uploadStatus.textContent = message || "";
  uploadStatus.className = `upload-status upload-status-${tone}`;
}

function renderCurrentUploadChips() {
  if (!uploadStatus) return;
  uploadStatus.innerHTML = "";
  uploadStatus.className = "upload-status upload-file-list";
  if (!state.currentUploads.length) return;

  state.currentUploads.forEach((file) => {
    const chip = document.createElement("span");
    chip.className = "upload-file-chip";

    const name = document.createElement("span");
    name.className = "upload-file-name";
    name.textContent = file.name;
    name.title = file.path;
    chip.appendChild(name);

    const removeBtn = document.createElement("button");
    removeBtn.className = "upload-file-remove";
    removeBtn.type = "button";
    removeBtn.title = "Delete uploaded file";
    removeBtn.textContent = "×";
    removeBtn.addEventListener("click", () => deleteUploadedFile(file));
    chip.appendChild(removeBtn);

    uploadStatus.appendChild(chip);
  });
}

function clearCurrentUploads() {
  state.currentUploads = [];
  renderCurrentUploadChips();
}

function mergeUploadedFiles(existingFiles, newFiles) {
  const merged = [...existingFiles];
  const seenPaths = new Set(existingFiles.map((file) => file?.path).filter(Boolean));

  newFiles.forEach((file) => {
    const path = file?.path;
    if (path && seenPaths.has(path)) return;
    if (path) seenPaths.add(path);
    merged.push(file);
  });

  return merged;
}

function sessionRelativeUploadPath(file) {
  const normalized = String(file?.path || "").replaceAll("\\", "/");
  const marker = `/${state.sessionId}/`;
  const markerIdx = normalized.indexOf(marker);
  if (markerIdx >= 0) return normalized.slice(markerIdx + marker.length);
  return file?.name ? `uploads/${file.name}` : normalized;
}

function messageWithUploadContext(message, uploads) {
  if (!uploads.length) return message;
  const fileLines = uploads.map((file) => {
    const relPath = sessionRelativeUploadPath(file);
    return `- ${file.name}: ${relPath} (absolute path: ${file.path})`;
  });
  return [
    message,
    "",
    "The user uploaded the following file(s) for this message. They are saved in the current session workspace. Use these paths when inspecting or processing the files:",
    ...fileLines,
  ].join("\n");
}

function formatUploadNames(uploadNames) {
  if (!uploadNames.length) return "";
  return `Attached: ${uploadNames.map((name) => `\`${name}\``).join(", ")}`;
}

function messageWithUploadNames(message, uploads) {
  const uploadNames = uploads.map((file) => file.name).filter(Boolean);
  const suffix = formatUploadNames(uploadNames);
  return suffix ? `${message}\n\n${suffix}` : message;
}

function displayMessageFromStoredUserText(message) {
  const marker = "\n\nThe user uploaded the following file(s) for this message.";
  const rawMessage = String(message || "");
  const markerIdx = rawMessage.indexOf(marker);
  if (markerIdx < 0) return rawMessage;

  const visibleMessage = rawMessage.slice(0, markerIdx);
  const hiddenContext = rawMessage.slice(markerIdx);
  const uploadNames = hiddenContext
    .split("\n")
    .map((line) => line.match(/^-\s+([^:]+):/)?.[1]?.trim())
    .filter(Boolean);
  const suffix = formatUploadNames(uploadNames);
  return suffix ? `${visibleMessage}\n\n${suffix}` : visibleMessage;
}

async function deleteUploadedFile(file) {
  if (!file?.path || !state.sessionId) return;
  try {
    const resp = await fetch(
      `/api/sessions/${encodeURIComponent(state.sessionId)}/files?path=${encodeURIComponent(file.path)}`,
      { method: "DELETE" }
    );
    if (!resp.ok) {
      const detail = await resp.text();
      throw new Error(detail || `HTTP ${resp.status}`);
    }
    state.currentUploads = state.currentUploads.filter((item) => item.path !== file.path);
    renderCurrentUploadChips();
    await refreshSessionFiles();
  } catch (err) {
    setUploadStatus(`Delete failed: ${err.message || err}`, "error");
  }
}

async function uploadFilesToSession(fileList) {
  const files = Array.from(fileList || []);
  if (!files.length) return;
  if (!state.userId) { showLoginModal(); return; }
  if (!canWriteActiveSession()) {
    addMessage("agent", `Admin view is read-only for ${state.activeSessionUserId}'s session.`);
    return;
  }

  if (!state.sessionReady) await createSession();
  if (!state.sessionReady) {
    setUploadStatus("Could not create session.", "error");
    return;
  }

  if (fileUploadBtn) fileUploadBtn.disabled = true;
  const uploaded = [];
  try {
    for (const file of files) {
      setUploadStatus(`Uploading ${file.name}...`, "busy");
      const formData = new FormData();
      formData.append("file", file);
      const resp = await fetch(`/api/sessions/${encodeURIComponent(state.sessionId)}/files`, {
        method: "POST",
        body: formData,
      });
      if (!resp.ok) {
        const detail = await resp.text();
        throw new Error(detail || `HTTP ${resp.status}`);
      }
      uploaded.push(await resp.json());
    }

    await refreshSessionFiles();
    state.currentUploads = mergeUploadedFiles(state.currentUploads, uploaded);
    renderCurrentUploadChips();
  } catch (err) {
    setUploadStatus(`Upload failed: ${err.message || err}`, "error");
  } finally {
    if (fileUploadBtn) fileUploadBtn.disabled = false;
    fileUploadInput.value = "";
  }
}

// ---------------------------------------------------------------------------
// Session summary (experimental)
// ---------------------------------------------------------------------------

function renderSessionBanner(summary) {
  if (!sessionSummaryHeader || !sessionSummaryText) return;
  const isVisible = sessionSummaryHeader.classList.contains("visible");
  if (summary) {
    const isFirstShow = !isVisible || sessionSummaryText.classList.contains("session-summary-placeholder");
    sessionSummaryText.textContent = summary;
    sessionSummaryText.classList.remove("session-summary-placeholder");
    if (isFirstShow) {
      sessionSummaryText.style.opacity = "0";
      sessionSummaryText.classList.remove("typewriter", "typewriter-done");
      sessionSummaryHeader.classList.add("visible");
      setTimeout(() => {
        runTypewriter(sessionSummaryText, summary);
      }, 700);
    }
  } else {
    sessionSummaryText.textContent = "";
    sessionSummaryText.classList.remove("session-summary-placeholder", "typewriter", "typewriter-done");
    sessionSummaryHeader.classList.remove("visible");
  }
}

function runTypewriter(el, text) {
  el.classList.remove("typewriter", "typewriter-done");
  el.style.opacity = "";
  el.style.maxWidth = "none";
  el.textContent = text;
  const fullW = el.scrollWidth;
  el.style.maxWidth = "";
  void el.offsetWidth;
  const len = [...text].length;
  el.style.setProperty("--tw-steps", len);
  el.style.setProperty("--tw-width", fullW + "px");
  el.textContent = text;
  el.classList.add("typewriter");
  el.addEventListener("animationend", function onEnd() {
    el.removeEventListener("animationend", onEnd);
    el.classList.remove("typewriter");
    el.classList.add("typewriter-done");
    el.style.removeProperty("--tw-steps");
    el.style.removeProperty("--tw-width");
  });
}

function startSummaryEdit() {
  if (!sessionSummaryText || !sessionSummaryHeader || sessionSummaryHeader.querySelector("input")) return;
  const isPlaceholder = sessionSummaryText.classList.contains("session-summary-placeholder");
  const original = isPlaceholder ? "" : sessionSummaryText.textContent;
  const input = document.createElement("input");
  input.type = "text";
  input.value = original;
  input.className = "session-summary-input";
  input.maxLength = 60;
  input.placeholder = "Enter session summary…";
  sessionSummaryText.style.display = "none";
  sessionSummaryHeader.insertBefore(input, sessionSummaryEdit);
  input.focus();
  input.select();

  const finish = async (save) => {
    const newValue = input.value.trim();
    input.remove();
    sessionSummaryText.style.display = "";
    if (save && newValue !== original) {
      if (newValue) {
        sessionSummaryText.textContent = newValue;
        sessionSummaryText.classList.remove("session-summary-placeholder");
        state.sessionSummaries[state.sessionId] = newValue;
        await saveSessionSummary(state.sessionId, newValue);
      } else {
        sessionSummaryText.textContent = "New Session";
        sessionSummaryText.classList.add("session-summary-placeholder");
        delete state.sessionSummaries[state.sessionId];
        await saveSessionSummary(state.sessionId, "");
      }
      renderSessionList._lastSessions && renderSessionList(renderSessionList._lastSessions);
    } else if (!save && isPlaceholder) {
      // Restore placeholder on cancel
      sessionSummaryText.textContent = "New Session";
      sessionSummaryText.classList.add("session-summary-placeholder");
    }
  };

  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter") { e.preventDefault(); finish(true); }
    else if (e.key === "Escape") { finish(false); }
  });
  input.addEventListener("blur", () => finish(true));
}

if (sessionSummaryEdit) {
  sessionSummaryEdit.addEventListener("click", startSummaryEdit);
}

async function saveSessionSummary(sessionId, summary) {
  try {
    await fetch(`/api/sessions/${encodeURIComponent(sessionId)}/summary`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ summary }),
    });
  } catch (_) {
    // silently ignore
  }
}

async function generateSessionSummary(sessionId) {
  try {
    const resp = await fetch(`/api/sessions/${encodeURIComponent(sessionId)}/summarize`, {
      method: "POST",
    });
    if (!resp.ok) return;
    const data = await resp.json();
    if (data.summary) {
      state.sessionSummaries[sessionId] = data.summary;
      // Only update banner if user is still on this session
      if (sessionId === state.sessionId) {
        renderSessionBanner(data.summary);
      }
      // Refresh session list to show summary
      renderSessionList._lastSessions && renderSessionList(renderSessionList._lastSessions);
    }
  } catch (_) {
    // silently ignore — summary is non-critical
  }
}

// ---------------------------------------------------------------------------
// Session management
// ---------------------------------------------------------------------------

async function createSession() {
  state.activeSessionUserId = state.userId;
  const url = `/apps/${APP_NAME}/users/${activeSessionBackendUserId()}/sessions/${state.sessionId}`;
  const defaultWorkdir = (state.defaultWorkdir || "").trim();
  const sessionWorkdir = state.customWorkdir || defaultWorkdir;
  try {
    const resp = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        ...(state.agentMode !== "normal" ? { agent_mode: state.agentMode } : {}),
        ...(state.agentMode === "bench" ? { benchmark_mode: true } : {}),
        ...(sessionWorkdir ? { custom_workdir: sessionWorkdir } : {}),
      }),
    });
    if (!resp.ok) {
      console.error(`Failed to create session: HTTP ${resp.status}`, await resp.text());
      return;
    }
    state.sessionReady = true;
    await startKnowledgeReview();
    await loadSessions();
  } catch (err) {
    console.error("Failed to create session:", err);
  }
}

function renderKnowledgeReviewStatus(review) {
  const status = review?.status || "idle";
  const running = status === "running";
  const progress = review?.progress || {};
  const phase = review?.phase || "memory";
  const completed = progress.completed || 0;
  const total = progress.total || 0;
  const results = Array.isArray(review?.results) ? review.results : [];
  const errors = Array.isArray(review?.errors) ? review.errors : [];
  if (knowledgeReviewBanner) {
    knowledgeReviewBanner.disabled = running;
    knowledgeReviewBanner.className = `knowledge-review-banner status-${status}`;
    const detail = review?.summary || errors[0];
    knowledgeReviewBanner.title = detail
      ? `${detail}${running ? "" : " Click to review memory and graph nodes."}`
      : running
        ? "Knowledge review is running"
        : "Click to review memory and graph nodes";
  }
  knowledgeReviewSpinner?.classList.toggle("hidden", !running);
  if (knowledgeReviewText) {
    if (running) {
      const phaseLabel = phase === "graph" ? "graph nodes" : "memory";
      knowledgeReviewText.textContent = total
        ? `Reviewing ${phaseLabel}: ${completed}/${total} (${progress.percent || 0}%)`
        : `Starting ${phaseLabel} review`;
    } else if (status === "failed") {
      knowledgeReviewText.textContent = `Review failed: ${errors[0] || "unknown error"} · click to retry`;
    } else if (status === "completed" || status === "completed_with_errors") {
      const memoryCount = results.filter((item) => item.phase === "memory").length;
      const graphCount = results.filter((item) => item.phase === "graph").length;
      const warning = errors.length ? `, ${errors.length} errors` : "";
      const summary = review?.summary?.trim();
      if (memoryCount === 0 && graphCount === 0 && summary) {
        knowledgeReviewText.textContent = `${summary}${warning} · click to run again`;
      } else {
        knowledgeReviewText.textContent = `Review complete: ${memoryCount} memory, ${graphCount} graph actions${warning} · click to run again`;
      }
    } else {
      knowledgeReviewText.textContent = "Review memory and graph · click to start";
    }
  }
  if (!running && knowledgeReviewPoll) {
    clearInterval(knowledgeReviewPoll);
    knowledgeReviewPoll = null;
  }
}

async function refreshKnowledgeReviewStatus() {
  try {
    const resp = await fetch("/api/knowledge-review/status");
    if (!resp.ok) return;
    const review = await resp.json();
    renderKnowledgeReviewStatus(review);
    if (review.status === "running" && !knowledgeReviewPoll) {
      knowledgeReviewPoll = setInterval(refreshKnowledgeReviewStatus, 2000);
    }
  } catch (_) {
    // The banner is informational; session work should continue if polling fails.
  }
}

async function startKnowledgeReview() {
  renderKnowledgeReviewStatus({
    status: "running",
    phase: "memory",
    message: "Starting Know-Do Graph review.",
  });
  try {
    const resp = await fetch("/api/knowledge-review/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: state.sessionId }),
    });
    const review = await resp.json().catch(() => ({}));
    if (!resp.ok) {
      renderKnowledgeReviewStatus({
        status: "failed",
        errors: [review.detail || `HTTP ${resp.status}`],
      });
      return;
    }
    renderKnowledgeReviewStatus(review);
    if (!knowledgeReviewPoll) {
      knowledgeReviewPoll = setInterval(refreshKnowledgeReviewStatus, 2000);
    }
  } catch (_) {
    renderKnowledgeReviewStatus({
      status: "failed",
      errors: ["Could not reach the review service"],
    });
  }
}

knowledgeReviewBanner?.addEventListener("click", () => {
  if (!knowledgeReviewBanner.disabled) startKnowledgeReview();
});

async function patchSessionAgentMode(mode) {
  if (!state.sessionReady || !state.sessionId) return;
  const url = `/apps/${APP_NAME}/users/${encodeURIComponent(activeSessionBackendUserId())}/sessions/${encodeURIComponent(state.sessionId)}`;
  try {
    const delta = { agent_mode: mode, benchmark_mode: mode === "bench" };
    const resp = await fetch(url, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ state_delta: delta }),
    });
    if (!resp.ok) console.error(`Failed to patch agent mode: HTTP ${resp.status}`);
  } catch (err) {
    console.error("Failed to patch agent mode:", err);
  }
}

async function fetchSessionData(sessionId) {
  const owner = state.activeSessionUserId || state.userId;
  const resp = await fetch(
    `/api/users/${encodeURIComponent(owner)}/sessions/${encodeURIComponent(sessionId)}`,
    { headers: { "Content-Type": "application/json" } }
  );
  if (!resp.ok) return null;
  return resp.json();
}

async function fetchSessionStepNodes(sessionId) {
  try {
    const graphResp = await fetch(`/api/agent-graph/${encodeURIComponent(sessionId)}`);
    if (!graphResp.ok) return [];
    const graphData = await graphResp.json();
    return Object.values(graphData.nodes || {})
      .filter((node) => node.type === "step")
      .sort((a, b) => stepNodeTimestamp(a) - stepNodeTimestamp(b));
  } catch (_) {
    return [];
  }
}

function eventTimestamp(event, fallbackOrder) {
  if (event.timestamp) {
    const raw = Number(event.timestamp);
    return raw < 1e12 ? raw * 1000 : raw;
  }
  if (event.createTime) return new Date(event.createTime).getTime();
  return fallbackOrder;
}

function stepNodeTimestamp(node) {
  return node.start_time ? new Date(node.start_time).getTime() : Infinity;
}

function buildSessionTimeline(events, stepNodes) {
  const timeline = [];
  events.forEach((event, idx) => {
    timeline.push({ type: "event", data: event, ts: eventTimestamp(event, idx), order: idx });
  });
  stepNodes.forEach((node, idx) => {
    timeline.push({ type: "step", data: node, ts: stepNodeTimestamp(node), order: 1e9 + idx });
  });
  timeline.sort((a, b) => a.ts - b.ts || a.order - b.order);
  return timeline;
}

function assignStepNodesToExecutorCalls(timeline, pendingStepNodes) {
  for (const item of timeline) {
    if (item.type !== "function_call" || !isExecutorLauncherTool(item.name)) continue;
    const nextStep = pendingStepNodes.shift();
    if (nextStep) item.stepNodes = [nextStep];
  }
  return timeline;
}

function collectFunctionResponsesById(events) {
  const frById = {};
  for (const event of events) {
    for (const part of (event.content?.parts || [])) {
      const fr = getFunctionResponse(part);
      if (fr?.id) frById[fr.id] = fr;
    }
  }
  return frById;
}

function eventToTimelineParts(event, frById, pairedResponseIds = new Set()) {
  const parts = event.content?.parts || [];
  const evtTimeline = [];
  let accText = "";

  for (const part of parts) {
    if (part.thought) {
      evtTimeline.push({ type: "thought", text: part.text || "" });
    } else if (getFunctionCall(part)) {
      const fc = getFunctionCall(part);
      const matchedFr = frById[fc.id];
      evtTimeline.push({
        type: "function_call",
        id: fc.id,
        name: fc.name || "Unknown",
        args: fc.args || {},
      });
      if (matchedFr) {
        if (matchedFr.id) pairedResponseIds.add(matchedFr.id);
        evtTimeline.push({
          type: "function_response",
          id: matchedFr.id,
          name: matchedFr.name || "Unknown",
          response: matchedFr.response || {},
        });
      }
    } else if (getFunctionResponse(part)) {
      const fr = getFunctionResponse(part);
      if (fr.id && pairedResponseIds.has(fr.id)) continue;
      const alreadyMatched = evtTimeline.some(
        (item) => item.type === "function_response" && item.id === fr.id
      );
      if (!alreadyMatched) {
        evtTimeline.push({
          type: "function_response",
          id: fr.id,
          name: fr.name || "Unknown",
          response: fr.response || {},
        });
      }
    } else if (part.text && !part.thought) {
      accText += part.text;
      const last = evtTimeline[evtTimeline.length - 1];
      if (last?.type === "text") {
        last.text = accText;
      } else {
        evtTimeline.push({ type: "text", text: accText });
      }
    }
  }

  return evtTimeline;
}

function renderSessionTimeline(events, stepNodes) {
  chatArea.innerHTML = "";
  stepExecutionFeed.reset();
  stepExecutionFeed.setHierarchy(stepNodes || []);

  const sortedEvents = (events || [])
    .map((event, idx) => ({ event, ts: eventTimestamp(event, idx), order: idx }))
    .sort((a, b) => a.ts - b.ts || a.order - b.order)
    .map((item) => item.event);
  const pendingStepNodes = (stepNodes || [])
    .filter((node) => stepExecutionFeed.isRootStep(node))
    .slice()
    .sort((a, b) => stepNodeTimestamp(a) - stepNodeTimestamp(b));
  const frById = collectFunctionResponsesById(events);
  const pairedResponseIds = new Set();
  let shownPlotPaths = new Set();
  let msgIdx = 0;

  for (const event of sortedEvents) {
    const role = event.author === "user" ? "user" : "agent";
    if (role === "user") {
      const text = displayMessageFromStoredUserText(
        (event.content?.parts || []).map((part) => part.text || "").join("")
      );
      if (text) addMessage("user", text, msgIdx++);
      shownPlotPaths = new Set();
      continue;
    }

    const evtTimeline = assignStepNodesToExecutorCalls(
      eventToTimelineParts(event, frById, pairedResponseIds),
      pendingStepNodes,
    );
    if (evtTimeline.length > 0) {
      addAgentTimelineMessage(evtTimeline, shownPlotPaths, msgIdx++);
    }
  }

  pendingStepNodes.forEach((node) => stepExecutionFeed.appendStatic(node));
}

function updateSessionWorkdirDisplay(sessionData) {
  const workdirDisplay = document.getElementById("session-workdir-display");
  if (!workdirDisplay) return;
  const workdir = sessionData.state?.workdir || sessionData.state?.custom_workdir || state.defaultWorkdir || "";
  workdirDisplay.textContent = workdir;
  workdirDisplay.style.display = workdir ? "" : "none";
}

// Reload full session history from the ADK server and re-render the chat,
// mirroring Streamlit's load_session() called after send_message_sse().
async function loadSession(sessionId) {
  try {
    const sessionData = await fetchSessionData(sessionId);
    if (!sessionData) {
      state.sessionReady = false;
      return;
    }
    state.sessionReady = true;
    if (state.deploymentMode === "local" && sessionData.userId) {
      state.activeSessionUserId = sessionData.userId;
    }
    const events = sessionData.events || [];

    // Show session summary banner if available, clear otherwise
    if (sessionData.summary) {
      state.sessionSummaries[sessionId] = sessionData.summary;
      state.summaryGeneratedFor.add(sessionId);
    }
    renderSessionBanner(sessionData.summary || "");

    const graphNodes = await fetchSessionStepNodes(sessionId);
    renderSessionTimeline(events, graphNodes);

    await refreshSessionFiles();
    updateSessionWorkdirDisplay(sessionData);
  } catch (err) {
    console.error("Failed to load session:", err);
  }
}

// ---------------------------------------------------------------------------
// Streaming deduplication helpers (ported from streamlit_app.py)
// ---------------------------------------------------------------------------

function mergeReplayedText(current, incoming) {
  if (!incoming) return current;
  if (!current) return incoming;
  if (incoming.startsWith(current)) return incoming;
  if (current.endsWith(incoming)) return current;
  const maxOverlap = Math.min(current.length, incoming.length);
  for (let overlap = maxOverlap; overlap > 0; overlap--) {
    if (current.endsWith(incoming.slice(0, overlap))) {
      return current + incoming.slice(overlap);
    }
  }
  return current + incoming;
}

function compactRepeatedPrefixSnapshots(text) {
  if (!text) return text;
  let compacted = text;
  let changed = true;
  while (changed) {
    changed = false;
    const maxPrefix = Math.floor(compacted.length / 2);
    for (let size = maxPrefix; size > 3; size--) {
      const prefix = compacted.slice(0, size);
      const rest = compacted.slice(size);
      if (rest.startsWith(prefix)) {
        compacted = rest;
        changed = true;
        break;
      }
    }
  }
  return compacted;
}

function upsertTimelineThought(timeline, text) {
  if (!text) return;
  const compacted = compactRepeatedPrefixSnapshots(text);
  const last = timeline[timeline.length - 1];
  if (last?.type === "thought") {
    last.text = compactRepeatedPrefixSnapshots(mergeReplayedText(last.text || "", compacted));
    return;
  }
  timeline.push({ type: "thought", text: compacted });
}

function upsertTimelineText(timeline, text) {
  for (let i = timeline.length - 1; i >= 0; i--) {
    if (timeline[i].type === "text") timeline.splice(i, 1);
  }
  if (text) timeline.push({ type: "text", text });
}

function timelineEventKey(event) {
  if (event.id) return `${event.type}:${event.id}`;
  const payload = event.type === "function_call" ? event.args : event.response;
  return `${event.type}:${event.name || "Unknown"}:${JSON.stringify(payload || {})}`;
}

function upsertTimelineEvent(timeline, event) {
  const eventKey = timelineEventKey(event);
  for (let i = 0; i < timeline.length; i++) {
    const item = timeline[i];
    if (
      (item.type === "function_call" || item.type === "function_response") &&
      timelineEventKey(item) === eventKey
    ) {
      timeline[i] = event;
      return;
    }
  }
  const last = timeline[timeline.length - 1];
  if (last && JSON.stringify(last) === JSON.stringify(event)) return;
  timeline.push(event);
}

// ---------------------------------------------------------------------------
// Message sending + SSE streaming
// ---------------------------------------------------------------------------

function setSendingState(isSending, controller = null) {
  state.isSending = isSending;
  state.sendController = controller;
  if (!sendBtn) return;
  sendBtn.textContent = isSending ? "■" : "➜";
  sendBtn.title = isSending ? "Stop" : "Send";
  sendBtn.classList.toggle("is-stopping", isSending);
}

function stopCurrentMessage() {
  if (!state.isSending || !state.sendController) return;
  fetch(`/api/sessions/${state.sessionId}/cancel`, { method: "POST" }).catch(() => {});
  state.sendController.abort();
  pollCancellationConfirmed(state.sessionId);
}

function pollCancellationConfirmed(sessionId, attempts = 0) {
  const MAX_ATTEMPTS = 20;  // 20 × 2s = 40s timeout
  const INTERVAL_MS = 2000;

  if (attempts >= MAX_ATTEMPTS) {
    addMessage("agent", "⚠️ Stop requested but execution may still be running in the background.");
    return;
  }

  setTimeout(async () => {
    try {
      const res = await fetch(`/api/sessions/${sessionId}/cancel`);
      const data = await res.json();
      if (!data.cancellation_requested) {
        addMessage("agent", "✓ Execution stopped.");
        return;
      }
    } catch (_) { /* ignore transient network errors */ }
    pollCancellationConfirmed(sessionId, attempts + 1);
  }, INTERVAL_MS);
}

async function sendMessage(message) {
  if (!message.trim()) return;
  if (state.isSending) return;
  if (!state.userId) { showLoginModal(); return; }
  if (!canWriteActiveSession()) {
    addMessage("agent", `Admin view is read-only for ${state.activeSessionUserId}'s session.`);
    return;
  }

  // Clear any stale cancellation flag left over from a previous stop so that
  // step executors launched in this new run don't abort immediately.
  try { await fetch(`/api/sessions/${state.sessionId}/cancel`, { method: "DELETE" }); } catch (_) {}

  const uploadsForMessage = state.currentUploads.slice();
  const userMessageEl = addMessage("user", messageWithUploadNames(message, uploadsForMessage));
  const liveStartedAt = Date.now();
  const backendMessage = messageWithUploadContext(message, uploadsForMessage);
  textInput.value = "";
  clearCurrentUploads();
  autoResizeTextInput();

  if (!state.sessionReady) await createSession();
  if (!state.sessionReady) {
    addMessage("agent", "Failed to create session — the backend may still be loading. Please try again in a moment.");
    stepExecutionFeed.finishLiveTurn();
    return;
  }
  agentGraph.reset();
  planGraph.reset();
  const liveTurnContainer = stepExecutionFeed.startLiveTurn(userMessageEl, liveStartedAt);
  agentGraph.startPolling(state.sessionId);
  planGraph.startPolling(state.sessionId);

  const controller = new AbortController();
  setSendingState(true, controller);
  const payload = {
    app_name: APP_NAME,
    user_id: activeSessionBackendUserId(),
    session_id: state.sessionId,
    new_message: {
      role: "user",
      parts: [{ text: backendMessage }],
    },
  };

  const timeline = [];
  let timelineContainer = null;
  let accText = "";
  let summaryTriggered = false;
  const shownPlotPaths = new Set();

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
    let lineBuf = "";
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      lineBuf += decoder.decode(value, { stream: true });
      const lines = lineBuf.split("\n");
      lineBuf = lines.pop(); // keep the incomplete last line in the buffer
      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed.startsWith("data: ")) continue;
        const dataStr = trimmed.slice(6);
        if (dataStr === "[DONE]") continue;
        try {
          const evt = JSON.parse(dataStr);
          const parts = evt?.content?.parts || [];
          for (const p of parts) {
            if (p.thought) {
              upsertTimelineThought(timeline, p.text || "");
            } else if (p.functionCall) {
              const fc = p.functionCall;
              upsertTimelineEvent(timeline, {
                type: "function_call",
                id: fc.id,
                name: fc.name || "Unknown",
                args: fc.args || {},
              });
            } else if (p.functionResponse) {
              const fr = p.functionResponse;
              upsertTimelineEvent(timeline, {
                type: "function_response",
                id: fr.id,
                name: fr.name || "Unknown",
                response: fr.response || {},
              });
            } else if (p.text) {
              accText = mergeReplayedText(accText, p.text);
              upsertTimelineText(timeline, compactRepeatedPrefixSnapshots(accText));
              // Trigger summary early: on first agent text output (after planning)
              if (!summaryTriggered && !state.summaryGeneratedFor.has(state.sessionId) && !state.sessionSummaries[state.sessionId]) {
                summaryTriggered = true;
                state.summaryGeneratedFor.add(state.sessionId);
                generateSessionSummary(state.sessionId);
              }
            }

            if (timeline.length > 0 && !timelineContainer) {
              timelineContainer = addAgentTimelineMessage(timeline, shownPlotPaths, undefined, liveTurnContainer);
            } else if (timelineContainer) {
              renderTimeline(timelineContainer, timeline, shownPlotPaths);
            }
          }
        } catch (_) {
          // ignore malformed lines
        }
      }
    }
    // Flush remaining data in the line buffer
    if (lineBuf.trim().startsWith("data: ")) {
      const dataStr = lineBuf.trim().slice(6);
      if (dataStr !== "[DONE]") {
        try {
          const evt = JSON.parse(dataStr);
          const parts = evt?.content?.parts || [];
          for (const p of parts) {
            if (p.thought) upsertTimelineThought(timeline, p.text || "");
            else if (p.functionCall) upsertTimelineEvent(timeline, { type: "function_call", id: p.functionCall.id, name: p.functionCall.name || "Unknown", args: p.functionCall.args || {} });
            else if (p.functionResponse) upsertTimelineEvent(timeline, { type: "function_response", id: p.functionResponse.id, name: p.functionResponse.name || "Unknown", response: p.functionResponse.response || {} });
            else if (p.text) { accText = mergeReplayedText(accText, p.text); upsertTimelineText(timeline, compactRepeatedPrefixSnapshots(accText)); }
            if (timeline.length > 0 && !timelineContainer) timelineContainer = addAgentTimelineMessage(timeline, shownPlotPaths, undefined, liveTurnContainer);
            else if (timelineContainer) renderTimeline(timelineContainer, timeline, shownPlotPaths);
          }
        } catch (_) {}
      }
    }
  } catch (err) {
    if (err?.name === "AbortError") {
      addMessage("agent", "Stopping execution…", undefined, liveTurnContainer);
    } else {
      addMessage("agent", `Backend error: ${err}`, undefined, liveTurnContainer);
    }
  } finally {
    await agentGraph._poll(state.sessionId);
    agentGraph.stopPolling();
    await planGraph._poll(state.sessionId);
    planGraph.stopPolling();
    await refreshSessionFiles();
    stepExecutionFeed.finishLiveTurn();
    setSendingState(false);
    await loadSession(state.sessionId);
  }
}

// ---------------------------------------------------------------------------
// Structure viewer
// ---------------------------------------------------------------------------

function structureTabId(path) {
  let hash = 0;
  const source = String(path || "");
  for (let i = 0; i < source.length; i++) {
    hash = ((hash << 5) - hash + source.charCodeAt(i)) | 0;
  }
  return `structure-${Math.abs(hash)}`;
}

function structureTabTitle(path) {
  const filename = String(path || "Structure").split(/[\\/]/).filter(Boolean).pop();
  return filename || "Structure";
}

function activateCenterTab(tabId) {
  state.activeCenterTabId = tabId;
  centerTabsScroll?.querySelectorAll(".center-tab")?.forEach((tab) => {
    const active = tab.dataset.tabId === tabId;
    tab.classList.toggle("active", active);
    tab.setAttribute("aria-selected", String(active));
  });
  centerTabPanels?.querySelectorAll(".center-tab-panel")?.forEach((panel) => {
    panel.classList.toggle("active", panel.dataset.tabId === tabId);
  });

  const structureTab = structureTabs.get(tabId);
  state.structure3dViewer = structureTab?.viewer || null;
  if (structureTab?.viewer) {
    requestAnimationFrame(() => {
      try {
        structureTab.viewer.resize();
        structureTab.viewer.render();
      } catch (_) {
        // ignore transient 3Dmol resize issues
      }
    });
  }
}

function closeCenterTab(tabId) {
  const tab = structureTabs.get(tabId);
  if (!tab) return;

  tab.button.remove();
  tab.panel.remove();
  structureTabs.delete(tabId);

  if (state.activeCenterTabId === tabId) {
    activateCenterTab("chat");
  }
}

function ensureStructureTab(item) {
  const tabId = structureTabId(item.path);
  const existing = structureTabs.get(tabId);
  if (existing) {
    activateCenterTab(tabId);
    return existing;
  }

  const button = document.createElement("button");
  button.className = "center-tab";
  button.type = "button";
  button.role = "tab";
  button.dataset.tabId = tabId;
  button.id = `tab-${tabId}`;
  button.setAttribute("aria-selected", "false");
  button.setAttribute("aria-controls", `${tabId}-panel`);
  button.title = item.path;

  const title = document.createElement("span");
  title.className = "center-tab-title";
  title.textContent = structureTabTitle(item.path);
  button.appendChild(title);

  const close = document.createElement("span");
  close.className = "center-tab-close";
  close.dataset.closeTabId = tabId;
  close.setAttribute("aria-hidden", "true");
  close.textContent = "×";
  button.appendChild(close);

  const panel = document.createElement("div");
  panel.className = "center-tab-panel structure-tab-panel";
  panel.id = `${tabId}-panel`;
  panel.role = "tabpanel";
  panel.dataset.tabId = tabId;
  panel.setAttribute("aria-labelledby", button.id);

  const header = document.createElement("div");
  header.className = "structure-tab-header";

  const labelWrap = document.createElement("div");
  const eyebrow = document.createElement("div");
  eyebrow.className = "eyebrow";
  eyebrow.textContent = "Structure";
  const meta = document.createElement("div");
  meta.className = "sv-meta";
  labelWrap.append(eyebrow, meta);
  header.appendChild(labelWrap);

  const canvas = document.createElement("div");
  canvas.className = "sv-canvas structure-tab-canvas";

  panel.append(header, canvas);
  centerTabsScroll?.appendChild(button);
  centerTabPanels?.appendChild(panel);

  const tab = { id: tabId, item, button, panel, canvas, meta, viewer: null };
  structureTabs.set(tabId, tab);
  activateCenterTab(tabId);
  return tab;
}

centerTabs?.addEventListener("click", (event) => {
  const closeEl = event.target.closest("[data-close-tab-id]");
  if (closeEl) {
    event.stopPropagation();
    closeCenterTab(closeEl.dataset.closeTabId);
    return;
  }

  const tab = event.target.closest(".center-tab");
  if (tab?.dataset.tabId) activateCenterTab(tab.dataset.tabId);
});

async function openViewer(item) {
  graphDetail.classList.add("hidden");
  const tab = ensureStructureTab(item);
  tab.canvas.innerHTML = '<div style="color:var(--muted);padding:16px;font-size:13px">Loading…</div>';
  tab.meta.textContent = "";

  try {
    const resp = await fetch(`/api/structure/view?path=${encodeURIComponent(item.path)}&session_id=${encodeURIComponent(state.sessionId || "")}`);
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const data = await resp.json();

    tab.canvas.innerHTML = "";

    const viewer = $3Dmol.createViewer(tab.canvas, { backgroundColor: state.theme === "light" ? "0xf8fbff" : "0x06080f" });
    tab.viewer = viewer;
    if (state.activeCenterTabId === tab.id) state.structure3dViewer = viewer;
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
    refreshGraphAndStructureLayout();

    tab.meta.textContent =
      `${data.formula}  ·  ${data.n_atoms} atoms${data.periodic ? "  ·  periodic" : ""}`;
  } catch (err) {
    tab.canvas.innerHTML =
      `<div style="color:#f87171;padding:16px;font-size:13px">Failed to load structure: ${err}</div>`;
  }
}

// ---------------------------------------------------------------------------
// File viewer
// ---------------------------------------------------------------------------

async function openFileViewer(file) {
  const modal = document.getElementById("file-viewer-modal");
  const content = document.getElementById("fv-content");
  const filenameEl = document.getElementById("fv-filename");
  if (!modal) return;

  filenameEl.textContent = file.name;
  content.innerHTML = '<p style="color:var(--muted);padding:16px 20px">Loading…</p>';
  modal.classList.remove("hidden");

  const type = classifyPath(file.path);
  const url = pathToApiUrl(file.path);

  if (type === "image") {
    const wrap = document.createElement("div");
    wrap.className = "fv-img-wrap";
    const img = document.createElement("img");
    img.src = url;
    img.alt = file.name;
    img.style.cursor = "zoom-in";
    img.addEventListener("click", () => lightbox.open(url));
    wrap.appendChild(img);
    content.innerHTML = "";
    content.appendChild(wrap);
    return;
  }

  try {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const text = await res.text();
    if (text.includes("\0")) {
      content.innerHTML = '<p style="color:var(--muted);padding:16px 20px">Binary file — cannot preview.</p>';
      return;
    }
    const pre = document.createElement("pre");
    pre.className = "fv-pre";
    pre.textContent = text;
    content.innerHTML = "";
    content.appendChild(pre);
  } catch (err) {
    content.innerHTML = `<p style="color:#f87171;padding:16px 20px">Failed to load: ${err.message}</p>`;
  }
}

document.getElementById("fv-close")?.addEventListener("click", () => {
  document.getElementById("file-viewer-modal")?.classList.add("hidden");
});
document.getElementById("file-viewer-modal")?.addEventListener("click", (e) => {
  if (e.target === e.currentTarget)
    e.currentTarget.classList.add("hidden");
});

// ---------------------------------------------------------------------------
// Image lightbox
// ---------------------------------------------------------------------------

const lightbox = {
  el: document.getElementById("image-lightbox"),
  img: document.getElementById("lightbox-img"),
  viewport: document.getElementById("lightbox-viewport"),
  label: document.getElementById("lightbox-zoom-label"),
  _scale: 1,
  _tx: 0,
  _ty: 0,
  _dragging: false,
  _dragStartX: 0,
  _dragStartY: 0,
  _dragStartTx: 0,
  _dragStartTy: 0,

  open(src) {
    this._scale = 1;
    this._tx = 0;
    this._ty = 0;
    this.img.src = src;
    this.img.style.transform = "";
    this.el.classList.remove("hidden");
    this._updateLabel();
  },

  close() {
    this.el.classList.add("hidden");
    this.img.src = "";
  },

  _apply() {
    this.img.style.transform = `translate(${this._tx}px, ${this._ty}px) scale(${this._scale})`;
    this._updateLabel();
  },

  _updateLabel() {
    if (this.label) this.label.textContent = `${Math.round(this._scale * 100)}%`;
  },

  zoomIn() {
    this._scale = Math.min(this._scale * 1.3, 20);
    this._apply();
  },

  zoomOut() {
    const newScale = this._scale / 1.3;
    if (newScale < 0.1) return;
    const oldScale = this._scale;
    this._scale = newScale;
    const factor = this._scale / oldScale;
    this._tx *= factor;
    this._ty *= factor;
    this._apply();
  },

  resetZoom() {
    this._scale = 1;
    this._tx = 0;
    this._ty = 0;
    this._apply();
  },
};

lightbox.viewport?.addEventListener("wheel", (e) => {
  e.preventDefault();
  if (e.deltaY < 0) lightbox.zoomIn();
  else lightbox.zoomOut();
}, { passive: false });

lightbox.viewport?.addEventListener("mousedown", (e) => {
  if (e.target === lightbox.img) {
    lightbox._dragging = true;
    lightbox._dragStartX = e.clientX;
    lightbox._dragStartY = e.clientY;
    lightbox._dragStartTx = lightbox._tx;
    lightbox._dragStartTy = lightbox._ty;
    e.preventDefault();
  }
});

document.addEventListener("mousemove", (e) => {
  if (!lightbox._dragging) return;
  lightbox._tx = lightbox._dragStartTx + (e.clientX - lightbox._dragStartX);
  lightbox._ty = lightbox._dragStartTy + (e.clientY - lightbox._dragStartY);
  lightbox._apply();
});

document.addEventListener("mouseup", () => {
  lightbox._dragging = false;
});

// Click on viewport backdrop (not the image) to close
lightbox.viewport?.addEventListener("click", (e) => {
  if (e.target === lightbox.viewport) lightbox.close();
});

document.getElementById("lightbox-close")?.addEventListener("click", () => lightbox.close());
document.getElementById("lightbox-zoom-in")?.addEventListener("click", () => lightbox.zoomIn());
document.getElementById("lightbox-zoom-out")?.addEventListener("click", () => lightbox.zoomOut());
document.getElementById("lightbox-zoom-reset")?.addEventListener("click", () => lightbox.resetZoom());

document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && !lightbox.el.classList.contains("hidden")) lightbox.close();
});

initPanelResizers();
initColResizers();

// ---------------------------------------------------------------------------
// Event listeners
// ---------------------------------------------------------------------------

sendBtn.addEventListener("click", () => {
  if (state.isSending) {
    stopCurrentMessage();
    return;
  }
  sendMessage(textInput.value);
});
textInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    if (state.isSending) return;
    sendMessage(textInput.value);
  }
});

if (fileUploadBtn && fileUploadInput) {
  fileUploadBtn.addEventListener("click", () => fileUploadInput.click());
  fileUploadInput.addEventListener("change", (e) => uploadFilesToSession(e.target.files));
}

// Avatar upload
const avatarUploadInput = document.getElementById("avatar-upload-input");
const avatarUploadBtn = document.getElementById("avatar-upload-btn");
if (avatarUploadBtn && avatarUploadInput) {
  applyUserAvatarToEl(avatarUploadBtn);
  avatarUploadBtn.addEventListener("click", () => avatarUploadInput.click());
  avatarUploadInput.addEventListener("change", (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (ev) => {
      setUserAvatar(ev.target.result);
      applyUserAvatarToEl(avatarUploadBtn);
    };
    reader.readAsDataURL(file);
    e.target.value = "";
  });
}

Array.from(document.querySelectorAll("[data-quick]"))
  .forEach((btn) => btn.addEventListener("click", () => sendMessage(btn.dataset.quick || "")));

// Agent mode selector
function updateComposerModeState(mode) {
  if (!inputContainer) return;
  inputContainer.dataset.agentMode = mode || "normal";
}

if (modeSelector) {
  modeSelector.querySelectorAll(".mode-btn").forEach((btn) => {
    btn.classList.toggle("mode-btn-active", btn.dataset.mode === state.agentMode);
  });
  updateComposerModeState(state.agentMode);
  modeSelector.addEventListener("click", (e) => {
    const btn = e.target.closest(".mode-btn");
    if (!btn) return;
    const mode = btn.dataset.mode;
    state.agentMode = mode;
    localStorage.setItem(AGENT_MODE_KEY, mode);
    modeSelector.querySelectorAll(".mode-btn").forEach((b) =>
      b.classList.toggle("mode-btn-active", b.dataset.mode === mode)
    );
    updateComposerModeState(mode);
    patchSessionAgentMode(mode);
  });
}

resetBtn.addEventListener("click", () => {
  _doNewSession(state.defaultWorkdir || "");
});

async function _doNewSession(customWorkdir) {
  state.customWorkdir = customWorkdir;
  state.sessionId = `session-${Math.floor(Date.now() / 1000)}`;
  state.activeSessionUserId = state.userId;
  state.sessionReady = false;
  localStorage.setItem("mat_sessionId", state.sessionId);
  sessionIdEl.textContent = state.sessionId;
  chatArea.innerHTML = "";
  stepExecutionFeed.reset();
  state.sessionSummaries = {};
  state.summaryGeneratedFor = new Set();
  renderSessionBanner("");
  renderSessionFilesTree([]);
  clearCurrentUploads();
  agentGraph.reset();
  planGraph.reset();
  hidePlanGraph();
  await createSession();
}

// ---------------------------------------------------------------------------
// Settings panel
// ---------------------------------------------------------------------------

const settingsModal = document.getElementById("settings-modal");
const settingsBtn = document.getElementById("settings-btn");
const settingsClose = document.getElementById("settings-close");
const settingsSave = document.getElementById("settings-save");
const settingsStatus = document.getElementById("settings-status");
const settingsUsername = document.getElementById("settings-username");
const settingsUuid = document.getElementById("settings-uuid");
const skillsChecklist = document.getElementById("skills-checklist");
const settingsRestartBtn = document.getElementById("settings-restart-btn");

// Env config input refs
const envInputs = {
  LLM_MODEL:              () => document.getElementById("settings-llm-model"),
  LLM_API_KEY:            () => document.getElementById("settings-llm-apikey"),
  LLM_BASE_URL:           () => document.getElementById("settings-llm-baseurl"),
  EMBEDDING_MODEL:        () => document.getElementById("settings-llm-embed"),
  BOHRIUM_EMAIL:          () => document.getElementById("settings-bohr-email"),
  BOHRIUM_PASSWORD:       () => document.getElementById("settings-bohr-password"),
  BOHRIUM_PROJECT_ID:     () => document.getElementById("settings-bohr-project"),
  BOHRIUM_VASP_IMAGE:     () => document.getElementById("settings-bohr-vasp-image"),
  BOHRIUM_VASP_MACHINE:   () => document.getElementById("settings-bohr-vasp-machine"),
  BOHRIUM_DEEPMD_IMAGE:   () => document.getElementById("settings-bohr-deepmd-image"),
  BOHRIUM_DEEPMD_MACHINE: () => document.getElementById("settings-bohr-deepmd-machine"),
  DEEPMD_MODEL_PATH:      () => document.getElementById("settings-bohr-deepmd-model"),
};

function openSettingsModal() {
  settingsModal.classList.remove("hidden");
  settingsUsername.value = state.displayName || "";
  settingsUuid.value = state.userId || "";
  loadSettingsData();
}

function closeSettingsModal() {
  settingsModal.classList.add("hidden");
}

// ---- tree helpers ----------------------------------------------------------

function _buildSkillTree(skills) {
  const byName = new Map(skills.map((s) => [s.name, { ...s, children: [] }]));
  const roots = [];
  for (const s of skills) {
    const node = byName.get(s.name);
    if (s.parent && byName.has(s.parent)) {
      byName.get(s.parent).children.push(node);
    } else {
      roots.push(node);
    }
  }
  const cmp = (a, b) => {
    const ak = a.children.length > 0 ? 0 : 1;
    const bk = b.children.length > 0 ? 0 : 1;
    return ak !== bk ? ak - bk : a.name.localeCompare(b.name);
  };
  roots.sort(cmp);
  roots.forEach((r) => r.children.sort(cmp));
  return roots;
}

function _syncParent(parentCb, childWrap) {
  const childCbs = Array.from(
    childWrap.querySelectorAll(":scope > .st-item > .st-row > .skill-checkbox")
  ).filter((c) => !c.disabled);
  if (!childCbs.length) return;
  const checkedCount = childCbs.filter((c) => c.checked).length;
  if (checkedCount === childCbs.length) {
    parentCb.indeterminate = false;
    parentCb.checked = true;
  } else if (checkedCount === 0) {
    parentCb.indeterminate = false;
    parentCb.checked = false;
  } else {
    parentCb.indeterminate = true;
  }
}

function _renderSkillNode(node, extraSkills, depth) {
  const hasChildren = node.children.length > 0;
  const isBuiltIn = node.planning_enabled && !extraSkills.has(node.name);

  const item = document.createElement("div");
  item.className = "st-item";

  const row = document.createElement("div");
  row.className = "st-row";
  if (depth > 0) row.style.paddingLeft = `${depth * 18}px`;

  const toggle = document.createElement("button");
  toggle.type = "button";
  toggle.className = "st-toggle";
  toggle.innerHTML = hasChildren ? "&#9654;" : "";
  toggle.disabled = !hasChildren;
  if (!hasChildren) toggle.style.visibility = "hidden";

  const cb = document.createElement("input");
  cb.type = "checkbox";
  cb.className = "skill-checkbox";
  cb.dataset.name = node.name;
  cb.checked = node.planning_enabled;
  cb.disabled = isBuiltIn;
  if (isBuiltIn) cb.title = "Enabled by built-in category";

  const enabledCb = document.createElement("input");
  enabledCb.type = "checkbox";
  enabledCb.className = "skill-enabled-checkbox";
  enabledCb.dataset.name = node.name;
  enabledCb.checked = node.enabled !== false;
  enabledCb.title = "Enable skill for graph search";

  const nameEl = document.createElement("span");
  nameEl.className = "st-name";
  nameEl.textContent = node.name;

  if (node.is_custom) {
    const badge = document.createElement("span");
    badge.className = "skill-badge-custom";
    badge.textContent = "Custom";
    nameEl.appendChild(badge);
  }

  const descEl = document.createElement("span");
  descEl.className = "st-desc";
  descEl.textContent = node.description;

  row.append(toggle, enabledCb, cb, nameEl, descEl);

  if (node.is_custom) {
    const delBtn = document.createElement("button");
    delBtn.type = "button";
    delBtn.className = "skill-delete-btn";
    delBtn.title = "Delete custom skill";
    delBtn.textContent = "✕";
    delBtn.addEventListener("click", async (e) => {
      e.stopPropagation();
      if (!confirm(`Delete custom skill "${node.name}"?`)) return;
      await deleteCustomSkill(node.name);
    });
    row.appendChild(delBtn);
  }

  item.appendChild(row);

  if (hasChildren) {
    const childWrap = document.createElement("div");
    childWrap.className = "st-children st-collapsed";

    for (const child of node.children) {
      childWrap.appendChild(_renderSkillNode(child, extraSkills, depth + 1));
    }
    item.appendChild(childWrap);

    // Expand / collapse
    toggle.addEventListener("click", () => {
      const collapsed = childWrap.classList.toggle("st-collapsed");
      toggle.innerHTML = collapsed ? "&#9654;" : "&#9660;";
    });

    // Parent → children propagation
    cb.addEventListener("change", () => {
      childWrap
        .querySelectorAll(".skill-checkbox:not(:disabled)")
        .forEach((c) => {
          c.checked = cb.checked;
          c.indeterminate = false;
        });
    });

    // Children → parent tri-state
    childWrap.addEventListener("change", () => _syncParent(cb, childWrap));

    // Initial tri-state
    queueMicrotask(() => _syncParent(cb, childWrap));
  }

  return item;
}

// ---- custom skill upload / delete ------------------------------------------

async function uploadCustomSkill() {
  const nameInput = document.getElementById("custom-skill-name");
  const mdInput = document.getElementById("custom-skill-md");
  const refsInput = document.getElementById("custom-skill-refs");
  const scriptsInput = document.getElementById("custom-skill-scripts");
  const errorEl = document.getElementById("custom-skill-error");
  const uploadBtn = document.getElementById("custom-skill-upload-btn");

  if (!nameInput || !mdInput) return;
  errorEl.textContent = "";

  const name = nameInput.value.trim();
  if (!name) { errorEl.textContent = "Skill name is required."; return; }
  if (!/^[a-z0-9][a-z0-9_-]*$/.test(name)) {
    errorEl.textContent = "Name must be lowercase alphanumeric with hyphens/underscores.";
    return;
  }
  if (!mdInput.files.length) { errorEl.textContent = "SKILL.md file is required."; return; }

  const formData = new FormData();
  formData.append("name", name);
  formData.append("skill_md", mdInput.files[0]);
  for (const ref of Array.from(refsInput?.files || [])) {
    formData.append("references", ref);
  }
  for (const script of Array.from(scriptsInput?.files || [])) {
    formData.append("scripts", script);
  }

  uploadBtn.disabled = true;
  uploadBtn.textContent = "Uploading…";
  try {
    const resp = await fetch("/api/skills/custom", { method: "POST", body: formData });
    const body = await resp.json().catch(() => ({}));
    if (!resp.ok) {
      errorEl.textContent = body.detail || `Upload failed (${resp.status})`;
      return;
    }
    nameInput.value = "";
    mdInput.value = "";
    if (refsInput) refsInput.value = "";
    if (scriptsInput) scriptsInput.value = "";
    settingsStatus.textContent = `Custom skill "${name}" uploaded ✓`;
    setTimeout(() => { settingsStatus.textContent = ""; }, 3000);
    await loadSettingsData();
  } catch (err) {
    errorEl.textContent = `Upload failed: ${err.message}`;
  } finally {
    uploadBtn.disabled = false;
    uploadBtn.textContent = "Upload Skill";
  }
}

async function deleteCustomSkill(skillName) {
  try {
    const resp = await fetch(`/api/skills/custom/${encodeURIComponent(skillName)}`, { method: "DELETE" });
    const body = await resp.json().catch(() => ({}));
    if (!resp.ok) {
      settingsStatus.textContent = body.detail || `Delete failed (${resp.status})`;
      return;
    }
    settingsStatus.textContent = `Custom skill "${skillName}" deleted ✓`;
    setTimeout(() => { settingsStatus.textContent = ""; }, 3000);
    await loadSettingsData();
  } catch (err) {
    settingsStatus.textContent = `Delete failed: ${err.message}`;
  }
}

// ---- load / save -----------------------------------------------------------

async function loadSettingsData() {
  skillsChecklist.innerHTML =
    '<p class="settings-hint" style="opacity:0.6">Loading…</p>';
  try {
    const [skillsRes, settingsRes, envRes] = await Promise.all([
      fetch("/api/skills"),
      fetch("/api/settings"),
      fetch("/api/env-config"),
    ]);
    const skills = await skillsRes.json();
    const cfg = await settingsRes.json();
    const envCfg = envRes.ok ? await envRes.json() : {};
    const extraSkills = new Set((cfg.planning || {}).extra_skills || []);

    // Populate default workdir from config
    state.defaultWorkdir = (cfg.workspace || {}).default_workdir || "";
    const wdInput = document.getElementById("settings-default-workdir");
    if (wdInput) wdInput.value = state.defaultWorkdir;

    const roots = _buildSkillTree(skills);
    skillsChecklist.innerHTML = "";

    const header = document.createElement("div");
    header.className = "st-header";
    header.innerHTML = '<span></span><span class="st-col-label" title="Skill is visible to graph search">Enabled</span><span class="st-col-label" title="Full SKILL.md available to planning agent">Planning</span><span></span>';
    skillsChecklist.appendChild(header);

    for (const node of roots) {
      skillsChecklist.appendChild(_renderSkillNode(node, extraSkills, 0));
    }

    // Custom skill upload section
    const uploadSection = document.createElement("div");
    uploadSection.className = "skill-upload-section";
    uploadSection.innerHTML = `
      <details class="skill-upload-details">
        <summary class="skill-upload-summary">+ Add Custom Skill</summary>
        <div class="skill-upload-form">
          <label class="settings-label">Skill Name <span style="font-weight:400;text-transform:none;letter-spacing:0">(lowercase, hyphens/underscores only)</span></label>
          <input id="custom-skill-name" class="text-input settings-env-input" placeholder="e.g. my-custom-skill" autocomplete="off" />
          <label class="settings-label" style="margin-top:8px">SKILL.md file <span style="color:#f87171">*</span></label>
          <input id="custom-skill-md" type="file" accept=".md,text/markdown,text/plain" class="skill-file-input" />
          <label class="settings-label" style="margin-top:8px">Reference files <span style="font-weight:400;text-transform:none;letter-spacing:0">(optional, multiple)</span></label>
          <input id="custom-skill-refs" type="file" multiple class="skill-file-input" />
          <label class="settings-label" style="margin-top:8px">Script files <span style="font-weight:400;text-transform:none;letter-spacing:0">(optional, .py/.sh/.js)</span></label>
          <input id="custom-skill-scripts" type="file" multiple accept=".py,.sh,.bash,.js" class="skill-file-input" />
          <p id="custom-skill-error" class="skill-upload-error"></p>
          <button id="custom-skill-upload-btn" type="button" class="ghost" style="margin-top:8px;width:100%;justify-content:center">Upload Skill</button>
        </div>
      </details>
    `;
    skillsChecklist.appendChild(uploadSection);

    document.getElementById("custom-skill-upload-btn")?.addEventListener("click", uploadCustomSkill);

    // Populate env config inputs
    for (const [key, getEl] of Object.entries(envInputs)) {
      const el = getEl();
      if (el && envCfg[key] !== undefined) {
        el.value = envCfg[key];
      }
    }
  } catch (err) {
    skillsChecklist.innerHTML = `<p class="settings-hint" style="color:#f87171">Failed to load: ${err.message}</p>`;
  }
}

async function saveSettings() {
  const username = settingsUsername.value.trim();
  const nextDefaultWorkdir = document.getElementById("settings-default-workdir")?.value?.trim() || "";
  const extraSkills = Array.from(
    skillsChecklist.querySelectorAll(".skill-checkbox:not(:disabled)")
  )
    .filter((cb) => cb.checked && !cb.indeterminate)
    .map((cb) => cb.dataset.name);

  const disabledSkills = Array.from(
    skillsChecklist.querySelectorAll(".skill-enabled-checkbox")
  )
    .filter((cb) => !cb.checked)
    .map((cb) => cb.dataset.name);

  // Collect env config values (skip empty sensitive fields with "***")
  const envValues = {};
  const sensitiveKeys = new Set(["LLM_API_KEY", "BOHRIUM_PASSWORD"]);
  for (const [key, getEl] of Object.entries(envInputs)) {
    const el = getEl();
    if (!el) continue;
    const val = el.value;
    if (sensitiveKeys.has(key) && (!val || val === "***")) continue;
    envValues[key] = val;
  }

  try {
    settingsSave.disabled = true;
    settingsStatus.textContent = "Saving…";

    const requests = [
      fetch("/api/settings", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          planning: { extra_skills: extraSkills },
          skills: { disabled: disabledSkills },
          user: username ? { name: username } : undefined,
          workspace: { default_workdir: nextDefaultWorkdir },
        }),
      }),
    ];
    if (Object.keys(envValues).length > 0) {
      requests.push(
        fetch("/api/env-config", {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ values: envValues }),
        })
      );
    }

    const results = await Promise.all(requests);
    for (const res of results) {
      if (!res.ok) throw new Error(await res.text());
    }

    if (state.deploymentMode === "server" && username && username !== state.displayName) {
      closeSettingsModal();
      await applyLogin(username, null);
    }
    state.defaultWorkdir = nextDefaultWorkdir;
    settingsStatus.textContent = "Saved ✓";
    setTimeout(() => { settingsStatus.textContent = ""; }, 2000);
  } catch (err) {
    settingsStatus.textContent = `Error: ${err.message}`;
  } finally {
    settingsSave.disabled = false;
  }
}

// ---- restart backend -------------------------------------------------------

async function _pollBackendReady(maxAttempts = 30, intervalMs = 2000) {
  for (let i = 0; i < maxAttempts; i++) {
    await new Promise((resolve) => setTimeout(resolve, intervalMs));
    try {
      const userQuery = state.userId ? `?user_id=${encodeURIComponent(state.userId)}` : "";
      const res = await fetch(`/api/backend-status${userQuery}`);
      if (res.ok) {
        const data = await res.json();
        if (data.ready) return;
      }
    } catch (_) {}
  }
  throw new Error("Backend did not come back online in time");
}

async function restartBackend() {
  if (!settingsRestartBtn) return;
  settingsRestartBtn.disabled = true;
  settingsRestartBtn.textContent = "Restarting…";
  settingsStatus.textContent = "Restarting backend…";
  try {
    const userQuery = state.userId ? `?user_id=${encodeURIComponent(state.userId)}` : "";
    const res = await fetch(`/api/restart-backend${userQuery}`, { method: "POST" });
    if (!res.ok) throw new Error(await res.text());
    await _pollBackendReady();
    settingsStatus.textContent = "Backend restarted ✓";
    setTimeout(() => { settingsStatus.textContent = ""; }, 3000);
  } catch (err) {
    settingsStatus.textContent = `Restart failed: ${err.message}`;
  } finally {
    settingsRestartBtn.disabled = false;
    settingsRestartBtn.textContent = "↺ Restart Backend";
  }
}

// Tab switching
document.querySelectorAll(".settings-tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".settings-tab").forEach((t) => t.classList.remove("active"));
    document.querySelectorAll(".settings-pane").forEach((p) => p.classList.add("hidden"));
    tab.classList.add("active");
    const pane = document.getElementById(`tab-${tab.dataset.tab}`);
    if (pane) pane.classList.remove("hidden");
  });
});

if (settingsBtn) settingsBtn.addEventListener("click", openSettingsModal);
if (settingsClose) settingsClose.addEventListener("click", closeSettingsModal);
if (settingsSave) settingsSave.addEventListener("click", saveSettings);
if (settingsRestartBtn) settingsRestartBtn.addEventListener("click", restartBackend);
document.getElementById("settings-workdir-reset")?.addEventListener("click", () => {
  const wdInput = document.getElementById("settings-default-workdir");
  if (wdInput) wdInput.value = "";
});
settingsModal?.addEventListener("click", (e) => {
  if (e.target === settingsModal) closeSettingsModal();
});
