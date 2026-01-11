"""
Interactive episode viewer for detailed evaluation results.

Fixes vs previous version:
- Lossless JSON embedding via <script type="application/json"> (no f-string “stringifies then loses fields” issues).
- Schema-flexible renderers: if a field exists anywhere in episodes.json, it can be surfaced (and raw blocks always show it).
- Correct mapping for:
  - responses per framing
  - ML classifier outputs (including model metadata if present)
  - metrics_by_framing (supports MetricResult objects OR floats)
  - signals + conditioned + decision
  - council verdict summary AND (when present) per-judge internals
- Adds 3 sidebar tabs:
  1) Episode View
  2) Systems View
  3) Council Internals
- Adds risk propagation diagrams (inline SVG) + pipeline diagram (best-effort from pipeline_* fields)

Expected input: episodes.json as produced by risklab.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
import json
import html
import hashlib


class EpisodeViewer:
    """Generate interactive HTML viewer for episode details."""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_episode_viewer(self, episodes_file: Path) -> Path:
        """Generate interactive episode viewer HTML."""
        episodes_file = Path(episodes_file)

        with open(episodes_file, "r", encoding="utf-8") as f:
            episodes = json.load(f)

        html_text = self._generate_viewer_html(episodes, source_name=episodes_file.name)

        output_file = self.output_dir / "episode_viewer.html"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(html_text)

        return output_file

    def _generate_viewer_html(self, episodes: List[Dict[str, Any]], source_name: str) -> str:
        """
        Generate HTML. We embed raw JSON as application/json, then parse client-side.
        This avoids truncation/escaping bugs and keeps the viewer schema-flexible.
        """
        episodes_json = json.dumps(episodes, ensure_ascii=False, default=str)
        build_id = hashlib.sha1(episodes_json.encode("utf-8")).hexdigest()[:10]

        return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>Episode Evaluation Viewer</title>
<style>
:root {{
  --bg:#f5f5f5; --panel:#fff; --text:#222; --muted:#666; --border:#e6e6e6;
  --accent:#667eea; --accent2:#764ba2;
  --mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
}}
*{{box-sizing:border-box}}
body{{margin:0;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;background:var(--bg);color:var(--text)}}
.header{{padding:18px 16px;text-align:center;color:#fff;background:linear-gradient(135deg,var(--accent),var(--accent2));box-shadow:0 2px 10px rgba(0,0,0,.1)}}
.header h1{{margin:0;font-size:20px}}
.header p{{margin:6px 0 0;font-size:12px;opacity:.92}}
.pill{{display:inline-block;padding:3px 9px;border-radius:999px;background:rgba(255,255,255,.18);font-weight:700}}
.container{{max-width:1500px;margin:0 auto;padding:16px}}
.panel{{background:var(--panel);border:1px solid var(--border);border-radius:12px;box-shadow:0 2px 8px rgba(0,0,0,.06)}}

.selector{{padding:14px}}
.selectorTop{{display:flex;gap:12px;flex-wrap:wrap;align-items:center;justify-content:space-between}}
.selectorTop h2{{margin:0;font-size:14px}}
.search{{display:flex;gap:8px;flex-wrap:wrap;align-items:center}}
.search input{{width:360px;max-width:80vw;padding:10px 12px;border-radius:10px;border:1px solid var(--border);outline:none;font-size:13px}}
.chip{{border:1px solid var(--border);border-radius:999px;padding:6px 10px;font-size:12px;color:var(--muted);cursor:pointer;user-select:none;background:#fff}}
.chip.active{{border-color:var(--accent);color:var(--accent);background:#f0f4ff}}

.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:12px;margin-top:12px}}
.card{{border:2px solid var(--border);border-radius:10px;padding:12px;background:#fafafa;cursor:pointer;transition:.15s}}
.card:hover{{border-color:var(--accent);transform:translateY(-1px);box-shadow:0 6px 16px rgba(102,126,234,.14)}}
.card.active{{border-color:var(--accent);background:#f0f4ff}}
.card .title{{font-weight:800;font-size:13px;margin-bottom:6px}}
.card .meta{{font-size:12px;color:var(--muted);margin-bottom:6px}}
.badge{{display:inline-block;padding:4px 10px;border-radius:999px;color:#fff;font-size:11px;font-weight:800}}
.badge.low{{background:#16a34a}} .badge.medium{{background:#d97706}} .badge.high{{background:#e67e22}} .badge.critical{{background:#dc2626}}

.viewer{{display:none;grid-template-columns:1fr 430px;gap:12px;margin-top:12px}}
@media (max-width:1100px){{.viewer{{grid-template-columns:1fr}}}}

.chat{{padding:14px;max-height:78vh;overflow:auto}}
.sidebar{{padding:14px;max-height:78vh;overflow:auto}}
@media (max-width:1100px){{.chat,.sidebar{{max-height:none}}}}

.tabs{{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:10px}}
.tab{{padding:8px 10px;border-radius:999px;border:1px solid var(--border);background:#fff;color:var(--muted);font-size:12px;font-weight:800;cursor:pointer;user-select:none}}
.tab.active{{border-color:var(--accent);color:var(--accent);background:#f0f4ff}}

.framingTabs{{display:flex;gap:10px;flex-wrap:wrap;border-bottom:2px solid var(--border);margin-bottom:12px}}
.framingTab{{padding:8px 10px;border:0;background:transparent;color:var(--muted);cursor:pointer;font-weight:800;border-bottom:2px solid transparent}}
.framingTab.active{{color:var(--accent);border-bottom-color:var(--accent)}}

.msg{{margin:0 0 14px 0}}
.msg.user{{text-align:right}}
.bubble{{display:inline-block;max-width:88%;padding:11px 13px;border-radius:16px;white-space:pre-wrap;word-wrap:break-word}}
.msg.user .bubble{{background:var(--accent);color:#fff;border-bottom-right-radius:4px}}
.msg.assistant .bubble{{background:#f1f3f4;color:var(--text);border-bottom-left-radius:4px}}

.sectionTitle{{margin:14px 0 10px;border-bottom:1px solid var(--border);padding-bottom:8px;font-size:12px;font-weight:900;display:flex;justify-content:space-between;gap:10px}}
.hint{{font-size:11px;color:var(--muted);font-weight:800}}

.grid2{{display:grid;grid-template-columns:repeat(2,1fr);gap:10px}}
.kv{{display:flex;justify-content:space-between;gap:12px;padding:8px 10px;background:#f8f9fa;border-radius:10px;font-size:12px}}
.kv .k{{color:var(--muted);white-space:nowrap}}
.kv .v{{font-weight:900;text-align:right;overflow-wrap:anywhere}}
.mono{{font-family:var(--mono);font-size:11px}}

.table{{width:100%;border-collapse:collapse;font-size:12px}}
.table th,.table td{{border-bottom:1px solid #f0f0f0;padding:8px 6px;vertical-align:top;text-align:left}}
.table th{{font-size:11px;color:var(--muted);font-weight:900;text-transform:uppercase;letter-spacing:.02em;width:42%}}

.jsonBlock{{border:1px solid var(--border);border-radius:10px;overflow:hidden;background:#fff;margin:10px 0}}
.jsonHead{{background:#f8fafc;padding:10px 12px;display:flex;justify-content:space-between;gap:10px;cursor:pointer;user-select:none}}
.jsonHead .t{{font-weight:900;font-size:12px}}
.jsonHead .m{{font-size:11px;color:var(--muted);font-weight:700}}
pre.json{{margin:0;padding:10px 12px;max-height:340px;overflow:auto;background:#0b1020;color:#e6edf3;font-family:var(--mono);font-size:11px;display:none}}
.jsonBlock.open pre.json{{display:block}}

.diagramWrap{{border:1px solid var(--border);border-radius:10px;background:#fff;padding:10px;overflow:auto;margin:10px 0}}
.diagramTitle{{font-weight:900;font-size:12px;margin-bottom:8px}}
svg.diagram{{width:100%;min-width:640px;height:280px}}
.node{{fill:#f8fafc;stroke:#cbd5e1;stroke-width:1}}
.edge{{stroke:#94a3b8;stroke-width:1.4;marker-end:url(#arrow)}}
.nodeLabel{{font-size:12px;font-weight:900;fill:#0f172a}}
.nodeSub{{font-size:10px;font-family:var(--mono);fill:#334155}}

.small{{font-size:12px;color:var(--muted)}}
</style>
</head>
<body>
  <div class="header">
    <h1>Episode Evaluation Viewer</h1>
    <p>Source <span class="pill">{html.escape(source_name)}</span> · Build <span class="pill">{build_id}</span></p>
  </div>

  <div class="container">
    <div class="panel selector">
      <div class="selectorTop">
        <h2>Select Episode</h2>
        <div class="search">
          <input id="searchBox" placeholder="Search by name, domain, stakes, outcome..." />
          <span class="chip" id="chipHigh">High risk</span>
          <span class="chip" id="chipCouncil">Has council</span>
        </div>
      </div>
      <div class="grid" id="episodeGrid"></div>
    </div>

    <div class="viewer" id="viewer">
      <div class="panel chat">
        <div class="framingTabs" id="framingTabs"></div>
        <div id="chatMessages"></div>
      </div>

      <div class="panel sidebar">
        <div class="tabs" id="viewTabs"></div>
        <div id="viewBody"></div>
      </div>
    </div>

    <div class="panel" id="emptyState" style="margin-top:12px;padding:22px;text-align:center;color:var(--muted)">
      <h3 style="margin:0 0 6px 0;color:var(--text);font-size:15px">No Episode Selected</h3>
      <div class="small">Pick an episode above to view details.</div>
    </div>
  </div>

  <script id="episodesData" type="application/json">{episodes_json}</script>

<script>
/* ----------------------------- utilities ----------------------------- */
function parseEpisodes() {{
  try {{ return JSON.parse(document.getElementById("episodesData").textContent || "[]"); }}
  catch(e) {{ console.error("episodes JSON parse failed", e); return []; }}
}}
function safe(obj, path, fallback=null) {{
  try {{
    const parts = path.split(".");
    let cur = obj;
    for (const p of parts) {{ if (cur == null) return fallback; cur = cur[p]; }}
    return (cur === undefined) ? fallback : cur;
  }} catch(e) {{ return fallback; }}
}}
function clamp01(x) {{
  const n = Number(x);
  if (!Number.isFinite(n)) return 0;
  return Math.max(0, Math.min(1, n));
}}
function fmt(x, d=3) {{
  const n = Number(x);
  if (!Number.isFinite(n)) return "N/A";
  return n.toFixed(d);
}}
function esc(s) {{
  if (s == null) return "";
  return String(s).replaceAll("&","&amp;").replaceAll("<","&lt;").replaceAll(">","&gt;").replaceAll('"',"&quot;").replaceAll("'","&#039;");
}}
function riskLevel(score) {{
  const s = clamp01(score);
  if (s < 0.25) return "low";
  if (s < 0.50) return "medium";
  if (s < 0.75) return "high";
  return "critical";
}}
function pretty(obj) {{
  try {{ return JSON.stringify(obj, null, 2); }} catch(e) {{ return String(obj); }}
}}
function hasCouncil(ep) {{
  const v = safe(ep, "full_evaluation.council_verdict", null);
  if (!v) return false;
  if (typeof v === "string") return v.trim().length > 0;
  if (typeof v === "object") return Object.keys(v).length > 0;
  return false;
}}
function metricToNumber(v) {{
  // supports:
  // - number
  // - {value: number} (MetricResult-like)
  // - null/undefined
  if (v == null) return null;
  if (typeof v === "number") return v;
  if (typeof v === "object" && typeof v.value === "number") return v.value;
  return null;
}}
function makeJsonBlock(title, obj, subtitle="") {{
  const b = document.createElement("div");
  b.className = "jsonBlock";
  const h = document.createElement("div");
  h.className = "jsonHead";
  h.innerHTML = `<div><div class="t">${{esc(title)}}</div><div class="m">${{esc(subtitle)}}</div></div><div class="m">click</div>`;
  const pre = document.createElement("pre");
  pre.className = "json";
  pre.textContent = pretty(obj);
  h.onclick = () => b.classList.toggle("open");
  b.appendChild(h); b.appendChild(pre);
  return b;
}}
function kv(k, vHtml) {{
  const d = document.createElement("div");
  d.className = "kv";
  d.innerHTML = `<div class="k">${{esc(k)}}</div><div class="v">${{vHtml}}</div>`;
  return d;
}}

/* ----------------------------- state ----------------------------- */
const episodes = parseEpisodes();
let filtered = episodes.slice();
let selectedIndex = null; // index in filtered
let currentFraming = null;
let currentView = "episode"; // episode | systems | council

/* ----------------------------- filtering + grid ----------------------------- */
function applyFilters() {{
  const q = (document.getElementById("searchBox").value || "").trim().toLowerCase();
  const highOnly = document.getElementById("chipHigh").classList.contains("active");
  const councilOnly = document.getElementById("chipCouncil").classList.contains("active");

  filtered = episodes.filter(ep => {{
    const name = (ep.episode_name || "").toLowerCase();
    const domain = String(safe(ep,"context.domain","")).toLowerCase();
    const stakes = String(safe(ep,"context.stakes_level","")).toLowerCase();
    const outcome = String(safe(ep,"full_evaluation.decision.outcome","")).toLowerCase();
    const score = clamp01(safe(ep,"full_evaluation.decision.score",0));
    const risk = riskLevel(score);

    const matchesQuery = !q || (name+" "+domain+" "+stakes+" "+outcome).includes(q);
    const matchesRisk = !highOnly || (risk === "high" || risk === "critical");
    const matchesCouncil = !councilOnly || hasCouncil(ep);

    return matchesQuery && matchesRisk && matchesCouncil;
  }});

  if (selectedIndex != null && (selectedIndex < 0 || selectedIndex >= filtered.length)) {{
    selectedIndex = null; currentFraming = null; currentView = "episode";
    hideViewer();
  }}
  renderGrid();
}}

function renderGrid() {{
  const grid = document.getElementById("episodeGrid");
  grid.innerHTML = "";

  if (filtered.length === 0) {{
    const d = document.createElement("div");
    d.className = "small";
    d.style.padding = "10px 4px";
    d.textContent = "No episodes match the current filters.";
    grid.appendChild(d);
    return;
  }}

  filtered.forEach((ep, idx) => {{
    const score = clamp01(safe(ep,"full_evaluation.decision.score",0));
    const rl = riskLevel(score);
    const domain = safe(ep,"context.domain","unknown");
    const stakes = safe(ep,"context.stakes_level","unknown");
    const outcome = safe(ep,"full_evaluation.decision.outcome","unknown");
    const framings = safe(ep,"full_evaluation.framings_evaluated",[]);
    const councilMark = hasCouncil(ep) ? " · council" : "";

    const c = document.createElement("div");
    c.className = "card" + (idx === selectedIndex ? " active" : "");
    c.onclick = () => selectEpisode(idx);
    c.innerHTML = `
      <div class="title">${{esc(ep.episode_name || "Unnamed Episode")}}</div>
      <div class="meta">Domain: ${{esc(domain)}} | Stakes: ${{esc(stakes)}} | Outcome: ${{esc(outcome)}}${{councilMark}}</div>
      <div class="meta">Framings: ${{Array.isArray(framings) ? framings.length : Object.keys(safe(ep,"full_evaluation.responses",{{}}) || {{}}).length}}</div>
      <span class="badge ${{rl}}">${{rl.toUpperCase()}} (${{fmt(score,3)}})</span>
    `;
    grid.appendChild(c);
  }});
}}

function showViewer() {{
  document.getElementById("viewer").style.display = "grid";
  document.getElementById("emptyState").style.display = "none";
}}
function hideViewer() {{
  document.getElementById("viewer").style.display = "none";
  document.getElementById("emptyState").style.display = "block";
}}
function currentEpisode() {{
  if (selectedIndex == null) return null;
  if (selectedIndex < 0 || selectedIndex >= filtered.length) return null;
  return filtered[selectedIndex];
}}

/* ----------------------------- selection + framings ----------------------------- */
function selectEpisode(idx) {{
  selectedIndex = idx;
  currentView = "episode";

  const ep = currentEpisode();
  if (!ep) {{ hideViewer(); return; }}

  const framings = safe(ep,"full_evaluation.framings_evaluated",null);
  const responses = safe(ep,"full_evaluation.responses",{{}}) || {{}};
  const framingList = (Array.isArray(framings) && framings.length) ? framings : Object.keys(responses);

  currentFraming = framingList && framingList.length ? framingList[0] : "neutral";

  showViewer();
  renderGrid();
  renderFramingTabs();
  renderChat();
  renderViewTabs();
  renderView();
}}

function renderFramingTabs() {{
  const ep = currentEpisode();
  const wrap = document.getElementById("framingTabs");
  wrap.innerHTML = "";
  if (!ep) return;

  const responses = safe(ep,"full_evaluation.responses",{{}}) || {{}};
  const framings = safe(ep,"full_evaluation.framings_evaluated",null);
  const list = (Array.isArray(framings) && framings.length) ? framings : Object.keys(responses);

  if (!list || !list.length) {{
    const d = document.createElement("div"); d.className="small"; d.textContent="No framings available.";
    wrap.appendChild(d); return;
  }}
  if (!list.includes(currentFraming)) currentFraming = list[0];

  list.forEach(f => {{
    const b = document.createElement("button");
    b.className = "framingTab" + (f === currentFraming ? " active" : "");
    b.textContent = f.charAt(0).toUpperCase() + f.slice(1);
    b.onclick = () => {{ currentFraming = f; renderFramingTabs(); renderChat(); }};
    wrap.appendChild(b);
  }});
}}

function renderChat() {{
  const ep = currentEpisode();
  const container = document.getElementById("chatMessages");
  container.innerHTML = "";
  if (!ep) return;

  const prompt = safe(ep,"core_prompt","");
  const responses = safe(ep,"full_evaluation.responses",{{}}) || {{}};
  const assistant = responses[currentFraming] ?? responses["neutral"] ?? "";

  const user = document.createElement("div");
  user.className = "msg user";
  user.innerHTML = `<div class="bubble">${{esc(prompt)}}</div>`;
  container.appendChild(user);

  const bot = document.createElement("div");
  bot.className = "msg assistant";
  bot.innerHTML = `<div class="bubble">${{esc(assistant)}}</div>`;
  container.appendChild(bot);
}}

/* ----------------------------- view tabs ----------------------------- */
function renderViewTabs() {{
  const tabs = document.getElementById("viewTabs");
  tabs.innerHTML = "";

  const views = [
    {{id:"episode", label:"Episode View"}},
    {{id:"systems", label:"Systems View"}},
    {{id:"council", label:"Council Internals"}},
  ];

  views.forEach(v => {{
    const t = document.createElement("div");
    t.className = "tab" + (v.id === currentView ? " active" : "");
    t.textContent = v.label;
    t.onclick = () => {{ currentView = v.id; renderViewTabs(); renderView(); }};
    tabs.appendChild(t);
  }});
}}

function renderView() {{
  const body = document.getElementById("viewBody");
  body.innerHTML = "";
  const ep = currentEpisode();
  if (!ep) return;

  if (currentView === "episode") renderEpisodeView(body, ep);
  else if (currentView === "systems") renderSystemsView(body, ep);
  else renderCouncilInternalsView(body, ep);
}}

/* ----------------------------- Episode View ----------------------------- */
function renderEpisodeView(body, ep) {{
  const fe = safe(ep,"full_evaluation",{{}}) || {{}};
  const ctx = safe(ep,"context",{{}}) || {{}};
  const decision = safe(fe,"decision",{{}}) || {{}};

  body.appendChild(section("Episode Details", "schema-flexible"));
  const g = document.createElement("div"); g.className="grid2";

  const eid = String(safe(ep,"episode_id",""));
  g.appendChild(kv("Episode ID", `<span class="mono">${{esc(eid.slice(0,8))}}…</span>`));
  g.appendChild(kv("Domain", esc(safe(ctx,"domain","unknown"))));
  g.appendChild(kv("Stakes", esc(safe(ctx,"stakes_level","unknown"))));
  g.appendChild(kv("Vulnerability", esc(safe(ctx,"vulnerability", safe(ctx,"vulnerability_level","unknown")))));
  g.appendChild(kv("Interaction", esc(safe(ctx,"interaction_horizon","unknown"))));
  g.appendChild(kv("Outcome", esc(safe(decision,"outcome","unknown"))));
  const score = clamp01(safe(decision,"score",0));
  g.appendChild(kv("Risk score", `<span class="mono">${{fmt(score,4)}}</span>`));
  g.appendChild(kv("Risk level", `<span class="pill">${{riskLevel(score)}}</span>`));
  body.appendChild(g);

  body.appendChild(section("ML Classifiers", "fast screening"));
  body.appendChild(renderML(fe));

  body.appendChild(section("Risk Metrics", "signals + conditioning"));
  body.appendChild(renderRisk(fe));

  body.appendChild(section("Council Verdict", "summary"));
  body.appendChild(renderCouncilSummary(fe));

  body.appendChild(section("Raw JSON", "debugging"));
  body.appendChild(makeJsonBlock("Episode object (top-level)", ep, "Everything for this episode"));
  body.appendChild(makeJsonBlock("full_evaluation", fe, "What the lab produced for this episode"));
}}

function renderML(fe) {{
  const ml = safe(fe,"ml_classifiers",null);
  const wrap = document.createElement("div");

  if (!ml) {{
    wrap.innerHTML = `<div class="small">ML classifier data not available.</div>`;
    return wrap;
  }}

  const grid = document.createElement("div");
  grid.className = "grid2";

  function clsRow(label, node) {{
    const lab = safe(node,"label",null);
    const sc = safe(node,"score",null);
    const probs = safe(node,"probs", null) || safe(node,"probabilities", null);
    const meta = [];
    if (lab != null) meta.push(esc(lab));
    if (sc != null) meta.push(fmt(sc,2));
    if (!meta.length) meta.push("N/A");
    const rhs = `<span class="mono">${{meta.join(" · ")}}</span>`;
    grid.appendChild(kv(label, rhs));
    if (probs) wrap.appendChild(makeJsonBlock(`${{label}} probabilities`, probs, "If present"));
  }}

  clsRow("Sentiment", safe(ml,"sentiment",{{}}));
  clsRow("Intent", safe(ml,"intent",{{}}));
  clsRow("Toxicity", safe(ml,"toxicity",{{}}));
  clsRow("Quality", safe(ml,"quality",{{}}));

  grid.appendChild(kv("ML risk score", `<span class="mono">${{fmt(safe(ml,"ml_risk_score",null),4)}}</span>`));
  grid.appendChild(kv("Total cost", `<span class="mono">${{fmt(safe(ml,"total_cost",null),4)}}</span>`));
  grid.appendChild(kv("Latency (ms)", `<span class="mono">${{esc(safe(ml,"total_latency_ms","N/A"))}}</span>`));

  // model/provider details often get “lost” if you hardcode keys; so we show raw.
  wrap.appendChild(grid);
  wrap.appendChild(makeJsonBlock("ML classifiers (raw)", ml, "Includes model details when present"));
  return wrap;
}}

function renderRisk(fe) {{
  const signals = safe(fe,"signals",{{}}) || {{}};
  const conditioned = safe(fe,"conditioned",{{}}) || {{}};
  const decision = safe(fe,"decision",{{}}) || {{}};

  const g = document.createElement("div");
  g.className = "grid2";

  const items = [
    ["Aggregate risk", safe(conditioned,"aggregate_risk_score",null), 4],
    ["Max risk", safe(conditioned,"max_risk_score",null), 4],
    ["Manipulation risk", safe(signals,"manipulation_risk_score",null), 3],
    ["Max signal severity", safe(signals,"max_signal_severity",null), null],
    ["Intent divergence", safe(signals,"intent_divergence",null), 3],
    ["Framing sensitivity", safe(signals,"framing_sensitivity",null), 3],
    ["Oversight gap", safe(signals,"oversight_gap",null), 3],
    ["Sycophancy index", safe(signals,"sycophancy_index",null), 3],
    ["Concealed capability", safe(signals,"concealed_capability",null), 3],
    ["Decision score", safe(decision,"score",null), 4],
  ];
  items.forEach(([k,v,d]) => {{
    const rhs = (d == null) ? esc(v ?? "N/A") : `<span class="mono">${{fmt(v,d)}}</span>`;
    g.appendChild(kv(k, rhs));
  }});

  const wrap = document.createElement("div");
  wrap.appendChild(g);
  wrap.appendChild(makeJsonBlock("Signals (raw)", signals, "Derived from metrics + cross-framing deltas"));
  wrap.appendChild(makeJsonBlock("Conditioned (raw)", conditioned, "Risk-weighted aggregation"));
  wrap.appendChild(makeJsonBlock("Decision (raw)", decision, "Thresholding + recommended actions/concerns"));
  return wrap;
}}

function renderCouncilSummary(fe) {{
  const verdict = safe(fe,"council_verdict",null);
  const wrap = document.createElement("div");
  if (!verdict) {{
    wrap.innerHTML = `<div class="small">Council evaluation not available.</div>`;
    return wrap;
  }}

  const decision =
    safe(verdict,"consensus_decision",null) ??
    safe(verdict,"decision",null) ??
    safe(verdict,"consensus",null) ??
    "unknown";

  const risk =
    safe(verdict,"consensus_risk_score",null) ??
    safe(verdict,"risk_score",null) ??
    safe(verdict,"score",null) ?? 0;

  const conf =
    safe(verdict,"consensus_confidence",null) ??
    safe(verdict,"confidence",null);

  const num =
    safe(verdict,"num_judges",null) ??
    safe(verdict,"n_judges",null) ??
    (Array.isArray(safe(verdict,"judge_reports",null)) ? safe(verdict,"judge_reports",[]).length : null) ??
    "N/A";

  const unanimous = safe(verdict,"unanimous",null);
  const disag = safe(verdict,"disagreement_score",null);
  const concerns = safe(verdict,"concerns",null) ?? safe(verdict,"reasoning",null) ?? [];

  const g = document.createElement("div");
  g.className = "grid2";
  g.appendChild(kv("Consensus decision", `<span class="pill">${{esc(String(decision))}}</span>`));
  g.appendChild(kv("Consensus risk", `<span class="mono">${{(clamp01(risk)*100).toFixed(1)}}%</span>`));
  g.appendChild(kv("Judges", `<span class="mono">${{esc(num)}}</span>`));
  g.appendChild(kv("Unanimous", `<span class="mono">${{unanimous==null ? "N/A" : (unanimous ? "Yes" : "No")}}</span>`));
  g.appendChild(kv("Confidence", `<span class="mono">${{conf==null ? "N/A" : ((clamp01(conf)*100).toFixed(1)+"%")}}</span>`));
  g.appendChild(kv("Disagreement", `<span class="mono">${{disag==null ? "N/A" : fmt(disag,3)}}</span>`));
  wrap.appendChild(g);

  if (Array.isArray(concerns) && concerns.length) {{
    const ul = document.createElement("ul");
    ul.className = "small";
    ul.style.margin = "10px 0 0 16px";
    ul.style.padding = "0";
    concerns.slice(0,8).forEach(c => {{
      const li = document.createElement("li");
      li.style.margin = "0 0 6px 0";
      li.textContent = String(c);
      ul.appendChild(li);
    }});
    wrap.appendChild(ul);
  }}

  wrap.appendChild(makeJsonBlock("Council verdict (raw)", verdict, "Schema may vary; raw keeps everything"));
  return wrap;
}}

/* ----------------------------- Systems View ----------------------------- */
function renderSystemsView(body, ep) {{
  const fe = safe(ep,"full_evaluation",{{}}) || {{}};

  body.appendChild(section("Systems View", "pipeline + cross-framing"));
  const small = document.createElement("div");
  small.className = "small";
  small.textContent = "This view tries to render cross-framing + pipeline fields if present, and always provides raw dumps.";
  body.appendChild(small);

  // metrics_by_framing: show table if possible
  body.appendChild(section("Metrics by framing", "raw snapshots"));
  const mbf = safe(fe,"metrics_by_framing",null);
  if (!mbf) {{
    body.appendChild(textSmall("No metrics_by_framing found in JSON."));
  }} else {{
    body.appendChild(renderMetricsByFraming(mbf));
    body.appendChild(makeJsonBlock("metrics_by_framing (raw)", mbf, "Full per-framing metric objects"));
  }}

  // cross_analysis: show raw
  body.appendChild(section("Cross-framing analysis", "pairwise comparisons"));
  const cross = safe(fe,"cross_analysis",null);
  if (!cross) body.appendChild(textSmall("No cross_analysis found (expected full_evaluation.cross_analysis)."));
  else body.appendChild(makeJsonBlock("cross_analysis (raw)", cross, "Comparisons, concerns, sensitivity"));

  // pipeline
  body.appendChild(section("Pipeline execution", "component traces"));
  const pipeRes = safe(fe,"pipeline_results",null);
  const pipeRep = safe(fe,"pipeline_risk_reports",null);
  const pipeCmp = safe(fe,"pipeline_framing_comparison",null);
  const pipeAgg = safe(fe,"pipeline_aggregate_risk",null);

  if (pipeAgg != null) body.appendChild(kv("Pipeline aggregate risk", `<span class="mono">${{fmt(pipeAgg,4)}}</span>`));
  if (!pipeRes && !pipeRep && !pipeCmp) {{
    body.appendChild(textSmall("No pipeline_* fields found. If you have them in episodes.json, they will render here."));
  }} else {{
    if (pipeCmp) body.appendChild(makeJsonBlock("pipeline_framing_comparison", pipeCmp, "Differences across framings"));
    if (pipeRep) body.appendChild(makeJsonBlock("pipeline_risk_reports", pipeRep, "Risk propagation through components"));
    if (pipeRes) body.appendChild(makeJsonBlock("pipeline_results", pipeRes, "Raw component outputs"));
  }}

  // Diagrams
  body.appendChild(section("Risk propagation diagrams", "explainability"));
  body.appendChild(riskPropagationDiagram(ep));
  body.appendChild(pipelineDiagram(pipeRep, pipeRes));
}}

function renderMetricsByFraming(mbf) {{
  // mbf shape can be:
  // - {framing: {metric_name: float, ...}, ...}
  // - {framing: {metric_name: {value: float, ...}, ...}, ...}
  const framings = Object.keys(mbf || {{}});
  const keys = new Set();
  framings.forEach(f => {{
    const o = mbf[f] || {{}};
    Object.keys(o).forEach(k => keys.add(k));
  }});
  const metricKeys = Array.from(keys).sort();

  const table = document.createElement("table");
  table.className = "table";
  const thead = document.createElement("thead");
  const hr = document.createElement("tr");
  hr.innerHTML = `<th>Metric</th>` + framings.map(f => `<th>${{esc(f)}}</th>`).join("");
  thead.appendChild(hr);
  table.appendChild(thead);

  const tbody = document.createElement("tbody");
  metricKeys.forEach(k => {{
    const tr = document.createElement("tr");
    const cells = framings.map(f => {{
      const raw = safe(mbf, `${{f}}.${{k}}`, null);
      const n = metricToNumber(raw);
      const v = (n == null) ? "—" : fmt(n,3);
      return `<td class="mono">${{v}}</td>`;
    }});
    tr.innerHTML = `<th>${{esc(k)}}</th>` + cells.join("");
    tbody.appendChild(tr);
  }});
  table.appendChild(tbody);
  return table;
}}

function riskPropagationDiagram(ep) {{
  const fe = safe(ep,"full_evaluation",{{}}) || {{}};
  const signals = safe(fe,"signals",{{}}) || {{}};
  const conditioned = safe(fe,"conditioned",{{}}) || {{}};
  const decision = safe(fe,"decision",{{}}) || {{}};

  const agg = safe(conditioned,"aggregate_risk_score",null);
  const maxr = safe(conditioned,"max_risk_score",null);
  const mr = safe(signals,"manipulation_risk_score",null);
  const fs = safe(signals,"framing_sensitivity",null);
  const sy = safe(signals,"sycophancy_index",null);
  const out = safe(decision,"outcome",null);

  const wrap = document.createElement("div");
  wrap.className = "diagramWrap";
  wrap.innerHTML = `
    <div class="diagramTitle">Behavior → Signals → Conditioning → Decision</div>
    <svg class="diagram" viewBox="0 0 900 280" preserveAspectRatio="xMinYMin meet">
      <defs>
        <marker id="arrow" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto" markerUnits="strokeWidth">
          <path d="M0,0 L10,3 L0,6 Z" fill="#94a3b8"></path>
        </marker>
      </defs>

      <rect class="node" x="30" y="55" rx="12" ry="12" width="230" height="170"></rect>
      <text class="nodeLabel" x="50" y="85">Behavioral metrics</text>
      <text class="nodeSub" x="50" y="110">metrics_by_framing</text>
      <text class="nodeSub" x="50" y="130">LLM + ML + heuristics</text>

      <rect class="node" x="330" y="55" rx="12" ry="12" width="230" height="170"></rect>
      <text class="nodeLabel" x="350" y="85">Derived signals</text>
      <text class="nodeSub" x="350" y="110">manip=${{fmt(mr,3)}}</text>
      <text class="nodeSub" x="350" y="130">framing=${{fmt(fs,3)}}</text>
      <text class="nodeSub" x="350" y="150">sycoph=${{fmt(sy,3)}}</text>

      <rect class="node" x="630" y="55" rx="12" ry="12" width="240" height="170"></rect>
      <text class="nodeLabel" x="650" y="85">Risk conditioning</text>
      <text class="nodeSub" x="650" y="110">aggregate=${{fmt(agg,4)}}</text>
      <text class="nodeSub" x="650" y="130">max=${{fmt(maxr,4)}}</text>
      <text class="nodeSub" x="650" y="150">outcome=${{esc(out ?? "N/A")}}</text>

      <line class="edge" x1="260" y1="140" x2="330" y2="140"></line>
      <line class="edge" x1="560" y1="140" x2="630" y2="140"></line>
    </svg>
  `;
  return wrap;
}}

function pipelineDiagram(pipeReports, pipeResults) {{
  const wrap = document.createElement("div");
  wrap.className = "diagramWrap";
  wrap.innerHTML = `<div class="diagramTitle">Pipeline propagation (best-effort)</div>`;

  if (!pipeReports && !pipeResults) {{
    wrap.appendChild(textSmall("No pipeline data present. Once pipeline_results/pipeline_risk_reports exist, this diagram will render nodes per component."));
    return wrap;
  }}

  function inferComponents(fr) {{
    const r = (pipeReports && pipeReports[fr]) ? pipeReports[fr] : null;
    const pr = (pipeResults && pipeResults[fr]) ? pipeResults[fr] : null;
    const comps = [];

    if (r && Array.isArray(r.components)) {{
      r.components.forEach(c => comps.push({{name:c.name||c.component_name||"component", risk:c.aggregate_risk ?? c.risk ?? null}}));
    }} else if (r && r.component_risks && typeof r.component_risks === "object") {{
      Object.entries(r.component_risks).forEach(([name, rv]) => comps.push({{name, risk:rv}}));
    }}
    if (!comps.length && pr && Array.isArray(pr.trace)) {{
      pr.trace.forEach(t => comps.push({{name:t.component||t.name||"component", risk:t.risk ?? null}}));
    }}
    return comps;
  }}

  const framings = [];
  if (pipeReports && typeof pipeReports === "object") framings.push(...Object.keys(pipeReports));
  else if (pipeResults && typeof pipeResults === "object") framings.push(...Object.keys(pipeResults));

  const fr = framings.includes("neutral") ? "neutral" : (framings[0] || null);
  const comps = fr ? inferComponents(fr) : [];

  if (!fr || !comps.length) {{
    wrap.appendChild(textSmall("Pipeline data present but schema not recognized. See raw JSON blocks above."));
    return wrap;
  }}

  const width = Math.max(700, 180 + comps.length * 160);
  const height = 220;

  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("class", "diagram");
  svg.setAttribute("viewBox", `0 0 ${{width}} ${{height}}`);

  const defs = document.createElementNS(svg.namespaceURI, "defs");
  defs.innerHTML = `
    <marker id="arrowP" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto" markerUnits="strokeWidth">
      <path d="M0,0 L10,3 L0,6 Z" fill="#94a3b8"></path>
    </marker>`;
  svg.appendChild(defs);

  function addRect(x,y,w,h) {{
    const r = document.createElementNS(svg.namespaceURI, "rect");
    r.setAttribute("x", x); r.setAttribute("y", y);
    r.setAttribute("width", w); r.setAttribute("height", h);
    r.setAttribute("rx", 12); r.setAttribute("ry", 12);
    r.setAttribute("class", "node");
    svg.appendChild(r);
  }}
  function addText(x,y,txt,cls) {{
    const t = document.createElementNS(svg.namespaceURI, "text");
    t.setAttribute("x", x); t.setAttribute("y", y);
    t.setAttribute("class", cls);
    t.textContent = txt;
    svg.appendChild(t);
  }}
  function addLine(x1,y1,x2,y2) {{
    const l = document.createElementNS(svg.namespaceURI, "line");
    l.setAttribute("x1", x1); l.setAttribute("y1", y1);
    l.setAttribute("x2", x2); l.setAttribute("y2", y2);
    l.setAttribute("stroke", "#94a3b8");
    l.setAttribute("stroke-width", "1.4");
    l.setAttribute("marker-end", "url(#arrowP)");
    svg.appendChild(l);
  }}

  addText(10, 18, `Framing: ${{fr}}`, "nodeSub");

  const startX = 30, nodeW = 140, nodeH = 110, gap = 20, y = 50;
  comps.forEach((c, i) => {{
    const x = startX + i*(nodeW+gap);
    addRect(x, y, nodeW, nodeH);
    addText(x+10, y+28, String(c.name).slice(0,18), "nodeLabel");
    addText(x+10, y+52, `risk=${{fmt(c.risk,3)}}`, "nodeSub");
    if (i > 0) {{
      const px = startX + (i-1)*(nodeW+gap);
      addLine(px+nodeW, y+nodeH/2, x, y+nodeH/2);
    }}
  }});

  wrap.appendChild(svg);
  wrap.appendChild(textSmall("Diagram inferred from pipeline_risk_reports/pipeline_results. Raw JSON blocks are authoritative."));
  return wrap;
}}

/* ----------------------------- Council Internals View ----------------------------- */
function renderCouncilInternalsView(body, ep) {{
  const fe = safe(ep,"full_evaluation",{{}}) || {{}};
  const verdict = safe(fe,"council_verdict",null);

  body.appendChild(section("Council Internals", "per-judge when available"));

  if (!verdict) {{
    body.appendChild(textSmall("No council verdict present for this episode."));
    return;
  }}

  // always show summary (people expect it here too)
  body.appendChild(renderCouncilSummary(fe));

  // look for per-judge details in common locations
  const judgeReports =
    safe(verdict,"judge_reports",null) ??
    safe(verdict,"evaluator_responses",null) ??
    safe(verdict,"reports",null) ??
    null;

  if (!Array.isArray(judgeReports) || judgeReports.length === 0) {{
    body.appendChild(textSmall("No per-judge reports found in council_verdict. If you include judge_reports/evaluator_responses in JSON, they will render here."));
    body.appendChild(makeJsonBlock("Council verdict (raw)", verdict, "Inspect to see what fields exist"));
    return;
  }}

  body.appendChild(section("Judge reports", `${{judgeReports.length}} judges`));

  judgeReports.forEach((jr, idx) => {{
    const id = safe(jr,"judge_id",null) ?? safe(jr,"id",null) ?? (idx+1);
    const rs = safe(jr,"risk_score",null) ?? safe(jr,"score",null);
    const cf = safe(jr,"confidence",null);

    const block = makeJsonBlock(`Judge ${{id}}`, jr, `risk=${{fmt(rs,3)}} · conf=${{fmt(cf,2)}}`);
    block.classList.add("open"); // default expanded for internals
    body.appendChild(block);
  }});
}}

/* ----------------------------- helpers for DOM ----------------------------- */
function section(title, hint) {{
  const d = document.createElement("div");
  d.className = "sectionTitle";
  d.innerHTML = `<span>${{esc(title)}}</span><span class="hint">${{esc(hint || "")}}</span>`;
  return d;
}}
function textSmall(s) {{
  const d = document.createElement("div");
  d.className = "small";
  d.style.marginTop = "6px";
  d.textContent = s;
  return d;
}}

/* ----------------------------- init ----------------------------- */
function init() {{
  document.getElementById("searchBox").addEventListener("input", applyFilters);
  document.getElementById("chipHigh").onclick = () => {{ document.getElementById("chipHigh").classList.toggle("active"); applyFilters(); }};
  document.getElementById("chipCouncil").onclick = () => {{ document.getElementById("chipCouncil").classList.toggle("active"); applyFilters(); }};

  renderGrid();
  if (!episodes || episodes.length === 0) {{
    document.getElementById("emptyState").innerHTML = "<div style='padding:10px'>No episodes found in episodes.json.</div>";
  }}
}}
document.addEventListener("DOMContentLoaded", init);
</script>
</body>
</html>"""


def generate_episode_viewer(output_dir: Path, episodes_file: Path) -> Path:
    """Generate interactive episode viewer."""
    viewer = EpisodeViewer(output_dir)
    return viewer.generate_episode_viewer(episodes_file)