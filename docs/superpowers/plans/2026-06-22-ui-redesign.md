# UI Redesign — Radial Tree + Dark Mode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transform the basic tree visualization into a modern radial tree with dark mode, animations, and enhanced side panel.

**Architecture:** Rewrite `style.css` for dark theme, rewrite `app.js` for radial D3 layout with interactions, add metrics endpoint to `server.py`. Minimal changes to `index.html`.

**Tech Stack:** D3.js v7, vanilla CSS, FastAPI/Python

## Global Constraints

- Python pinned to 3.12
- Embeddings are L2-normalized (N, d) float32
- Labels optional everywhere
- Tool emits NO verdict
- Desktop-only (no mobile responsiveness)
- Animation speed: 250ms (fast, responsive)

---

## Task 1: Add metrics endpoint to server.py

**Files:**
- Modify: `src/gyo/api/server.py`
- Test: `tests/test_server.py`

**Interfaces:**
- Consumes: `node_stats`, `build_tree` from existing modules
- Produces: `GET /api/node/{prefix}/metrics` → JSON with node metrics + label distribution

- [ ] **Step 1: Write the failing test**

`tests/test_server.py` — append:

```python
def test_node_metrics_endpoint(tmp_path):
    _seed_run(tmp_path)
    client = TestClient(create_app(str(tmp_path)))
    r = client.get("/api/node/0/metrics")
    assert r.status_code == 200
    body = r.json()
    assert "occupancy" in body
    assert "mean_residual" in body
    assert "label_distribution" in body
    assert body["occupancy"] == 2  # nodes 0 and 1 have c_0=0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_server.py::test_node_metrics_endpoint -v`
Expected: FAIL with 404

- [ ] **Step 3: Write minimal implementation**

In `src/gyo/api/server.py`, add inside `create_app`:

```python
@app.get("/api/node/{prefix}/metrics")
def node_metrics(prefix: str):
    codes, final_res, labels, meta_df = _load()
    root = build_tree(codes, final_res, labels)
    pfx = () if prefix == "root" else tuple(int(p) for p in prefix.split(","))
    target = node_at(root, pfx)
    if target is None:
        raise HTTPException(404, "prefix not found")
    from gyo.tree.signals import node_stats
    stats = node_stats(root, labels)[0]
    # Find the specific node's stats
    all_stats = node_stats(root, labels)
    node_stat = next((s for s in all_stats if s.prefix == pfx), None)
    if node_stat is None:
        raise HTTPException(404, "node stats not found")
    # Label distribution
    from collections import Counter
    label_dist = {}
    if labels:
        counts = Counter(labels[i] for i in target.item_indices)
        label_dist = dict(counts.most_common(20))
    return {
        "prefix": list(pfx),
        "level": node_stat.level,
        "occupancy": node_stat.occupancy,
        "mean_residual": node_stat.mean_residual,
        "residual_norm": node_stat.residual_norm,
        "size_norm": node_stat.size_norm,
        "purity": node_stat.purity,
        "label_distribution": label_dist,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_server.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/gyo/api/server.py tests/test_server.py
git commit -m "feat: add /api/node/{prefix}/metrics endpoint with label distribution"
```

---

## Task 2: Rewrite style.css for dark mode

**Files:**
- Modify: `src/gyo/web/style.css`

**Interfaces:**
- Consumes: nothing
- Produces: Dark theme CSS with all required classes

- [ ] **Step 1: Write the CSS**

Replace `src/gyo/web/style.css` entirely:

```css
/* ===== Variables ===== */
:root {
  --bg-primary: #0f0f1a;
  --bg-secondary: #1a1a2e;
  --bg-tertiary: #252540;
  --text-primary: #e0e0e0;
  --text-secondary: #888;
  --accent-blue: #4a9eff;
  --accent-cyan: #00d4aa;
  --accent-pink: #ff6b6b;
  --border-color: #333;
  --glow-blue: rgba(74, 158, 255, 0.4);
}

/* ===== Reset & Base ===== */
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
  font-family: 'Inter', system-ui, -apple-system, sans-serif;
  background: var(--bg-primary);
  color: var(--text-primary);
  min-height: 100vh;
}

/* ===== Header ===== */
header {
  padding: 12px 20px;
  border-bottom: 1px solid var(--border-color);
  display: flex;
  align-items: center;
  gap: 20px;
  background: var(--bg-secondary);
}
header h1 {
  font-size: 18px;
  font-weight: 600;
  color: var(--accent-blue);
}
.disclaimer {
  color: var(--text-secondary);
  font-size: 12px;
  flex: 1;
}
.controls {
  display: flex;
  align-items: center;
  gap: 12px;
}
.controls label {
  font-size: 13px;
  color: var(--text-secondary);
}
.controls input[type="range"] {
  width: 100px;
  accent-color: var(--accent-blue);
}
.controls span {
  font-size: 13px;
  color: var(--accent-blue);
  min-width: 20px;
}
button {
  background: var(--bg-tertiary);
  border: 1px solid var(--border-color);
  color: var(--text-primary);
  padding: 6px 12px;
  border-radius: 4px;
  cursor: pointer;
  font-size: 12px;
  transition: all 0.15s ease;
}
button:hover {
  background: var(--accent-blue);
  border-color: var(--accent-blue);
}

/* ===== Main Layout ===== */
main {
  display: flex;
  height: calc(100vh - 50px);
}

/* ===== Tree SVG ===== */
#tree {
  flex: 1;
  background: var(--bg-primary);
}

/* ===== Side Panel ===== */
#panel {
  width: 320px;
  background: var(--bg-secondary);
  border-left: 1px solid var(--border-color);
  display: flex;
  flex-direction: column;
  overflow: hidden;
}
#panel-header {
  padding: 12px 16px;
  border-bottom: 1px solid var(--border-color);
  font-size: 14px;
  font-weight: 600;
}
#panel-header .prefix {
  color: var(--accent-cyan);
  font-family: monospace;
}
#panel-header .count {
  color: var(--accent-pink);
}
#panel-metrics {
  padding: 12px 16px;
  border-bottom: 1px solid var(--border-color);
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 8px;
}
.metric {
  display: flex;
  flex-direction: column;
  gap: 2px;
}
.metric-label {
  font-size: 10px;
  color: var(--text-secondary);
  text-transform: uppercase;
  letter-spacing: 0.5px;
}
.metric-value {
  font-size: 14px;
  font-weight: 600;
  font-family: monospace;
}
.metric-bar {
  height: 4px;
  background: var(--bg-tertiary);
  border-radius: 2px;
  overflow: hidden;
  margin-top: 4px;
}
.metric-bar-fill {
  height: 100%;
  border-radius: 2px;
  transition: width 0.25s ease;
}
#panel-grid {
  flex: 1;
  overflow-y: auto;
  padding: 8px;
  display: grid;
  grid-template-columns: repeat(5, 1fr);
  gap: 4px;
  align-content: start;
}
#panel-grid::-webkit-scrollbar {
  width: 6px;
}
#panel-grid::-webkit-scrollbar-track {
  background: var(--bg-tertiary);
}
#panel-grid::-webkit-scrollbar-thumb {
  background: var(--border-color);
  border-radius: 3px;
}
.thumb {
  aspect-ratio: 1;
  border: 1px solid var(--border-color);
  border-radius: 4px;
  overflow: hidden;
  cursor: pointer;
  transition: all 0.15s ease;
  position: relative;
}
.thumb:hover {
  border-color: var(--accent-blue);
  box-shadow: 0 0 8px var(--glow-blue);
  transform: scale(1.05);
}
.thumb img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  image-rendering: pixelated;
}
.thumb .label {
  position: absolute;
  bottom: 0;
  left: 0;
  right: 0;
  background: rgba(0, 0, 0, 0.8);
  font-size: 8px;
  text-align: center;
  padding: 2px;
  color: var(--text-primary);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

/* ===== Tooltip ===== */
.tooltip {
  position: fixed;
  background: var(--bg-secondary);
  border: 1px solid var(--border-color);
  border-radius: 6px;
  padding: 10px 14px;
  pointer-events: none;
  z-index: 1000;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);
  font-size: 12px;
  max-width: 250px;
  opacity: 0;
  transition: opacity 0.15s ease;
}
.tooltip.visible {
  opacity: 1;
}
.tooltip-title {
  font-weight: 600;
  color: var(--accent-blue);
  margin-bottom: 6px;
  font-family: monospace;
}
.tooltip-row {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  margin: 2px 0;
}
.tooltip-label {
  color: var(--text-secondary);
}
.tooltip-value {
  color: var(--text-primary);
  font-family: monospace;
}

/* ===== Legend Dialog ===== */
dialog {
  background: var(--bg-secondary);
  border: 1px solid var(--border-color);
  border-radius: 8px;
  color: var(--text-primary);
  padding: 20px;
  max-width: 400px;
}
dialog::backdrop {
  background: rgba(0, 0, 0, 0.7);
}
dialog h3 {
  margin-bottom: 12px;
  color: var(--accent-blue);
}
dialog ul {
  list-style: none;
  padding: 0;
}
dialog li {
  padding: 6px 0;
  font-size: 13px;
  color: var(--text-secondary);
  border-bottom: 1px solid var(--border-color);
}
dialog li:last-child {
  border-bottom: none;
}
dialog form {
  margin-top: 16px;
  text-align: right;
}
```

- [ ] **Step 2: Verify visual rendering**

Run: `uv run gyo serve --data-dir run --port 8000`
Open http://localhost:8000 — verify dark background, blue accents, styled panel

- [ ] **Step 3: Commit**

```bash
git add src/gyo/web/style.css
git commit -m "feat: dark mode CSS with radial layout styles, panel, tooltip"
```

---

## Task 3: Update index.html structure

**Files:**
- Modify: `src/gyo/web/index.html`

**Interfaces:**
- Consumes: CSS classes from Task 2
- Produces: Updated HTML structure

- [ ] **Step 1: Rewrite index.html**

Replace `src/gyo/web/index.html` entirely:

```html
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>gyo — RQ embedding inspector</title>
  <link rel="stylesheet" href="/style.css" />
  <script src="https://d3js.org/d3.v7.min.js"></script>
</head>
<body>
  <header>
    <h1>gyo</h1>
    <p class="disclaimer">Embedding signal viewer — size = occupancy, color = residual</p>
    <div class="controls">
      <label>Level: <input id="level" type="range" min="0" max="3" value="1" /></label>
      <span id="levelVal">1</span>
      <button id="legendBtn">Legend</button>
    </div>
  </header>
  <main>
    <svg id="tree"></svg>
    <aside id="panel">
      <div id="panel-header">Click a node to inspect</div>
      <div id="panel-metrics"></div>
      <div id="panel-grid"></div>
    </aside>
  </main>
  <div class="tooltip" id="tooltip"></div>
  <dialog id="legend">
    <h3>Possible readings (you decide — not produced by the tool)</h3>
    <ul>
      <li>Big &amp; cool (low residual): region maybe over-represented in data.</li>
      <li>Big &amp; hot (high residual): embedder maybe collapsing distinct things here.</li>
      <li>Dead codewords / empty: region maybe under-represented.</li>
      <li>Distinct labels in one deep prefix: maybe missing signal to separate them.</li>
      <li>Similar items split already at c_0: maybe nuisance dominating variance.</li>
    </ul>
    <form method="dialog"><button>Close</button></form>
  </dialog>
  <script src="/app.js"></script>
</body>
</html>
```

- [ ] **Step 2: Commit**

```bash
git add src/gyo/web/index.html
git commit -m "feat: updated HTML structure for radial layout and enhanced panel"
```

---

## Task 4: Rewrite app.js for radial layout with interactions

**Files:**
- Modify: `src/gyo/web/app.js`

**Interfaces:**
- Consumes: `/api/tree`, `/api/node/{prefix}`, `/api/node/{prefix}/metrics`, `/thumb/{idx}`
- Produces: Radial tree visualization with zoom, hover, expand/collapse

- [ ] **Step 1: Rewrite app.js**

Replace `src/gyo/web/app.js` entirely:

```javascript
// ===== State =====
let currentRoot = null;
let currentZoom = d3.zoomIdentity;
const duration = 250;

// ===== Color Scale =====
const color = d3.scaleSequential(d3.interpolateRdYlBu).domain([1, 0]);

// ===== DOM Elements =====
const svg = d3.select("#tree");
const panel = d3.select("#panel");
const tooltip = d3.select("#tooltip");
const levelInput = document.getElementById("level");
const levelVal = document.getElementById("levelVal");
document.getElementById("legendBtn").onclick = () => document.getElementById("legend").showModal();

// ===== Setup SVG =====
const width = window.innerWidth - 320;
const height = window.innerHeight - 50;
svg.attr("width", width).attr("height", height);

const g = svg.append("g");
const zoom = d3.zoom()
  .scaleExtent([0.3, 3])
  .on("zoom", (event) => {
    currentZoom = event.transform;
    g.attr("transform", event.transform);
  });
svg.call(zoom);

// ===== Load Tree =====
async function loadTree(level) {
  const res = await fetch(`/api/tree?level=${level}`);
  const data = await res.json();
  levelInput.max = data.num_levels;
  render(data);
}

// ===== Render Radial Tree =====
function render(data) {
  g.selectAll("*").remove();
  
  // Build hierarchy
  const root = { prefix: [], children: [] };
  const byPrefix = new Map([["", root]]);
  for (const n of data.nodes) byPrefix.set(n.prefix.join(","), { ...n, children: [] });
  for (const n of data.nodes) {
    if (n.prefix.length === 0) continue;
    const parentKey = n.prefix.slice(0, -1).join(",");
    const parent = byPrefix.get(parentKey);
    if (parent) parent.children.push(byPrefix.get(n.prefix.join(",")));
  }
  
  currentRoot = d3.hierarchy(byPrefix.get(""));
  currentRoot.x0 = 0;
  currentRoot.y0 = 0;
  
  // Collapse children by default (except root)
  currentRoot.descendants().forEach((d, i) => {
    d.id = i;
    if (d.depth > 0) {
      d._children = d.children;
      d.children = null;
    }
  });
  
  update(currentRoot);
}

// ===== Update Tree =====
function update(source) {
  const treeLayout = d3.tree()
    .size([2 * Math.PI, Math.min(width, height) / 2 - 60])
    .separation((a, b) => (a.parent === b.parent ? 1 : 2) / a.depth);
  
  treeLayout(currentRoot);
  
  const nodes = currentRoot.descendants();
  const links = currentRoot.links();
  
  // ===== Links =====
  const link = g.selectAll("path.link")
    .data(links, d => d.target.data.prefix.join(","));
  
  const linkEnter = link.enter()
    .append("path")
    .attr("class", "link")
    .attr("fill", "none")
    .attr("stroke", "#333")
    .attr("stroke-opacity", 0.4)
    .attr("stroke-width", 1.5)
    .attr("d", () => {
      const o = { x: source.x0, y: source.y0 };
      return radialLink({ source: o, target: o });
    });
  
  const linkUpdate = linkEnter.merge(link);
  linkUpdate.transition().duration(duration)
    .attr("d", d => radialLink(d));
  
  link.exit().transition().duration(duration)
    .attr("d", () => {
      const o = { x: source.x, y: source.y };
      return radialLink({ source: o, target: o });
    })
    .remove();
  
  // ===== Nodes =====
  const node = g.selectAll("g.node")
    .data(nodes, d => d.data.prefix.join(","));
  
  const nodeEnter = node.enter()
    .append("g")
    .attr("class", "node")
    .attr("transform", () => `rotate(${source.x0 * 180 / Math.PI - 90}) translate(${source.y0},0)`)
    .on("click", (event, d) => {
      if (d.children) {
        d._children = d.children;
        d.children = null;
      } else if (d._children) {
        d.children = d._children;
        d._children = null;
      }
      update(d);
      inspect(d.data.prefix);
    })
    .on("dblclick", (event, d) => {
      event.stopPropagation();
      zoomToNode(d);
    })
    .on("mouseenter", (event, d) => showTooltip(event, d))
    .on("mouseleave", hideTooltip);
  
  nodeEnter.append("circle")
    .attr("r", 0)
    .attr("fill", d => getNodeFill(d))
    .attr("stroke", d => getNodeStroke(d))
    .attr("stroke-width", 2)
    .style("cursor", "pointer");
  
  nodeEnter.append("text")
    .attr("dy", "0.31em")
    .attr("x", d => d.x < Math.PI === !d.children ? 12 : -12)
    .attr("text-anchor", d => d.x < Math.PI === !d.children ? "start" : "end")
    .attr("fill", "#888")
    .attr("font-size", "10px")
    .attr("transform", d => d.x >= Math.PI ? "rotate(180)" : null)
    .text(d => d.data.occupancy || "");
  
  const nodeUpdate = nodeEnter.merge(node);
  
  nodeUpdate.transition().duration(duration)
    .attr("transform", d => `rotate(${d.x * 180 / Math.PI - 90}) translate(${d.y},0)`);
  
  nodeUpdate.select("circle")
    .transition().duration(duration)
    .attr("r", d => d._children ? 8 : 6)
    .attr("fill", d => getNodeFill(d))
    .attr("stroke", d => getNodeStroke(d));
  
  node.exit().transition().duration(duration)
    .attr("transform", () => `rotate(${source.x * 180 / Math.PI - 90}) translate(${source.y},0)`)
    .remove()
    .select("circle").attr("r", 0);
  
  // Store old positions
  nodes.forEach(d => {
    d.x0 = d.x;
    d.y0 = d.y;
  });
}

// ===== Radial Link Generator =====
function radialLink(d) {
  return `M${d.source.y},${d.source.x}
    C${(d.source.y + d.target.y) / 2},${d.source.x}
     ${(d.source.y + d.target.y) / 2},${d.target.x}
     ${d.target.y},${d.target.x}`;
}

// ===== Node Colors =====
function getNodeFill(d) {
  if (d.data.residual_norm == null) return "#333";
  if (d._children) return "#4a9eff";
  return color(d.data.residual_norm);
}

function getNodeStroke(d) {
  if (d.data.residual_norm == null) return "#555";
  return "#fff";
}

// ===== Tooltip =====
function showTooltip(event, d) {
  const data = d.data;
  tooltip.html(`
    <div class="tooltip-title">[${data.prefix.join(",") || "root"}]</div>
    <div class="tooltip-row"><span class="tooltip-label">Occupancy:</span><span class="tooltip-value">${data.occupancy}</span></div>
    <div class="tooltip-row"><span class="tooltip-label">Residual:</span><span class="tooltip-value">${(data.mean_residual || 0).toFixed(4)}</span></div>
    ${data.purity != null ? `<div class="tooltip-row"><span class="tooltip-label">Purity:</span><span class="tooltip-value">${(data.purity * 100).toFixed(1)}%</span></div>` : ""}
  `)
    .style("left", (event.pageX + 15) + "px")
    .style("top", (event.pageY - 10) + "px")
    .classed("visible", true);
}

function hideTooltip() {
  tooltip.classed("visible", false);
}

// ===== Zoom to Node =====
function zoomToNode(d) {
  const scale = 1.5;
  const x = -d.y * scale + width / 2;
  const y = -d.x * scale * 0.5 + height / 2;
  
  svg.transition().duration(duration)
    .call(zoom.transform, d3.zoomIdentity.translate(x, y).scale(scale));
}

// ===== Inspect Node (Side Panel) =====
async function inspect(prefix) {
  const key = prefix.length ? prefix.join(",") : "root";
  
  // Load metrics
  const metricsRes = await fetch(`/api/node/${key}/metrics`);
  const metrics = await metricsRes.json();
  
  // Load items
  const itemsRes = await fetch(`/api/node/${key}`);
  const items = await itemsRes.json();
  
  // Update panel header
  panel.select("#panel-header")
    .html(`<span class="prefix">[${prefix.join(",") || "root"}]</span> — <span class="count">${metrics.occupancy} items</span>`);
  
  // Update metrics
  const residualColor = color(metrics.residual_norm || 0);
  panel.select("#panel-metrics")
    .html(`
      <div class="metric">
        <span class="metric-label">Occupancy</span>
        <span class="metric-value">${metrics.occupancy}</span>
      </div>
      <div class="metric">
        <span class="metric-label">Residual</span>
        <span class="metric-value">${(metrics.mean_residual || 0).toFixed(4)}</span>
        <div class="metric-bar">
          <div class="metric-bar-fill" style="width: ${(metrics.residual_norm || 0) * 100}%; background: ${residualColor}"></div>
        </div>
      </div>
      <div class="metric">
        <span class="metric-label">Purity</span>
        <span class="metric-value">${metrics.purity != null ? (metrics.purity * 100).toFixed(1) + "%" : "N/A"}</span>
      </div>
      <div class="metric">
        <span class="metric-label">Size Norm</span>
        <span class="metric-value">${(metrics.size_norm || 0).toFixed(3)}</span>
      </div>
    `);
  
  // Update thumbnail grid
  const grid = panel.select("#panel-grid");
  grid.html("");
  for (const it of items.items) {
    const thumb = grid.append("div").attr("class", "thumb");
    thumb.append("img").attr("src", `/thumb/${it.idx}`).attr("alt", it.label);
    if (it.label) {
      thumb.append("div").attr("class", "label").text(it.label);
    }
  }
}

// ===== Event Listeners =====
levelInput.oninput = () => {
  levelVal.textContent = levelInput.value;
  loadTree(+levelInput.value);
};

// ===== Init =====
loadTree(+levelInput.value);
```

- [ ] **Step 2: Verify rendering**

Run: `uv run gyo serve --data-dir run --port 8000`
Open http://localhost:8000 — verify:
- Radial tree with root at center
- Nodes colored by residual
- Click to expand/collapse
- Hover shows tooltip
- Double-click zooms
- Side panel shows metrics + thumbnails

- [ ] **Step 3: Commit**

```bash
git add src/gyo/web/app.js
git commit -m "feat: radial tree with zoom, hover tooltips, expand/collapse"
```

---

## Task 5: Run full test suite and verify

**Files:**
- None (verification only)

- [ ] **Step 1: Run all tests**

Run: `uv run pytest -v -m "not slow"`
Expected: ALL PASS

- [ ] **Step 2: Manual verification**

Run: `uv run gyo serve --data-dir run --port 8000`
Open http://localhost:8000 and verify:
- Dark background with blue accents
- Radial tree displays correctly
- Level slider works
- Click nodes to expand/collapse
- Hover for tooltips
- Side panel shows metrics + thumbnails
- Legend dialog opens

- [ ] **Step 3: Commit any fixes**

```bash
git add -A
git commit -m "fix: UI polish and test adjustments"
```

---

## Self-Review

**Spec coverage:**
- ✅ Dark mode with accent colors → Task 2
- ✅ Radial tree layout → Task 4
- ✅ Expand/collapse → Task 4
- ✅ Zoom on double-click → Task 4
- ✅ Tooltips on hover → Task 4
- ✅ Side panel with metrics → Task 1 + Task 4
- ✅ Thumbnail grid → Task 4
- ✅ Animations (250ms) → Task 4
- ✅ Metrics endpoint → Task 1

**Placeholder scan:** No TBD/TODO. All code blocks complete.

**Type consistency:** API responses match frontend consumption. CSS classes consistent across HTML/JS.