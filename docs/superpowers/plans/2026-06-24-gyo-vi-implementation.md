# gyo vi Interface Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the text-tree inspector UI with the `gyo vi` collapse-based image icicle (sliding 3-level window, free-follow+snap scroll, branch identity, responsive), wired to the existing API.

**Architecture:** Modularize `src/gyo/web/` into small ES modules with a **pure, DOM-free layout engine** as the testable core. Rendering writes placements into a single transformed wrapper so depth-sliding moves a CSS transform (no DOM rebuild). The validated prototype `prototypes/mock-vi.html` is the **authoritative source** for every algorithm and CSS rule; each UI task ports a named prototype function and replaces its fake data with API calls.

**Tech Stack:** Python (FastAPI) backend unchanged; vanilla ES modules (no bundler) served statically; **Vitest** (Node) for JS unit tests; pytest for backend; Playwright for behavioral checks.

## Global Constraints

- No backend API schema changes; reuse `GET /api/tree`, `GET /api/node/{prefix}`, `GET /thumb/{idx}` verbatim.
- No Co-Authored-By trailers in commits (user rule).
- Residual color = continuous green→yellow→red on `residual_norm`; reuse the exact `residualColor()` from `prototypes/mock-vi.html`.
- Branch = immediate subtree of the **current focus** (relative); branch hue from the prototype's `BRANCH_COLORS` palette.
- Window = 3 full levels (`WIN = 3`); `PEEK = 24`px.
- Scroll MUST free-follow continuously (no transition while scrolling) and snap to nearest level ~130ms after the last wheel event (animated ~0.22s). Discrete per-notch jumps are NOT acceptable.
- Dead-codeword nodes are not drawn; the aggregate `dead [...]` count stays as a header metric.
- Persist `mode` and window `winTop` in `localStorage`; do NOT persist `focus`/`collapsed`.
- Responsive: recompute layout + window geometry on viewport resize (debounced).
- The prototype is reference-only; do not ship `prototypes/` to the served app.

## File Structure

Created/served under `src/gyo/web/`:

- `index.html` — rewritten shell (header, rail, `#icicle > #scroller`, fades, tooltip, breadcrumb, legend).
- `style.css` — rewritten; port all rules from the prototype `<style>`.
- `js/main.js` — entry module: holds app state, wires modules, initial load.
- `js/api.js` — `fetchTree()`, `fetchNodeItems(prefix)` (cached), `thumbUrl(idx)`.
- `js/model.js` — `buildModel(treeJson)` → root node tree (parent links, prefixes, occ/residual/purity/residual_norm), `depthOf`, `assignBranches(focus)`, `residualColor(t)`.
- `js/layout.js` — **pure** engine: `computeLayout(focus, viewportW, opts)` → `placements[]`; plus `clamp`, `winTopFromScrollPx`, `scrollBounds`. No DOM, no globals.
- `js/render.js` — `renderPlacements(scroller, placements, ctx)`; `fillMosaic(el, items, ctx)`.
- `js/window.js` — sliding-window controller: owns `scrollPx`/`winTop`, wheel free-follow+snap, keys, rail, peek/fades, wrapper transform.
- `js/interactions.js` — click/dblclick/hover/breadcrumb wiring (collapse, drill).
- `js/persist.js` — `loadPrefs()`, `savePrefs(partial)` over `localStorage`.
- `js/__tests__/layout.test.js` — Vitest unit tests for `layout.js`.

Repo root tooling: `package.json`, `vitest.config.js`.
Backend: `src/gyo/api/server.py` gains a static route for `js/` modules; `tests/test_api_node.py` adds the `/api/node` contract test.

---

## PR 1 — Foundation: JS test runner + `/api/node` contract + module serving

**Deliverable:** Vitest runs; a green `/api/node` contract test; server can serve `web/js/*.js` as ES modules. No UI change yet.

### Task 1.1: Add Vitest tooling

**Files:**
- Create: `package.json`
- Create: `vitest.config.js`

- [ ] **Step 1: Create `package.json`**

```json
{
  "name": "gyo-web",
  "private": true,
  "type": "module",
  "scripts": { "test": "vitest run", "test:watch": "vitest" },
  "devDependencies": { "vitest": "^2.1.0" }
}
```

- [ ] **Step 2: Create `vitest.config.js`**

```js
import { defineConfig } from "vitest/config";
export default defineConfig({
  test: { include: ["src/gyo/web/js/__tests__/**/*.test.js"], environment: "node" },
});
```

- [ ] **Step 3: Install and verify the runner exists**

Run: `npm install && npx vitest run`
Expected: vitest runs and reports "No test files found" (exit 0 or the no-tests notice).

- [ ] **Step 4: Commit**

```bash
git add package.json vitest.config.js package-lock.json
git commit -m "build(web): add vitest for JS unit tests"
```

### Task 1.2: `/api/node` contract test (leaf + internal prefix)

**Files:**
- Create: `tests/test_api_node.py`

**Interfaces:**
- Consumes: `gyo.api.server.create_app`, `gyo.data.fashion_mnist.prepare_fashion_mnist`, CLI `extract/fit-rq/encode` (as in `tests/test_e2e.py`).
- Produces: confidence that `/api/node/{prefix}` returns `{items:[{idx,path,label}], occupancy}` for both a leaf prefix and an internal (1-level) prefix.

- [ ] **Step 1: Write the failing test**

```python
import pandas as pd
from PIL import Image
from typer.testing import CliRunner
from fastapi.testclient import TestClient
from gyo.cli import app
from gyo.data.fashion_mnist import prepare_fashion_mnist
from gyo.api.server import create_app

runner = CliRunner()


def _fake(n=120):
    return [(Image.new("L", (28, 28), color=(i * 7) % 255), i % 10) for i in range(n)]


def test_api_node_returns_items_for_leaf_and_internal(tmp_path):
    prepare_fashion_mnist(tmp_path, n=120, dataset=_fake(120))
    runner.invoke(app, ["extract", "--data-dir", str(tmp_path), "--embedder", "dummy"])
    runner.invoke(app, ["fit-rq", "--data-dir", str(tmp_path), "--levels", "3",
                        "--codebook-size", "8", "--iters", "10"])
    runner.invoke(app, ["encode", "--data-dir", str(tmp_path)])
    client = TestClient(create_app(str(tmp_path)))

    tree = client.get("/api/tree?level=3").json()
    internal = next(n for n in tree["nodes"] if len(n["prefix"]) == 1)
    pfx = ",".join(str(c) for c in internal["prefix"])

    body = client.get(f"/api/node/{pfx}").json()
    assert body["occupancy"] == internal["occupancy"]
    assert len(body["items"]) == min(200, internal["occupancy"])
    assert set(body["items"][0]) == {"idx", "path", "label"}

    root = client.get("/api/node/root").json()
    assert root["occupancy"] == 120
```

- [ ] **Step 2: Run to verify it passes (contract already satisfied by current server)**

Run: `uv run pytest tests/test_api_node.py -v`
Expected: PASS. (If it fails, the regression is in `server.py` `/api/node` — fix there before continuing.)

- [ ] **Step 3: Commit**

```bash
git add tests/test_api_node.py
git commit -m "test(api): contract test for /api/node on leaf and internal prefixes"
```

### Task 1.3: Serve `web/js/` ES modules

**Files:**
- Modify: `src/gyo/api/server.py` (the static routes near `appjs`/`style`)

**Interfaces:**
- Produces: `GET /js/{name}` → the file `web/js/{name}` with `application/javascript`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_api_node.py`:

```python
def test_js_module_route(tmp_path, monkeypatch):
    (tmp_path / "codes.parquet")  # not needed; route is static
    import gyo.api.server as srv
    js_dir = srv.WEB / "js"
    js_dir.mkdir(parents=True, exist_ok=True)
    (js_dir / "ping.js").write_text("export const ping = 1;\n")
    client = TestClient(create_app(str(tmp_path)))
    r = client.get("/js/ping.js")
    assert r.status_code == 200
    assert "javascript" in r.headers["content-type"]
    assert "export const ping" in r.text
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_api_node.py::test_js_module_route -v`
Expected: FAIL (404).

- [ ] **Step 3: Add the route**

In `create_app`, alongside the existing `appjs` route:

```python
    @app.get("/js/{name}")
    def js_module(name: str):
        path = WEB / "js" / name
        if ".." in name or not path.exists():
            raise HTTPException(404, "module not found")
        return FileResponse(path, media_type="application/javascript",
                            headers={"Cache-Control": "no-cache"})
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_api_node.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/gyo/api/server.py tests/test_api_node.py
git commit -m "feat(api): serve web/js ES modules"
```

---

## PR 2 — Pure layout engine + unit tests

**Deliverable:** `js/layout.js` (DOM-free) with full Vitest coverage of the partition rules. No UI wiring yet.

### Task 2.1: Layout engine module

**Files:**
- Create: `src/gyo/web/js/layout.js`
- Test: `src/gyo/web/js/__tests__/layout.test.js`

**Interfaces:**
- Consumes: plain node objects `{ code, occ, residual, residual_norm, purity, leaf, collapsed, children:[], isRoot }`.
- Produces:
  - `clamp(v,a,b) → number`
  - `computeLayout(focus, viewportW, { collapsedSet }) → Array<{node, x, w, depth, branch, spine}>` where `branch` is a hex string or `null`, `spine` is boolean. Widths: collapsed/spine slivers fixed `SPINE_W=18`, dead `DEAD_W=12`, expanded siblings share remaining width **equally**; gutters `GAP=max(2,18-depth*6)` between expanded pairs, `SMALL_GAP=2` when either side is a sliver; collapsed slivers ordered first (left). Branch hue assigned at `depth===0` children from `BRANCH_COLORS`, inherited by descendants.
  - `scrollBounds(maxWinTop, rowH, PEEK) → {min, max}` with `max = PEEK`, `min = PEEK - maxWinTop*rowH`.
  - `winTopFromScrollPx(scrollPx, rowH, PEEK, maxWinTop) → int` (rounded, clamped).
  - Exports `WIN=3`, `PEEK=24`, `SPINE_W`, `DEAD_W`, `BRANCH_COLORS`.

- [ ] **Step 1: Write the failing tests**

```js
import { describe, it, expect } from "vitest";
import { computeLayout, winTopFromScrollPx, scrollBounds, clamp, BRANCH_COLORS } from "../layout.js";

const leaf = (code, occ = 10) => ({ code, occ, residual: 0.5, residual_norm: 0.5, purity: 1, leaf: true, children: [] });
const node = (code, kids) => ({ code, children: kids, occ: kids.reduce((s, c) => s + c.occ, 0), residual: 0.5, residual_norm: 0.5, purity: 1, leaf: false });

function focusWith(kids) { const r = node(null, kids); r.isRoot = true; return r; }

describe("computeLayout", () => {
  it("gives expanded siblings equal width", () => {
    const f = focusWith([leaf(4), leaf(10), leaf(15)]);
    const p = computeLayout(f, 1000, { collapsedSet: new Set() });
    const lvl1 = p.filter(x => x.depth === 1).map(x => Math.round(x.w));
    expect(new Set(lvl1).size).toBe(1);          // all equal
  });

  it("orders collapsed slivers to the left and shrinks them", () => {
    const a = leaf(4), b = leaf(10), c = leaf(15);
    const f = focusWith([a, b, c]);
    const p = computeLayout(f, 1000, { collapsedSet: new Set([c]) });
    const lvl1 = p.filter(x => x.depth === 1).sort((m, n) => m.x - n.x);
    expect(lvl1[0].node).toBe(c);                // collapsed first
    expect(lvl1[0].spine).toBe(true);
    expect(lvl1[0].w).toBeLessThan(lvl1[1].w);   // sliver is narrow
  });

  it("assigns one branch hue per focus child, inherited by descendants", () => {
    const a = node(4, [leaf(1), leaf(2)]);
    const f = focusWith([a, leaf(10)]);
    const p = computeLayout(f, 1000, { collapsedSet: new Set() });
    const childHues = p.filter(x => x.node === a.children[0] || x.node === a.children[1]).map(x => x.branch);
    expect(new Set(childHues).size).toBe(1);     // siblings under same branch share hue
    expect(BRANCH_COLORS).toContain(childHues[0]);
  });
});

describe("scroll math", () => {
  it("derives winTop from a scroll offset and clamps", () => {
    const rowH = 100, PEEK = 24;
    expect(winTopFromScrollPx(PEEK, rowH, PEEK, 3)).toBe(0);
    expect(winTopFromScrollPx(PEEK - 2 * rowH, rowH, PEEK, 3)).toBe(2);
    expect(winTopFromScrollPx(PEEK - 99 * rowH, rowH, PEEK, 3)).toBe(3);  // clamp to maxWinTop
  });
  it("scrollBounds spans PEEK down to PEEK-maxWinTop*rowH", () => {
    expect(scrollBounds(3, 100, 24)).toEqual({ min: 24 - 300, max: 24 });
  });
});
```

- [ ] **Step 2: Run to verify it fails**

Run: `npx vitest run`
Expected: FAIL ("computeLayout is not a function" / module missing).

- [ ] **Step 3: Implement `layout.js`** (port the prototype's `layout` IIFE into a pure function)

```js
export const WIN = 3;
export const PEEK = 24;
export const SPINE_W = 18;
export const DEAD_W = 12;
export const BRANCH_COLORS = ["#4a9eff","#00d4aa","#ff6b6b","#ffd166","#c792ea","#f78c6c","#7fdbca","#82aaff","#f48fb1","#a5d6a7"];

export const clamp = (v, a, b) => Math.max(a, Math.min(b, v));

const isSliver = (n, collapsed) => (collapsed.has(n) && !n.dead) || n.dead;

export function computeLayout(focus, viewportW, { collapsedSet }) {
  const placed = [];
  (function layout(nodeN, x, w, depth, branch) {
    const spine = collapsedSet.has(nodeN) && nodeN !== focus;
    placed.push({ node: nodeN, x, w, depth, branch, spine });
    if (spine || nodeN.leaf || !nodeN.children || !nodeN.children.length) return;

    let nExpanded = 0, fixed = 0;
    for (const c of nodeN.children) {
      if (c.dead) fixed += DEAD_W;
      else if (collapsedSet.has(c)) fixed += SPINE_W;
      else nExpanded++;
    }
    const GAP = Math.max(2, 18 - depth * 6), SMALL = 2;
    const ordered = nodeN.children.slice().sort(
      (a, b) => ((collapsedSet.has(a) && !a.dead) ? 0 : 1) - ((collapsedSet.has(b) && !b.dead) ? 0 : 1)
    );
    const gapBetween = (a, b) => (isSliver(a, collapsedSet) || isSliver(b, collapsedSet)) ? SMALL : GAP;
    let gaps = 0;
    for (let i = 0; i < ordered.length - 1; i++) gaps += gapBetween(ordered[i], ordered[i + 1]);
    const avail = Math.max(0, w - fixed - gaps);
    const eachW = nExpanded ? avail / nExpanded : 0;
    let cx = x;
    ordered.forEach((c, i) => {
      const cw = c.dead ? DEAD_W : collapsedSet.has(c) ? SPINE_W : eachW;
      const cb = depth === 0 ? BRANCH_COLORS[nodeN.children.indexOf(c) % BRANCH_COLORS.length] : branch;
      layout(c, cx, cw, depth + 1, cb);
      cx += cw + (i < ordered.length - 1 ? gapBetween(c, ordered[i + 1]) : 0);
    });
  })(focus, 0, viewportW, 0, null);
  return placed;
}

export const scrollBounds = (maxWinTop, rowH, peek = PEEK) => ({ max: peek, min: peek - maxWinTop * rowH });
export const winTopFromScrollPx = (scrollPx, rowH, peek, maxWinTop) =>
  clamp(Math.round((peek - scrollPx) / rowH), 0, maxWinTop);
```

- [ ] **Step 4: Run to verify it passes**

Run: `npx vitest run`
Expected: PASS (all describe blocks green).

- [ ] **Step 5: Commit**

```bash
git add src/gyo/web/js/layout.js src/gyo/web/js/__tests__/layout.test.js
git commit -m "feat(web): pure layout engine with unit tests"
```

---

## PR 3 — Static icicle renderer (no images, no window yet)

**Deliverable:** A working static icicle replacing the text tree: fill=residual, branch caps, purity bar, labels/occupancy, fed by `/api/tree`. Renders ALL levels at fixed `y=depth*ROW_H` inside `#scroller` (no sliding yet — the page just shows the top of the tree).

### Task 3.1: API + model modules

**Files:**
- Create: `src/gyo/web/js/api.js`, `src/gyo/web/js/model.js`
- Test: extend `src/gyo/web/js/__tests__/layout.test.js` or add `model.test.js`

**Interfaces:**
- `api.js`: `fetchTree() → Promise<json>`, `fetchNodeItems(prefix) → Promise<{items,occupancy}>` (memoized by prefix string), `thumbUrl(idx) → string`.
- `model.js`: `buildModel(treeJson) → root` (nodes carry `code, prefix, level, occ, residual, residual_norm, purity, leaf, children, parent, isRoot`); `depthOf(node) → int`; `residualColor(t) → "rgb(...)"`; `prefixKey(node) → "root"|"c0,c1"`.

- [ ] **Step 1: Write failing model test**

```js
import { describe, it, expect } from "vitest";
import { buildModel, depthOf, residualColor } from "../model.js";

const tree = {
  num_levels: 2,
  nodes: [
    { prefix: [], level: 0, occupancy: 30, mean_residual: 0.4, residual_norm: 0.5, purity: 1 },
    { prefix: [4], level: 1, occupancy: 18, mean_residual: 0.3, residual_norm: 0.2, purity: 1 },
    { prefix: [9], level: 1, occupancy: 12, mean_residual: 0.6, residual_norm: 0.9, purity: 1 },
    { prefix: [4, 1], level: 2, occupancy: 18, mean_residual: 0.3, residual_norm: 0.2, purity: 1 },
  ],
};

describe("buildModel", () => {
  it("builds a rooted tree with parent links and depth", () => {
    const root = buildModel(tree);
    expect(root.isRoot).toBe(true);
    expect(root.children.length).toBe(2);
    expect(depthOf(root)).toBe(2);
    const c4 = root.children.find(c => c.code === 4);
    expect(c4.children[0].parent).toBe(c4);
  });
  it("residualColor returns green low, red high", () => {
    expect(residualColor(0)).toMatch(/^rgb/);
    expect(residualColor(1)).not.toBe(residualColor(0));
  });
});
```

- [ ] **Step 2: Run to verify it fails** — `npx vitest run` → FAIL (module missing).

- [ ] **Step 3: Implement `model.js`** (port `residualColor`, branch/depth/prefix helpers from the prototype; build the tree from `nodes[].prefix` as in the prototype's `render()` map-building, but as a model not DOM):

```js
export const residualColor = (t) => { /* exact copy from prototypes/mock-vi.html residualColor() */ };
export function buildModel(tree) {
  const byKey = new Map();
  const root = { code: null, prefix: [], level: 0, children: [], isRoot: true };
  byKey.set("", root);
  for (const n of tree.nodes) {
    if (n.prefix.length === 0) { Object.assign(root, { occ: n.occupancy, residual: n.mean_residual, residual_norm: n.residual_norm, purity: n.purity }); continue; }
    byKey.set(n.prefix.join(","), {
      code: n.prefix[n.prefix.length - 1], prefix: n.prefix, level: n.level,
      occ: n.occupancy, residual: n.mean_residual, residual_norm: n.residual_norm,
      purity: n.purity, leaf: n.level === tree.num_levels, children: [],
    });
  }
  for (const n of tree.nodes) {
    if (n.prefix.length === 0) continue;
    const me = byKey.get(n.prefix.join(","));
    const parent = byKey.get(n.prefix.slice(0, -1).join(","));
    me.parent = parent; parent.children.push(me);
  }
  return root;
}
export const depthOf = (n) => (!n.children || n.leaf || !n.children.length) ? 0
  : 1 + Math.max(...n.children.filter(c => !c.dead).map(depthOf));
export const prefixKey = (n) => n.isRoot ? "root" : "c_" + n.prefix.join(",");
```

- [ ] **Step 4: Implement `api.js`**

```js
const cache = new Map();
export async function fetchTree() { return (await fetch("/api/tree?level=99")).json(); }
export async function fetchNodeItems(prefixKey) {
  if (cache.has(prefixKey)) return cache.get(prefixKey);
  const p = fetch(`/api/node/${prefixKey}`).then(r => r.json());
  cache.set(prefixKey, p); return p;
}
export const thumbUrl = (idx) => `/thumb/${idx}`;
```

- [ ] **Step 5: Run to verify model tests pass** — `npx vitest run` → PASS.

- [ ] **Step 6: Commit**

```bash
git add src/gyo/web/js/api.js src/gyo/web/js/model.js src/gyo/web/js/__tests__/
git commit -m "feat(web): api + model modules"
```

### Task 3.2: Renderer + new shell (static icicle)

**Files:**
- Create: `src/gyo/web/js/render.js`, `src/gyo/web/js/main.js`
- Rewrite: `src/gyo/web/index.html`, `src/gyo/web/style.css`
- Modify: `src/gyo/api/server.py` (point `index()` at the new shell; keep `/style.css`)

**Interfaces:**
- `render.js`: `renderPlacements(scroller, placements, ctx)` where `ctx = { mode, ROW_H, maxDepth, residualColor, onFillMosaic }`. Draws each placement: fill color, branch cap, label chip, occupancy chip, purity bar; spines get the vertical-label treatment; node `top = depth*ROW_H`, `height = (spine ? (maxDepth+1-depth) : 1)*ROW_H - 8`. (Mosaic + interactions are added in later PRs via `ctx` hooks left as no-ops here.)

- [ ] **Step 1: Rewrite `index.html`** — port the prototype `<body>` shell: `header` (brand, breadcrumb `#crumbs`, `#winstat`, `#dead`, legend, mode button), `#stage` with `#rail`, `#icicle > #scroller`, `.fade-top`/`.fade-bot`, `#tip`. Replace `<script src="/app.js">` with `<script type="module" src="/js/main.js">`. Remove the old `#panel`/`#tree`/context-menu.

- [ ] **Step 2: Rewrite `style.css`** — port every rule from the prototype `<style>` (header, crumbs, rail, node, spine, branchcap, purity, thumbs, tip, fades, icicle/scroller). Keep the dark theme variables.

- [ ] **Step 3: Implement `render.js`** — port the placement-rendering portion of the prototype `render()` loop (the per-node DOM build: cap, label, occ, purity, spine branch) into `renderPlacements`, reading geometry from `ctx` instead of locals. Leave a `ctx.onFillMosaic(el, node)` hook (no-op in this PR).

- [ ] **Step 4: Implement `main.js` (static)**

```js
import { fetchTree } from "./api.js";
import { buildModel, depthOf, residualColor, prefixKey } from "./model.js";
import { computeLayout, PEEK, WIN } from "./layout.js";
import { renderPlacements } from "./render.js";

const icicle = document.getElementById("icicle");
const scroller = document.getElementById("scroller");
let root, focus, collapsedSet = new Set();

async function boot() {
  root = buildModel(await fetchTree());
  focus = root;
  render();
}
function render() {
  const W = icicle.clientWidth, H = icicle.clientHeight;
  const maxDepth = depthOf(focus);
  const ROW_H = Math.max(70, (H - 2 * PEEK) / WIN);
  scroller.style.height = (maxDepth + 1) * ROW_H + "px";
  const placements = computeLayout(focus, W, { collapsedSet });
  renderPlacements(scroller, placements, { mode: "residual", ROW_H, maxDepth, residualColor, onFillMosaic: () => {} });
}
addEventListener("resize", render);
boot();
```

- [ ] **Step 5: Point the server shell at the module entry** — confirm `index()` returns the rewritten `index.html` (already reads `WEB/index.html`; no change needed if filename is unchanged).

- [ ] **Step 6: Verify in the browser (Playwright)**

Run the app (`uv run gyo serve --data-dir run --port 8001`), navigate, screenshot. Expected: a static icicle of residual-colored, branch-capped boxes with labels — no images, no sliding. Confirm no console errors (favicon 404 is fine).

- [ ] **Step 7: Commit**

```bash
git add src/gyo/web/ src/gyo/api/server.py
git commit -m "feat(web): static residual icicle renderer + new shell"
```

---

## PR 4 — Image mosaic + mode toggle + lazy thumbnails

**Deliverable:** Every visible node fills with a sampled thumbnail mosaic (lazy + cached); `C`/button toggles images↔residual; `mode` persisted.

### Task 4.1: Mosaic fill via lazy node items

**Files:**
- Modify: `src/gyo/web/js/render.js` (implement mosaic), `src/gyo/web/js/main.js` (wire `onFillMosaic`, mode state)
- Create: `src/gyo/web/js/persist.js`
- Test: `src/gyo/web/js/__tests__/sample.test.js` (pure sampling helper)

**Interfaces:**
- `render.js` exports `sampleToSlots(items, slots) → items[]` (deterministic even-stride sub-sample) and uses it in `fillMosaic`.
- `persist.js`: `loadPrefs() → {mode, winTop}`, `savePrefs(partial)`.

- [ ] **Step 1: Write failing test for `sampleToSlots`**

```js
import { describe, it, expect } from "vitest";
import { sampleToSlots } from "../render.js";
describe("sampleToSlots", () => {
  it("returns all when fewer than slots", () => {
    expect(sampleToSlots([1,2,3], 10)).toEqual([1,2,3]);
  });
  it("evenly sub-samples to exactly slots", () => {
    const out = sampleToSlots([0,1,2,3,4,5,6,7,8,9], 5);
    expect(out.length).toBe(5);
    expect(out[0]).toBe(0); expect(out[4]).toBe(8);
  });
});
```

- [ ] **Step 2: Run to verify it fails** — `npx vitest run` → FAIL.

- [ ] **Step 3: Implement `sampleToSlots` + `fillMosaic`** in `render.js` (port the prototype mosaic block: compute `cols/rows` from box pixels, `tile = clamp(11+depth*6,11,34)`, build `.thumbs` of `<img src=thumbUrl(idx)>`; use `sampleToSlots`). `fillMosaic` is called from `onFillMosaic(el, node)` which fetches `fetchNodeItems(prefixKey(node))` then fills.

```js
export const sampleToSlots = (items, slots) => {
  if (items.length <= slots) return items.slice();
  const out = []; const step = items.length / slots;
  for (let s = 0; s < slots; s++) out.push(items[Math.min(items.length - 1, Math.floor(s * step))]);
  return out;
};
```

- [ ] **Step 4: Implement `persist.js`**

```js
const KEY = "gyo-vi-prefs";
export const loadPrefs = () => { try { return JSON.parse(localStorage.getItem(KEY)) || {}; } catch { return {}; } };
export const savePrefs = (p) => localStorage.setItem(KEY, JSON.stringify({ ...loadPrefs(), ...p }));
```

- [ ] **Step 5: Wire mode + mosaic in `main.js`** — add `let mode = loadPrefs().mode || "images";`, the `C` key + button `toggleMode()` (saves via `savePrefs({mode})` and re-renders), and pass a real `onFillMosaic` that runs only when `mode==="images"`.

- [ ] **Step 6: Run JS tests + verify in browser** — `npx vitest run` PASS; Playwright: mosaics of real thumbnails appear; pressing `C` toggles to residual color; reload keeps the last mode.

- [ ] **Step 7: Commit**

```bash
git add src/gyo/web/js/ 
git commit -m "feat(web): lazy thumbnail mosaics + image/residual toggle (persisted)"
```

---

## PR 5 — Interactions: collapse/spine, drill, breadcrumb, tooltip

**Deliverable:** Single click collapses a group into a left-rail spine; click spine expands; double click drills; breadcrumb navigates; hover shows the tooltip.

### Task 5.1: Interaction layer

**Files:**
- Create: `src/gyo/web/js/interactions.js`
- Modify: `src/gyo/web/js/render.js` (attach handlers via `ctx` callbacks), `src/gyo/web/js/main.js` (state: collapsedSet, focus; breadcrumb render)

**Interfaces:**
- `render.js` `ctx` gains: `onCollapse(node)`, `onExpand(node)`, `onDrill(node)`, `onHover(e,node)`, `onLeave()`. Renderer attaches single/double-click (220ms timer disambiguation, exactly as the prototype) and hover.
- `interactions.js` exports `makeClickHandlers({onCollapse,onDrill})` returning `{onClick, onDblClick}` implementing the timer split.
- `main.js` owns `collapsedSet` and `focus`; `onCollapse` adds to set + re-render; `onExpand` deletes + re-render; `onDrill` = `setFocus(node)`.

- [ ] **Step 1: Write failing test for the click splitter** (pure timing logic)

```js
import { describe, it, expect, vi } from "vitest";
import { makeClickHandlers } from "../interactions.js";
describe("makeClickHandlers", () => {
  it("fires collapse on single click after the delay", () => {
    vi.useFakeTimers();
    const onCollapse = vi.fn(), onDrill = vi.fn();
    const h = makeClickHandlers({ onCollapse, onDrill, delay: 220 });
    h.onClick({ stopPropagation() {} }, "N");
    vi.advanceTimersByTime(230);
    expect(onCollapse).toHaveBeenCalledWith("N");
    expect(onDrill).not.toHaveBeenCalled();
  });
  it("double click cancels collapse and drills", () => {
    vi.useFakeTimers();
    const onCollapse = vi.fn(), onDrill = vi.fn();
    const h = makeClickHandlers({ onCollapse, onDrill, delay: 220 });
    h.onClick({ stopPropagation() {} }, "N");
    h.onDblClick({ stopPropagation() {} }, "N");
    vi.advanceTimersByTime(230);
    expect(onDrill).toHaveBeenCalledWith("N");
    expect(onCollapse).not.toHaveBeenCalled();
  });
});
```

- [ ] **Step 2: Run to verify it fails** — `npx vitest run` → FAIL.

- [ ] **Step 3: Implement `interactions.js`**

```js
export function makeClickHandlers({ onCollapse, onDrill, delay = 220 }) {
  let timer = null;
  return {
    onClick(e, node) { e.stopPropagation(); if (timer) clearTimeout(timer); timer = setTimeout(() => { onCollapse(node); timer = null; }, delay); },
    onDblClick(e, node) { e.stopPropagation(); if (timer) { clearTimeout(timer); timer = null; } onDrill(node); },
  };
}
```

- [ ] **Step 4: Wire handlers in `render.js` + `main.js`** — renderer calls `ctx.onClick/onDblClick` on expanded nodes, `ctx.onExpand` on spines, `ctx.onHover/onLeave` everywhere; port the prototype tooltip (`showTip`) into `interactions.js` or `render.js`. `main.js` adds `setFocus(node)` (reset window state added in PR6; here just `focus=node; render()`), breadcrumb render (port `renderCrumbs`).

- [ ] **Step 5: Run JS tests + browser verify** — `npx vitest run` PASS; Playwright: click collapses (spine appears on the left), click spine expands, double-click drills + breadcrumb grows, hover tooltip shows.

- [ ] **Step 6: Commit**

```bash
git add src/gyo/web/js/
git commit -m "feat(web): collapse/spine, drill, breadcrumb, tooltip interactions"
```

---

## PR 6 — Sliding window: rail, peek/fades, free-follow+snap scroll

**Deliverable:** The 3-level window with depth rail, peek/fades, and the required free-follow + snap scroll; keys + rail jump.

### Task 6.1: Window/scroll controller

**Files:**
- Create: `src/gyo/web/js/window.js`
- Modify: `src/gyo/web/js/main.js` (own `scrollPx`/`winTop`; sliding moves transform only — no re-render), `src/gyo/web/index.html`/`style.css` (rail/fade already present from PR3)
- Test: `src/gyo/web/js/__tests__/layout.test.js` (scroll math already covered in PR2; add a snap test if needed)

**Interfaces:**
- `window.js` exports `createWindowController({ scroller, icicle, railEl, getMaxDepth, getRowH, onWinTopChange })` returning `{ setWinTop(nv, animate), slide(d), syncToRender(winTop), renderRail() }`. Internals own `scrollPx` and implement: wheel free-follow (`transition:none`, `scrollPx -= deltaY`, clamp via `scrollBounds`), 130ms debounce snap (`transition .22s`, `translateY(PEEK - winTop*ROW_H)`), arrow keys, rail tick click → `setWinTop`. Uses `winTopFromScrollPx` from `layout.js`.

- [ ] **Step 1: Add a snap-on-stop unit test** (the controller's pure step) — extract the snap target computation as `snapTarget(scrollPx, rowH, PEEK, maxWinTop) → {winTop, scrollPx}` in `layout.js` and test:

```js
import { snapTarget } from "../layout.js";
it("snaps to nearest level", () => {
  expect(snapTarget(24 - 130, 100, 24, 3)).toEqual({ winTop: 1, scrollPx: 24 - 100 });
});
```

- [ ] **Step 2: Run to verify it fails** — `npx vitest run` → FAIL.

- [ ] **Step 3: Add `snapTarget` to `layout.js`**

```js
export function snapTarget(scrollPx, rowH, peek, maxWinTop) {
  const winTop = winTopFromScrollPx(scrollPx, rowH, peek, maxWinTop);
  return { winTop, scrollPx: peek - winTop * rowH };
}
```

- [ ] **Step 4: Implement `window.js`** — port the prototype's scroll model: `setScrollPx(px, animate)`, `winTopFromScroll`, `setWinTop`, `slide`, the `wheel` free-follow + `snapTimer` (130ms) handler, `↑/↓` keys, `renderRail` (ticks `0..maxDepth`, highlight `winTop..winTop+WIN-1`, click→`setWinTop`), and the live `#winstat` update. On `onWinTopChange` it does NOT re-render the icicle — only the wrapper transform + rail.

- [ ] **Step 5: Wire into `main.js`** — instantiate the controller; `render()` sets `ROW_H`, positions via `setScrollPx(PEEK - winTop*ROW_H, false)`; `setFocus` resets `winTop` (clamp 1). Ensure sliding does not call `render()`.

- [ ] **Step 6: Run JS tests + browser verify** — `npx vitest run` PASS; Playwright: scroll free-follows (transform changes with `transition:none` during; snaps with `transition .22s` after ~130ms); rail highlight + `#winstat` track live; `↑/↓` and rail clicks jump; peek + fades visible.

- [ ] **Step 7: Commit**

```bash
git add src/gyo/web/js/ src/gyo/web/
git commit -m "feat(web): sliding 3-level window with rail and free-follow+snap scroll"
```

---

## PR 7 — Responsiveness, persistence, dead-count, cleanup

**Deliverable:** Resize reflow; window position persisted; `dead [...]` header metric; remove dead `app.js`; final polish.

### Task 7.1: Responsive reflow + window persistence + dead metric

**Files:**
- Modify: `src/gyo/web/js/main.js`, `src/gyo/web/js/window.js`, `src/gyo/web/js/persist.js`
- Delete: `src/gyo/web/app.js` (old monolith, now unused)

**Interfaces:**
- `main.js`: debounced `resize` (≈120ms) recomputes `ROW_H`/widths via `render()` and re-clamps `scrollPx` from the preserved `winTop`.
- Window persistence: `setWinTop` calls `savePrefs({winTop})`; boot restores `winTop` (clamped) from `loadPrefs()`.
- Dead metric: `main.js` writes `tree.dead_codewords` into `#dead` on load.

- [ ] **Step 1: Debounced resize in `main.js`**

```js
let rT; addEventListener("resize", () => { clearTimeout(rT); rT = setTimeout(render, 120); });
```

- [ ] **Step 2: Persist + restore `winTop`** — in the controller's `setWinTop`, after clamping, `savePrefs({ winTop })`; in `boot()`, set initial `winTop = clamp(loadPrefs().winTop ?? 1, 0, maxWinTop)`.

- [ ] **Step 3: Dead-count header** — on load: `document.getElementById("dead").textContent = JSON.stringify(tree.dead_codewords);`

- [ ] **Step 4: Delete the old monolith and its routes** — remove `src/gyo/web/app.js`; in `server.py` remove the now-unused `/app.js` route (keep `/style.css`, `/js/{name}`, `/`).

- [ ] **Step 5: Verify in the browser (Playwright) — responsiveness**

Navigate; `browser_resize` to e.g. 900×600 then 1600×1000; screenshot each. Expected: 3 levels always fill the height, widths reflow, mosaic density adapts, `winTop` preserved. Reload: mode + window restored. No console errors.

- [ ] **Step 6: Full test sweep**

Run: `uv run pytest -q && npx vitest run`
Expected: all green.

- [ ] **Step 7: Commit**

```bash
git add -A src/gyo/web src/gyo/api/server.py
git commit -m "feat(web): responsive reflow, window persistence, dead metric; drop legacy app.js"
```

---

## Self-Review (filled in)

**Spec coverage:** §3 encodings → PR3 (fill/cap/purity/label) + PR4 (mosaic/toggle). §3.2 dive/sampling → PR4. §4 API contracts → PR1 (route+contract) + PR3/4 (api.js). §5 layout (equal width, gutters, slivers-left, spine, branch) → PR2 engine + PR3 render. §5.6 dead → PR7 metric (not drawn anywhere). §6 window/peek/rail → PR6. §6.3 free-follow+snap → PR6. §7 interactions → PR5. §8 state → PR3/5/6. §8.1 persistence → PR4 (mode) + PR7 (winTop). §9 modules → file structure. §11 testing → vitest tasks throughout + Playwright steps. §12 responsiveness → PR7. §13 decisions → reflected (dead kept, test runner, persistence, prefetch budget in api.js cache + PR4 fetch-only-visible).

**Placeholder scan:** mosaic/render/style tasks reference `prototypes/mock-vi.html` as the authoritative source to port (the code exists and is committed) rather than re-inlining ~500 lines; all new/pure logic (layout, sampling, click-split, scroll math, persistence, model, api) has complete code.

**Type consistency:** `computeLayout`, `winTopFromScrollPx`, `scrollBounds`, `snapTarget`, `sampleToSlots`, `makeClickHandlers`, `buildModel`, `depthOf`, `residualColor`, `prefixKey`, `fetchTree`, `fetchNodeItems`, `thumbUrl`, `loadPrefs`, `savePrefs` are defined once and referenced consistently. `ctx` keys (`mode, ROW_H, maxDepth, residualColor, onFillMosaic, onClick, onDblClick, onExpand, onHover, onLeave`) are introduced in PR3 and extended in PR4/PR5.

**Note for executor:** the prototype `prototypes/mock-vi.html` is the canonical reference for any CSS rule or DOM-building detail not fully inlined here. Port it; swap fake data for the API modules.
