# gyo vi — Interface Redesign Design Spec

**Status:** Draft for review
**Date:** 2026-06-23
**Reference prototype:** `mock-vi.html` (static, fake-data prototype validated interactively; the prototype is a throwaway visual reference, NOT the implementation target)

## 1. Overview

`gyo vi` is a redesign of the embedding-space inspector UI. It replaces the
current collapsible text-tree (`src/gyo/web/`) with a **collapse-based,
image-filled icicle** over the residual-quantization (RQ) tree, with a
**3-level sliding window** for depth navigation.

The single view must serve four jobs at once (all confirmed as in-scope):

1. **Explore the dataset visually** — see the actual images grouped by similarity.
2. **Understand the hierarchy** — read the RQ tree structure at a glance.
3. **Diagnose the RQ** — spot high-residual / impure clusters.
4. **Inspect / compare clusters** — drill into specific nodes.

## 2. Goals / Non-goals

**Goals**
- One hero visualization that encodes hierarchy, occupancy context, residual, branch identity, and the underlying images simultaneously.
- Focused analysis via collapsing irrelevant branches out of the way.
- Ergonomic depth navigation over trees deeper than one screen.
- **Responsive**: adapts fluidly to any viewport size and to live resizing (see §12).
- Reuse the existing backend API unchanged where possible.

**Non-goals**
- No 2D embedding map (UMAP/t-SNE). Rejected during design (loses tree topology, adds dependency/compute).
- No dead-codeword nodes in the visualization (the aggregate dead count is kept as a header metric — see §5.6).
- No re-training / data-pipeline changes.

## 3. Hero visualization: collapse-based icicle

A vertical icicle: depth increases downward, each level is a horizontal row of
boxes; a node's box sits horizontally within its parent's span (containment
preserved). The view always shows a **window of 3 full levels** (see §6).

### 3.1 Visual encodings (channels)

| Channel | Encodes | Notes |
|---|---|---|
| **Box width** | *equal among expanded siblings* | NOT proportional to occupancy — equal width was explicitly chosen for legibility. Each parent divides its span equally among its expanded children. |
| **Fill color** | residual (`residual_norm`) | Continuous green→yellow→red gradient. Applies to every node incl. aggregates; aggregate nodes carry the mean residual of all items beneath them (matches `mean_residual` semantics). |
| **Image mosaic** | the node's contents | A sampled grid of thumbnails fills every node (see §3.2). Replaces fill color in "images" mode; fill color shows through gaps/borders. |
| **Top cap stripe** | branch identity | A ~4px colored bar at the top of each node, colored by the node's branch (immediate subtree of the *current focus*), shared by the whole subtree (see §5.4). |
| **Bottom bar** | purity | A thin bar at the node's base, width = purity fraction. |
| **Corridor (gutter)** | branch separation | Empty vertical channels between branches; wider at shallow splits (see §5.4). |

### 3.2 Image mosaic ("dive into images")

- Every node — at every level, including the root/aggregate nodes — renders a
  **sampled mosaic** of the images it contains, so no band is ever empty.
- Tiles grow with depth (deeper node ⇒ fewer, larger tiles) producing a
  continuous "dive" from overview to individual images as you go deeper / drill.
- The number of tiles is bounded by the box area (cols×rows of fitting tiles);
  the node's item set is **evenly sub-sampled** to fill exactly those slots.
- Mode toggle (`C` key / header button) switches between **images** (mosaic)
  and **residual** (pure color bands) globally. Default: images.

## 4. Data model & API contracts

Reuses the existing endpoints. No backend schema change is required for the
core; one clarification on lazy thumbnail loading below.

### 4.1 `GET /api/tree?level=<n>`
Provides the tree structure + per-node stats. Fields consumed per node:
`prefix`, `level`, `occupancy`, `mean_residual`, `residual_norm`, `purity`.
`num_levels` = tree depth used to bound the depth rail / window. (`dead_codewords`
optional, see §5.6.) The front end builds the node tree from `prefix` arrays.

### 4.2 `GET /api/node/{prefix}`
Returns `items` (`idx`, `path`, `label`) for the node at `prefix` — this works
for **internal nodes too** (item_indices includes all descendants), which is how
mosaics for aggregate nodes are populated. `prefix` = `root` or `c0,c1,...`.

### 4.3 `GET /thumb/{idx}`
Returns the image for item `idx`. (Content-type + `Cache-Control: no-cache`
already fixed.)

### 4.4 Lazy thumbnail loading + caching (front-end behavior)
- The mosaic for a node is filled by fetching `/api/node/{prefix}` **on demand**
  (only for nodes currently inside the window) and caching the result by prefix.
- Tiles render as `<img src="/thumb/{idx}">` for a sampled subset of the node's
  items. Off-screen / collapsed nodes are not fetched.
- Sampling is deterministic (even stride over the item list) so a node's mosaic
  is stable across re-renders.

**Prefetch budget (sane defaults; tune later):**
- Fetch items for nodes in the **current window + the one peek level above and
  below** only. Do not prefetch the whole tree.
- Per node, request at most a bounded sample (e.g. cap the `/api/node` items to a
  few hundred) and sub-sample to the box's visible tile slots.
- Cache fetched node-item lists for the session (unbounded for the session is
  fine at expected dataset sizes; revisit if memory becomes a concern).
- Browser image cache + `Cache-Control` handle `/thumb` reuse; no manual image cache.

## 5. Layout algorithm

Input: the focused node and its subtree. Produces `{node, x, width, depth, branch, spine}` placements.

### 5.1 Equal-width partition
For a node of width `w` at `depth`:
- Slivers (collapsed spines, see §5.3) take a fixed width `SPINE_W`.
- Remaining width is split **equally** among the expanded children.
- (Occupancy is shown as a number chip, not as width.)

### 5.2 Gutters (branch corridors)
- A gutter is inserted between adjacent expanded siblings; its size **decreases
  with depth** (`max(2, 18 − depth*6)` px in the prototype) so the shallowest
  splits read as full-height corridors between branches, narrowing for sub-branches.
- Because layout is nested, a top-level gutter produces an empty corridor running
  the full height between branches.
- Between a sliver and anything (sliver↔sliver, sliver↔expanded), the gutter is
  **minimal** (`SMALL_GAP` ≈ 2px) so minimized items hug together and reclaim space.

### 5.3 Collapse → spine
- **Single click** on an expanded node collapses it (and its whole subtree) into
  a **spine**: a thin (`SPINE_W` ≈ 18px) vertical bar that spans the full window
  height. The spine keeps: residual fill color, branch cap, a vertical label
  (`cN`), the occupancy chip, and a `⊞` expand affordance.
- Clicking a spine expands it back.
- **Collapsed siblings slide to the left** of their parent's span (a left "rail"
  of minimized branches), in original sibling order; expanded siblings keep their
  order on the right. This keeps the analysis area contiguous on the right.
- Collapse state is per-node and persists across drilling and window sliding.

### 5.4 Branch identity
- "Branch" = an immediate subtree of the **current focus** (relative, so drilling
  re-assigns branches).
- Each branch gets a distinct hue from a fixed palette; descendants inherit it.
- Rendered as the top cap stripe; combined with corridors, branches are readable
  by both separation (corridor) and identity (color).

### 5.5 Ordering rule
Within each parent: collapsed slivers first (left), then expanded children in
original order. Stable sort.

### 5.6 Dead codewords
- Dead-codeword nodes are **not drawn** in the icicle (no slivers cluttering the tree).
- **Decision:** the aggregate `dead [...]` count **is kept as a header metric**
  (per-level dead counts from `/api/tree`'s `dead_codewords`). It is a diagnostic
  summary only — it occupies no space in the tree itself.

## 6. Sliding window (depth navigation)

The icicle shows **3 full levels at a time** (`WIN = 3`) and slides vertically
over the tree depth. Vertical = depth (window); horizontal = which subtree
(focus/collapse). They compose: drill into a branch to widen it, then slide the
window down within it.

### 6.1 Window & peek
- All rendered levels live in a single transformed wrapper; the viewport clips it
  (`overflow: hidden`).
- At rest the window shows 3 full levels between a top/bottom **peek** band: the
  level just above and just below are partially visible (clipped), hinting more
  content. Soft **fade** overlays at the top/bottom edges reinforce the cutoff.
- Row height = `(viewportH − 2·PEEK) / 3`; a level `d` sits at `y = d · ROW_H`
  in the wrapper.

### 6.2 Depth rail (orientation)
- A thin vertical rail on the left lists **all levels under the current focus**
  (`0…maxDepth`, where `maxDepth` = remaining depth from the focus) as ticks; the
  3 ticks inside the window are highlighted. Drilling rebuilds the rail.
- Click a tick → jump so that level becomes the window top (animated snap).
- The header shows `janela X–Y / Z` live.

### 6.3 Scroll behavior — free-follow + snap-on-stop (REQUIRED)
This was iterated on explicitly; the required behavior is:
- While scrolling (wheel/trackpad), the wrapper **follows the input continuously
  and fractionally**, with **no animation/transition** — it tracks 1:1 so it feels
  fluid and is easy to control.
- The window position is a continuous pixel offset clamped to
  `[PEEK − maxWinTop·ROW_H, PEEK]`.
- The rail highlight and `janela X–Y` update **live** during the scroll (derived
  from the rounded current offset).
- When scrolling **stops** (~130 ms debounce after the last event), it **snaps**
  to the nearest level with a short animated transition (~0.22 s).
- **Not acceptable:** per-notch discrete jumps, or hyper-sensitive multi-level
  jumps per gesture. (Both were rejected during design.)

### 6.4 Other depth controls
- `↑` / `↓` keys: snap one level (animated).
- Rail tick click / peek-region: jump/slide (animated).

## 7. Interaction summary

| Gesture | Action |
|---|---|
| Single click on node | Collapse group → spine (slides to left rail) |
| Click on spine (`⊞`) | Expand |
| Double click on node | Drill (re-root focus on it); breadcrumb grows |
| Breadcrumb crumb click | Re-root to that ancestor |
| Scroll / trackpad | Free-follow depth, snap on stop (§6.3) |
| `↑`/`↓` | Slide window one level (snap) |
| Rail tick click | Jump window to that level |
| Hover | Tooltip: occupancy, residual, purity, label |
| `C` / header toggle | Images mosaic ↔ residual color |

Single vs double click are disambiguated with a short timer (≈220 ms), matching
the existing pattern in the codebase.

## 8. State model (front-end)

- `focus`: current root node (drill changes it; reset window on change).
- `collapsed`: per-node boolean (spine state).
- `scrollPx` / derived `winTop`: continuous window offset + its snapped level.
- `mode`: `"images" | "residual"`.
- `maxDepth`: depth under focus (= remaining `num_levels`), recomputed on focus change.
- Caches: node-items by prefix (for mosaics).

On **drill** (`setFocus`): recompute `maxDepth`, reset window to a sensible top
(level 1 by default, clamped), full re-render.

### 8.1 Persistence (localStorage)
- **`mode`** (images/residual) and **window position** (`winTop`) **persist across
  reloads** via `localStorage`, restored on load (clamped to the current tree).
- `focus` and `collapsed` state are **not** persisted (a reload returns to the
  root view) — keeps reloads predictable; revisit if requested.

## 9. Component / module breakdown (front-end)

Keep units small and independently testable:

1. **Tree model** — build node tree from `/api/tree`; parent links, prefixes,
   per-focus branch assignment, depth.
2. **Layout engine** — pure function: `(focusSubtree, viewportW, collapsedSet) → placements[]`
   implementing §5 (equal widths, gutters, slivers-left, spines). No DOM.
3. **Renderer** — placements → DOM nodes (fill/mosaic/cap/purity/label/occ) into
   the scroll wrapper at fixed `y = depth·ROW_H`.
4. **Window/scroll controller** — owns `scrollPx`/`winTop`, the wheel free-follow
   + snap, keys, peek/fades, and the wrapper transform (§6). Sliding does NOT
   rebuild the DOM (transform only) — only focus/collapse/mode changes re-render.
5. **Depth rail** — renders ticks, highlight, jump-on-click.
6. **Interaction layer** — click/dblclick/hover/breadcrumb wiring.
7. **Thumbnail loader** — lazy fetch + cache of `/api/node` items; sampling.

The layout engine being a pure, DOM-free function is the key testability boundary.

## 10. Edge cases

- **Slide must not rebuild** the DOM (perf + smoothness): only the wrapper
  transform moves; re-render happens on focus/collapse/mode change.
- **Deep + narrow:** sliding deep without drilling yields many narrow boxes; this
  is expected — drilling widens. Document in UI hint.
- **Drilling into a leaf:** no-op (no children to show).
- **Collapse + window:** a branch collapsed above the window is represented by its
  spine spanning the window; its subtree contributes nothing else.
- **Empty / single-child levels:** equal-width split still applies (one child = full width).
- **Window clamping:** `winTop ∈ [0, max(0, num_levels − 3)]`.

## 11. Testing strategy (Definition of Done)

- **Backend (pytest):** existing pipeline/tree tests stay green; add a contract
  test for `GET /api/node/{prefix}` returning correct items for both leaf and
  internal prefixes (the mosaic feature depends on it).
- **Layout engine (unit) — REQUIRED:** a JS test runner **will be introduced**
  (e.g. Vitest/Node test). The layout engine is extracted as a pure, DOM-free
  module and unit-tested: equal widths among expanded siblings, sliver-left
  ordering, depth-scaled gutter sizing, minimal sliver gaps, spine spans,
  window→`winTop` derivation, and clamping. Choice of runner is an implementation
  detail for the plan.
- **Front-end (behavioral, Playwright):** load; verify 3-level window + peeks;
  scroll free-follow then snap-to-level; collapse → spine slides left; drill
  updates breadcrumb; mosaics appear; rail highlight tracks window; **resize the
  viewport and confirm the layout reflows** (§12). Capture screenshots as evidence.

## 12. Responsiveness (required)

The interface must adapt fluidly to viewport size and live resizing.

- **Reflow on resize:** `viewportW` and `viewportH` are re-read and the layout +
  window geometry recomputed on every container resize (debounced). Nothing is
  hard-coded to a fixed canvas size.
- **Vertical:** `ROW_H = (viewportH − 2·PEEK) / WIN` — the 3 levels always fill
  the available height; `PEEK` and the fade bands scale sensibly.
- **Horizontal:** all box widths derive from `viewportW` (equal-width partition,
  gutters, slivers); they recompute on resize.
- **Mosaic density adapts:** tile counts per node are a function of the box's
  current pixel area, so resizing changes how many thumbnails show — no fixed grid.
- **Small viewports:** the layout must remain usable when narrow/short — minimum
  legible box widths, the depth rail and header stay accessible (exact small-screen
  behavior, e.g. collapsing the rail, is an implementation detail to validate).
- **Scroll offset stays valid:** on resize, `winTop` is preserved and `scrollPx`
  recomputed from it (re-clamped) so the same depth window stays in view.
- Verified behaviorally (Playwright resize, §11).

## 13. Resolved decisions

1. **Dead count:** dead nodes are not drawn in the icicle, but the aggregate
   `dead [...]` count **is kept** as a header diagnostic metric (§5.6).
2. **Layout-engine tests:** a JS test runner is introduced; the pure layout engine
   is unit-tested (§11).
3. **Persistence:** `mode` and window position persist across reloads via
   `localStorage`; `focus`/`collapsed` reset (§8.1).
4. **Thumbnail prefetch:** sane defaults — fetch only the window ± one peek level,
   bounded per-node sample, session cache (§4.4).
5. **Responsiveness:** required, viewport-adaptive with reflow on resize (§12).
