# Semantic Atlas UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the icicle inspector with a hierarchical semantic atlas that positions sibling semantic IDs using deterministic metric MDS and exposes large representative, outlier, and parent-comparison samples.

**Architecture:** Keep numerical work server-side in focused geometry and sampling modules. Add one atlas API that returns the current focus, its immediate children, normalized MDS geometry, quality metrics, and ranked sample IDs. Replace the current DOM renderer with a responsive SVG map plus a persistent HTML inspector; keep geometry normalization and UI state in pure JavaScript modules.

**Tech Stack:** Python 3.12, NumPy, SciPy, FastAPI, vanilla ES modules, SVG, CSS, pytest, Vitest, Playwright.

---

## File structure

- Create `src/gyo/atlas/geometry.py`: codeword reconstruction, pairwise distances, deterministic SMACOF, normalized positions, stress.
- Create `src/gyo/atlas/sampling.py`: representative and outlier ranking from original embeddings.
- Create `src/gyo/atlas/__init__.py`: public atlas-domain exports.
- Create `tests/test_atlas_geometry.py`: numerical contracts.
- Create `tests/test_atlas_sampling.py`: ranking contracts.
- Modify `src/gyo/api/server.py`: cached run loading and `GET /api/atlas/{prefix}`.
- Modify `tests/test_server.py`: atlas endpoint and failure contracts.
- Replace `src/gyo/web/index.html`: semantic atlas shell.
- Replace `src/gyo/web/style.css`: scientific-instrument visual system and responsive layout.
- Create `src/gyo/web/js/atlas-model.js`: response normalization and pure state transitions.
- Create `src/gyo/web/js/atlas-layout.js`: viewport fitting and territory sizing.
- Create `src/gyo/web/js/atlas-render.js`: SVG territories, paths, thumbnail samples, and inspector DOM.
- Replace `src/gyo/web/js/main.js`: loading, navigation, selection, zoom, retry, and keyboard orchestration.
- Create `src/gyo/web/js/__tests__/atlas-model.test.js`: state contracts.
- Create `src/gyo/web/js/__tests__/atlas-layout.test.js`: responsive layout contracts.
- Replace `tests/test_e2e_vi_flows.py`: user-facing atlas flows.

### Task 1: Deterministic metric MDS geometry

**Files:**
- Create: `src/gyo/atlas/__init__.py`
- Create: `src/gyo/atlas/geometry.py`
- Create: `tests/test_atlas_geometry.py`

- [ ] **Step 1: Write failing reconstruction and distance tests**

```python
# tests/test_atlas_geometry.py
import numpy as np
from gyo.atlas.geometry import prefix_vector, sibling_distance_matrix


CODEBOOKS = [
    np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    np.array([[0.5, 0.0], [0.0, 0.5]], dtype=np.float32),
]


def test_prefix_vector_sums_path_codewords():
    np.testing.assert_allclose(prefix_vector((0, 1), CODEBOOKS), [1.0, 0.5])


def test_sibling_distance_matrix_uses_reconstructed_vectors():
    distances = sibling_distance_matrix([(0, 0), (0, 1)], CODEBOOKS)
    np.testing.assert_allclose(distances, [[0.0, np.sqrt(0.5)], [np.sqrt(0.5), 0.0]])
```

- [ ] **Step 2: Run the tests and verify the import fails**

Run: `uv run pytest tests/test_atlas_geometry.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'gyo.atlas'`.

- [ ] **Step 3: Implement reconstruction and Euclidean distances**

```python
# src/gyo/atlas/geometry.py
from __future__ import annotations
import numpy as np


def prefix_vector(prefix: tuple[int, ...], codebooks: list[np.ndarray]) -> np.ndarray:
    if not codebooks:
        raise ValueError("at least one codebook is required")
    out = np.zeros(codebooks[0].shape[1], dtype=np.float64)
    for level, code in enumerate(prefix):
        out += np.asarray(codebooks[level][code], dtype=np.float64)
    return out


def sibling_distance_matrix(prefixes, codebooks) -> np.ndarray:
    vectors = np.stack([prefix_vector(tuple(p), codebooks) for p in prefixes])
    delta = vectors[:, None, :] - vectors[None, :, :]
    return np.linalg.norm(delta, axis=-1)
```

```python
# src/gyo/atlas/__init__.py
from .geometry import prefix_vector, sibling_distance_matrix

__all__ = ["prefix_vector", "sibling_distance_matrix"]
```

- [ ] **Step 4: Run the focused tests**

Run: `uv run pytest tests/test_atlas_geometry.py -v`

Expected: 2 passed.

- [ ] **Step 5: Add failing deterministic MDS tests**

```python
# append to tests/test_atlas_geometry.py
from gyo.atlas.geometry import metric_mds


def test_metric_mds_is_deterministic_and_normalized():
    d = np.array([[0, 1, 1], [1, 0, np.sqrt(2)], [1, np.sqrt(2), 0]], dtype=float)
    first = metric_mds(d)
    second = metric_mds(d)
    np.testing.assert_allclose(first.positions, second.positions)
    assert np.max(np.linalg.norm(first.positions, axis=1)) <= 1.0 + 1e-9
    assert first.stress < 0.01


def test_metric_mds_handles_zero_and_single_point():
    assert metric_mds(np.zeros((0, 0))).positions.shape == (0, 2)
    np.testing.assert_allclose(metric_mds(np.zeros((1, 1))).positions, [[0.0, 0.0]])
```

- [ ] **Step 6: Verify the new tests fail**

Run: `uv run pytest tests/test_atlas_geometry.py -v`

Expected: FAIL because `metric_mds` is not defined.

- [ ] **Step 7: Implement classical initialization plus deterministic SMACOF**

```python
# append to src/gyo/atlas/geometry.py
from dataclasses import dataclass


@dataclass(frozen=True)
class MDSResult:
    positions: np.ndarray
    stress: float


def _classical_init(distances: np.ndarray) -> np.ndarray:
    n = len(distances)
    centering = np.eye(n) - np.ones((n, n)) / n
    gram = -0.5 * centering @ (distances ** 2) @ centering
    values, vectors = np.linalg.eigh(gram)
    order = np.argsort(values)[::-1][:2]
    positive = np.maximum(values[order], 0.0)
    coords = vectors[:, order] * np.sqrt(positive)
    if coords.shape[1] < 2:
        coords = np.pad(coords, ((0, 0), (0, 2 - coords.shape[1])))
    return coords


def _normalized_stress(target: np.ndarray, actual: np.ndarray) -> float:
    mask = np.triu(np.ones_like(target, dtype=bool), 1)
    numerator = np.sum((target[mask] - actual[mask]) ** 2)
    denominator = np.sum(actual[mask] ** 2)
    return float(np.sqrt(numerator / denominator)) if denominator else 0.0


def metric_mds(distances: np.ndarray, max_iter: int = 300, eps: float = 1e-7) -> MDSResult:
    distances = np.asarray(distances, dtype=np.float64)
    if distances.shape != (len(distances), len(distances)):
        raise ValueError("distances must be square")
    n = len(distances)
    if n == 0:
        return MDSResult(np.empty((0, 2)), 0.0)
    if n == 1:
        return MDSResult(np.zeros((1, 2)), 0.0)
    x = _classical_init(distances)
    previous = np.inf
    for _ in range(max_iter):
        actual = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=-1)
        safe = np.where(actual > 1e-12, actual, 1e-12)
        ratio = np.divide(distances, safe, out=np.zeros_like(distances), where=safe > 0)
        b = -ratio
        np.fill_diagonal(b, 0.0)
        np.fill_diagonal(b, -b.sum(axis=1))
        updated = (b @ x) / n
        updated -= updated.mean(axis=0, keepdims=True)
        delta = np.linalg.norm(updated - x)
        x = updated
        if abs(previous - delta) < eps:
            break
        previous = delta
    radius = np.max(np.linalg.norm(x, axis=1))
    if radius > 0:
        x = x / radius
    actual = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=-1)
    scale = np.sum(distances * actual) / max(np.sum(actual ** 2), 1e-12)
    return MDSResult(x, _normalized_stress(distances, actual * scale))
```

Export `metric_mds` and `MDSResult` from `src/gyo/atlas/__init__.py`.

- [ ] **Step 8: Run geometry and full Python tests**

Run: `uv run pytest tests/test_atlas_geometry.py -v && uv run pytest -q`

Expected: geometry tests pass and the existing suite remains green.

- [ ] **Step 9: Commit the geometry unit**

```bash
git add src/gyo/atlas tests/test_atlas_geometry.py
git commit -m "feat: add deterministic semantic atlas geometry"
```

### Task 2: Representative and outlier sampling

**Files:**
- Create: `src/gyo/atlas/sampling.py`
- Create: `tests/test_atlas_sampling.py`
- Modify: `src/gyo/atlas/__init__.py`

- [ ] **Step 1: Write failing ranking tests**

```python
# tests/test_atlas_sampling.py
import numpy as np
from gyo.atlas.sampling import ranked_samples


def test_ranked_samples_returns_nearest_and_farthest_members():
    embeddings = np.array([[0.0, 0.0], [0.1, 0.0], [2.0, 0.0], [9.0, 9.0]])
    result = ranked_samples(embeddings, [0, 1, 2], np.array([0.0, 0.0]), limit=2)
    assert result.representative == [0, 1]
    assert result.outliers == [2, 1]


def test_ranked_samples_never_duplicates_or_leaks_nonmembers():
    result = ranked_samples(np.eye(3), [2], np.array([0.0, 0.0, 0.0]), limit=8)
    assert result.representative == [2]
    assert result.outliers == [2]
```

- [ ] **Step 2: Run and verify failure**

Run: `uv run pytest tests/test_atlas_sampling.py -v`

Expected: FAIL because `gyo.atlas.sampling` does not exist.

- [ ] **Step 3: Implement Euclidean sample ranking**

```python
# src/gyo/atlas/sampling.py
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class RankedSamples:
    representative: list[int]
    outliers: list[int]


def ranked_samples(embeddings, member_indices, center, limit=24) -> RankedSamples:
    members = np.asarray(member_indices, dtype=np.int64)
    if len(members) == 0:
        return RankedSamples([], [])
    distances = np.linalg.norm(np.asarray(embeddings)[members] - center, axis=1)
    order = np.argsort(distances, kind="stable")
    count = min(int(limit), len(members))
    return RankedSamples(
        representative=members[order[:count]].astype(int).tolist(),
        outliers=members[order[::-1][:count]].astype(int).tolist(),
    )
```

Export `ranked_samples` and `RankedSamples` from `src/gyo/atlas/__init__.py`.

- [ ] **Step 4: Run tests and commit**

Run: `uv run pytest tests/test_atlas_sampling.py -v && uv run pytest -q`

Expected: all tests pass.

```bash
git add src/gyo/atlas tests/test_atlas_sampling.py
git commit -m "feat: rank atlas representative and outlier samples"
```

### Task 3: Atlas API contract

**Files:**
- Modify: `src/gyo/api/server.py`
- Modify: `tests/test_server.py`

- [ ] **Step 1: Extend the seeded run with embeddings and codebooks**

Add to `_seed_run` in `tests/test_server.py`:

```python
    np.save(data_dir / "embeddings.npy", np.array([[1.0, 0.0], [1.4, 0.0], [0.0, 1.0]], dtype=np.float32))
    cb = data_dir / "codebooks" / "v1"
    cb.mkdir(parents=True)
    np.save(cb / "level_0.npy", np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
    np.save(cb / "level_1.npy", np.array([[0.0, 0.0], [0.4, 0.0]], dtype=np.float32))
    (cb / "config.json").write_text('{"num_levels":2,"codebook_size":2,"dim":2,"proj_dim":null,"seed":0}')
```

- [ ] **Step 2: Write failing atlas endpoint tests**

```python
def test_atlas_endpoint_returns_geometry_samples_and_metrics(tmp_path):
    _seed_run(tmp_path)
    client = TestClient(create_app(str(tmp_path)))
    response = client.get("/api/atlas/root")
    assert response.status_code == 200
    body = response.json()
    assert body["focus"]["prefix"] == []
    assert set(body["focus"]["samples"]) == {"representative", "outliers"}
    assert {tuple(node["prefix"]) for node in body["children"]} == {(0,), (1,)}
    assert all(len(node["position"]) == 2 for node in body["children"])
    assert body["projection"]["metric"] == "euclidean"
    assert body["projection"]["method"] == "metric-mds"
    assert set(body["children"][0]["samples"]) == {"representative", "outliers"}


def test_atlas_endpoint_rejects_missing_geometry_inputs(tmp_path):
    _seed_run(tmp_path)
    (tmp_path / "embeddings.npy").unlink()
    response = TestClient(create_app(str(tmp_path))).get("/api/atlas/root")
    assert response.status_code == 409
    assert "embeddings.npy" in response.json()["detail"]
```

- [ ] **Step 3: Run and verify the endpoint is missing**

Run: `uv run pytest tests/test_server.py -v`

Expected: atlas requests return 404.

- [ ] **Step 4: Add cached run loading and item serialization**

In `create_app`, add a run cache that loads `codes.parquet`, `meta.parquet`, `embeddings.npy`, and `ResidualQuantizer.load(data_dir / "codebooks" / "v1")` once. Introduce these helpers inside `create_app` so app instances remain isolated in tests:

```python
    atlas_cache = None

    def _load_atlas_inputs():
        nonlocal atlas_cache
        required = [data_dir / "embeddings.npy", data_dir / "codebooks" / "v1" / "config.json"]
        missing = [p.name for p in required if not p.exists()]
        if missing:
            raise HTTPException(409, f"atlas inputs missing: {', '.join(missing)}")
        if atlas_cache is None:
            codes, final_res, labels, meta_df = _load()
            atlas_cache = (
                codes, final_res, labels, meta_df,
                np.load(data_dir / "embeddings.npy"),
                ResidualQuantizer.load(data_dir / "codebooks" / "v1"),
            )
        return atlas_cache

    def _item(meta_df, idx):
        return {"idx": int(idx), "path": str(meta_df.loc[idx, "path"]), "label": str(meta_df.loc[idx, "label"])}
```

Add imports for `ResidualQuantizer`, `prefix_vector`, `sibling_distance_matrix`, `metric_mds`, and `ranked_samples`.

- [ ] **Step 5: Implement `GET /api/atlas/{prefix}`**

The handler must parse `root` or comma-separated codes, resolve the tree node, compute immediate-child prefix vectors, run metric MDS, and serialize this exact shape:

```python
    @app.get("/api/atlas/{prefix}")
    def atlas(prefix: str):
        codes, final_res, labels, meta_df, embeddings, rq = _load_atlas_inputs()
        pfx = () if prefix == "root" else tuple(int(part) for part in prefix.split(","))
        root = build_tree(codes, final_res, labels)
        focus = node_at(root, pfx)
        if focus is None:
            raise HTTPException(404, "prefix not found")
        children = list(focus.children.values())
        child_prefixes = [child.prefix for child in children]
        distances = sibling_distance_matrix(child_prefixes, rq.codebooks) if children else np.empty((0, 0))
        projection = metric_mds(distances)

        stats_by_prefix = {stat.prefix: stat for stat in node_stats(root, labels)}

        def serialize(node, position):
            center = prefix_vector(node.prefix, rq.codebooks)
            samples = ranked_samples(embeddings, node.item_indices, center)
            parent_center = prefix_vector(node.prefix[:-1], rq.codebooks)
            stats = stats_by_prefix[node.prefix]
            return {
                "prefix": list(node.prefix), "level": node.level,
                "occupancy": node.occupancy, "mean_residual": node.mean_residual,
                "purity": stats.purity, "residual_norm": stats.residual_norm,
                "position": [float(position[0]), float(position[1])],
                "parent_distance": float(np.linalg.norm(center - parent_center)),
                "token_norm": float(np.linalg.norm(rq.codebooks[node.level - 1][node.prefix[-1]])),
                "has_children": bool(node.children),
                "samples": {
                    "representative": [_item(meta_df, i) for i in samples.representative],
                    "outliers": [_item(meta_df, i) for i in samples.outliers],
                },
            }

        focus_center = prefix_vector(focus.prefix, rq.codebooks)
        focus_samples = ranked_samples(embeddings, focus.item_indices, focus_center)
        focus_stats = stats_by_prefix[focus.prefix]
        focus_payload = {
            "prefix": list(focus.prefix), "level": focus.level,
            "occupancy": focus.occupancy, "mean_residual": focus.mean_residual,
            "purity": focus_stats.purity, "residual_norm": focus_stats.residual_norm,
            "samples": {
                "representative": [_item(meta_df, i) for i in focus_samples.representative],
                "outliers": [_item(meta_df, i) for i in focus_samples.outliers],
            },
        }

        return {
            "focus": focus_payload,
            "children": [serialize(node, projection.positions[i]) for i, node in enumerate(children)],
            "projection": {"method": "metric-mds", "metric": "euclidean", "stress": projection.stress, "warning": projection.stress > 0.10},
        }
```

- [ ] **Step 6: Run API and full tests**

Run: `uv run pytest tests/test_server.py -v && uv run pytest -q`

Expected: all tests pass.

- [ ] **Step 7: Commit the API increment**

```bash
git add src/gyo/api/server.py tests/test_server.py
git commit -m "feat: expose semantic atlas API"
```

### Task 4: Pure frontend model and layout

**Files:**
- Create: `src/gyo/web/js/atlas-model.js`
- Create: `src/gyo/web/js/atlas-layout.js`
- Create: `src/gyo/web/js/__tests__/atlas-model.test.js`
- Create: `src/gyo/web/js/__tests__/atlas-layout.test.js`

- [ ] **Step 1: Write failing model tests**

```javascript
// src/gyo/web/js/__tests__/atlas-model.test.js
import { describe, expect, it } from "vitest";
import { createState, prefixKey, selectNode, enterNode, setSampleMode } from "../atlas-model.js";

const payload = { focus: { prefix: [], level: 0 }, children: [{ prefix: [2], has_children: true }], projection: { stress: 0.04 } };

describe("atlas state", () => {
  it("uses stable prefix keys", () => expect(prefixKey([2, 7])).toBe("2,7"));
  it("selects without changing focus", () => {
    const next = selectNode(createState(payload), [2]);
    expect(next.focus).toEqual([]);
    expect(next.selected).toEqual([2]);
  });
  it("enters a node and resets selection", () => {
    const next = enterNode(selectNode(createState(payload), [2]));
    expect(next.focus).toEqual([2]);
    expect(next.selected).toBeNull();
  });
  it("switches between representative and outlier samples", () => {
    expect(setSampleMode(createState(payload), "outliers").sampleMode).toBe("outliers");
  });
});
```

- [ ] **Step 2: Write failing layout tests**

```javascript
// src/gyo/web/js/__tests__/atlas-layout.test.js
import { expect, it } from "vitest";
import { fitTerritories } from "../atlas-layout.js";

it("fits normalized positions and gives occupied groups larger radii", () => {
  const nodes = [
    { prefix: [0], position: [-1, 0], occupancy: 100 },
    { prefix: [1], position: [1, 0], occupancy: 25 },
  ];
  const result = fitTerritories(nodes, 800, 600);
  expect(result[0].cx).toBeLessThan(result[1].cx);
  expect(result[0].r).toBeGreaterThan(result[1].r);
  expect(result.every((p) => p.cx - p.r >= 0 && p.cx + p.r <= 800)).toBe(true);
});
```

- [ ] **Step 3: Run and verify module failures**

Run: `npm test -- --run src/gyo/web/js/__tests__/atlas-model.test.js src/gyo/web/js/__tests__/atlas-layout.test.js`

Expected: FAIL because both modules are missing.

- [ ] **Step 4: Implement immutable state transitions**

```javascript
// src/gyo/web/js/atlas-model.js
export const prefixKey = (prefix) => prefix.length ? prefix.join(",") : "root";
export const createState = (payload) => ({ payload, focus: [...payload.focus.prefix], selected: null, sampleMode: "representative" });
export const selectNode = (state, prefix) => ({ ...state, selected: [...prefix], sampleMode: "representative" });
export const enterNode = (state) => state.selected ? ({ ...state, focus: [...state.selected], selected: null, sampleMode: "representative" }) : state;
export const setSampleMode = (state, mode) => {
  if (!['representative', 'outliers', 'parent'].includes(mode)) throw new Error(`invalid sample mode: ${mode}`);
  return { ...state, sampleMode: mode };
};
export const parentPrefix = (prefix) => prefix.slice(0, -1);
```

- [ ] **Step 5: Implement responsive fitting**

```javascript
// src/gyo/web/js/atlas-layout.js
const clamp = (value, min, max) => Math.max(min, Math.min(max, value));

export function fitTerritories(nodes, width, height) {
  if (!nodes.length) return [];
  const pad = 56;
  const maxOcc = Math.max(...nodes.map((node) => node.occupancy), 1);
  const minR = clamp(Math.min(width, height) * 0.075, 38, 72);
  const maxR = clamp(Math.min(width, height) * 0.15, minR, 116);
  return nodes.map((node) => {
    const r = minR + (maxR - minR) * Math.sqrt(node.occupancy / maxOcc);
    return {
      ...node,
      cx: clamp(width / 2 + node.position[0] * (width / 2 - pad - maxR), r + pad, width - r - pad),
      cy: clamp(height / 2 + node.position[1] * (height / 2 - pad - maxR), r + pad, height - r - pad),
      r,
    };
  });
}
```

- [ ] **Step 6: Run JS tests and commit**

Run: `npm test`

Expected: all Vitest tests pass.

```bash
git add src/gyo/web/js/atlas-model.js src/gyo/web/js/atlas-layout.js src/gyo/web/js/__tests__
git commit -m "feat: add semantic atlas client model and layout"
```

### Task 5: Semantic atlas shell and renderer

**Files:**
- Replace: `src/gyo/web/index.html`
- Replace: `src/gyo/web/style.css`
- Create: `src/gyo/web/js/atlas-render.js`
- Replace: `src/gyo/web/js/main.js`
- Modify: `src/gyo/web/js/api.js`

- [ ] **Step 1: Add the atlas fetch contract**

```javascript
// append to src/gyo/web/js/api.js
export const fetchAtlas = async (prefix = "root") => {
  const response = await fetch(`/api/atlas/${encodeURIComponent(prefix)}`);
  if (!response.ok) throw new Error((await response.json().catch(() => null))?.detail || `atlas request failed (${response.status})`);
  return response.json();
};
```

- [ ] **Step 2: Replace the HTML with the accessible application shell**

The document must contain these stable integration hooks:

```html
<header class="topbar">
  <a class="brand" href="#" aria-label="Voltar à raiz">gyo <span>semantic atlas</span></a>
  <nav id="breadcrumbs" aria-label="Caminho hierárquico"></nav>
  <div id="projectionStatus" class="projection-status"></div>
</header>
<main class="workspace">
  <section class="map-panel" aria-label="Mapa semântico">
    <div class="map-toolbar"><button id="backBtn">Voltar</button><button id="resetViewBtn">Recentrar</button></div>
    <div id="mapLoading" class="map-state">Calculando territórios…</div>
    <svg id="atlas" role="tree" aria-label="Grupos semânticos"></svg>
    <div id="mapError" class="map-state error" hidden><p></p><button id="retryBtn">Tentar novamente</button></div>
  </section>
  <aside id="inspector" class="inspector" aria-live="polite">
    <div class="inspector-empty"><h1>Selecione um território</h1><p>Examine imagens representativas, outliers e a contribuição do token residual.</p></div>
  </aside>
</main>
<template id="imageTemplate"><figure class="sample"><div class="skeleton"></div><img loading="lazy"><figcaption></figcaption></figure></template>
```

- [ ] **Step 3: Implement the scientific-instrument CSS**

Define variables `--canvas:#090e14`, `--surface:#111922`, `--surface-raised:#17232e`, `--line:#283746`, `--text:#e7edf3`, `--muted:#8c9aa8`, `--accent:#52d6b8`, `--warning:#f0b45c`, and `--danger:#ef6b73`. Implement:

- a 52px topbar;
- `.workspace` as `grid-template-columns:minmax(0,7fr) minmax(320px,3fr)`;
- a full-height map panel with SVG occupying the available area;
- territory circles and cards with visible focus states;
- a scrollable inspector with a two-column image grid;
- stable skeleton aspect ratios;
- a single-column layout below 860px where the inspector becomes a 42vh bottom panel;
- `prefers-reduced-motion` disabling transitions.

- [ ] **Step 4: Implement SVG map and inspector rendering**

`src/gyo/web/js/atlas-render.js` must export:

```javascript
import { prefixKey } from "./atlas-model.js";

export function renderMap(svg, placements, state, handlers) {
  svg.replaceChildren();
  svg.setAttribute("viewBox", `0 0 ${handlers.width} ${handlers.height}`);
  for (const node of placements) {
    const group = document.createElementNS("http://www.w3.org/2000/svg", "g");
    group.classList.add("territory");
    group.dataset.prefix = prefixKey(node.prefix);
    group.setAttribute("role", "treeitem");
    group.setAttribute("tabindex", "0");
    group.setAttribute("aria-label", `Grupo ${prefixKey(node.prefix)}, ${node.occupancy} imagens`);
    group.setAttribute("transform", `translate(${node.cx} ${node.cy})`);
    const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    circle.setAttribute("r", node.r);
    group.append(circle);
    const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
    label.textContent = `c${node.prefix.at(-1)} · ${node.occupancy}`;
    label.setAttribute("y", -node.r + 18);
    group.append(label);
    handlers.appendPreview(group, node);
    group.addEventListener("click", () => handlers.select(node));
    group.addEventListener("dblclick", () => node.has_children && handlers.enter(node));
    group.addEventListener("keydown", (event) => {
      if (event.key === "Enter") handlers.select(node);
      if (event.key === " " && node.has_children) { event.preventDefault(); handlers.enter(node); }
    });
    svg.append(group);
  }
}

export function renderInspector(container, node, mode, handlers) {
  if (!node) return;
  const items = mode === "outliers" ? node.samples.outliers : node.samples.representative;
  container.innerHTML = `<header><p class="eyebrow">Território ${prefixKey(node.prefix)}</p><h1>${node.occupancy} imagens</h1></header>
    <dl class="metrics"><div><dt>Distância do pai</dt><dd>${node.parent_distance.toFixed(4)}</dd></div><div><dt>Norma do token</dt><dd>${node.token_norm.toFixed(4)}</dd></div></dl>
    <div class="segmented"><button data-mode="representative">Representativas</button><button data-mode="outliers">Outliers</button><button data-mode="parent">Comparar pai</button></div>
    <div class="sample-grid"></div>${node.has_children ? '<button class="primary" data-enter>Entrar no grupo</button>' : ''}`;
  container.querySelectorAll("[data-mode]").forEach((button) => button.addEventListener("click", () => handlers.mode(button.dataset.mode)));
  container.querySelector("[data-enter]")?.addEventListener("click", handlers.enter);
  handlers.appendSamples(container.querySelector(".sample-grid"), items);
}
```

Keep private helper functions in this file and preserve the two exported function contracts shown above.

- [ ] **Step 5: Replace orchestration in `main.js`**

Implement a single `load(prefix)` path that:

1. shows `#mapLoading` and hides stale errors;
2. calls `fetchAtlas(prefixKey(prefix))`;
3. creates state with `createState`;
4. fits immediate children using `fitTerritories` and current SVG bounds;
5. renders breadcrumbs, projection status, map, and inspector;
6. retains the last successful payload if a later request fails;
7. connects retry, back, root-brand, resize, territory selection, visible **Enter group**, double click, and keyboard actions;
8. renders a visible warning when `payload.projection.warning` is true;
9. lazy-loads preview and inspector images through `/thumb/{idx}` with a per-slot retry button on error.

For `parent` sample mode, render `state.payload.focus.samples.representative` beside the selected child's representative samples. Cache atlas responses by `prefixKey`.

- [ ] **Step 6: Run unit and API tests**

Run: `npm test && uv run pytest -q`

Expected: all tests pass.

- [ ] **Step 7: Commit the frontend replacement**

```bash
git add src/gyo/web
git commit -m "feat: replace icicle with semantic atlas interface"
```

### Task 6: Browser-level behavior and responsive verification

**Files:**
- Replace: `tests/test_e2e_vi_flows.py`

- [ ] **Step 1: Replace obsolete icicle flows with atlas flows**

The Playwright suite must run these assertions against the existing seeded server fixture:

```python
async def test_atlas_boot(page):
    await page.goto(BASE)
    await page.wait_for_selector("#atlas .territory", timeout=TIMEOUT)
    assert await page.locator("#projectionStatus").inner_text()
    assert await page.locator("#atlas .territory").count() > 0


async def test_selection_and_inspector(page):
    await boot(page)
    territory = page.locator("#atlas .territory").first
    await territory.click()
    await page.wait_for_selector("#inspector .sample-grid img")
    assert "Distância do pai" in await page.locator("#inspector").inner_text()


async def test_enter_and_breadcrumb_return(page):
    await boot(page)
    territory = page.locator("#atlas .territory").filter(has=page.locator("text=/./")).first
    await territory.click()
    enter = page.locator("#inspector [data-enter]")
    if await enter.count():
        await enter.click()
        await page.wait_for_timeout(250)
        assert await page.locator("#breadcrumbs button").count() >= 2
        await page.locator("#breadcrumbs button").first.click()
        await page.wait_for_timeout(250)
        assert await page.locator("#breadcrumbs button").count() == 1


async def test_outlier_and_keyboard_flows(page):
    await boot(page)
    territory = page.locator("#atlas .territory").first
    await territory.focus()
    await page.keyboard.press("Enter")
    await page.locator('#inspector [data-mode="outliers"]').click()
    assert await page.locator("#inspector .sample-grid img").count() > 0


async def test_responsive_layout(page):
    await boot(page)
    await page.set_viewport_size({"width": 760, "height": 900})
    await page.wait_for_timeout(250)
    columns = await page.locator(".workspace").evaluate("el => getComputedStyle(el).gridTemplateColumns")
    assert " " not in columns.strip()
    assert await page.locator("#atlas .territory").count() > 0
```

Remove tests for collapse spines, residual/image modes, the depth rail, and the old tooltip because those behaviors were intentionally removed.

- [ ] **Step 2: Run browser verification**

Make `tests/test_e2e_vi_flows.py` self-contained: create a `TemporaryDirectory`, seed the same images, tables, embeddings, and codebooks defined in `tests/test_server.py::_seed_run`, launch `uvicorn.run(create_app(run_dir), host="127.0.0.1", port=8000)` in a `multiprocessing.Process`, poll `/api/tree` until ready, run the browser flows, and terminate/join the process in `finally`. Then run:

`uv run python tests/test_e2e_vi_flows.py`

Expected: all atlas flow labels report passed and the process exits 0.

- [ ] **Step 3: Run the full verification matrix**

Run:

```bash
uv run pytest -q
npm test
uv run python tests/test_e2e_vi_flows.py
```

Expected: all three commands exit 0.

- [ ] **Step 4: Commit browser verification**

```bash
git add tests/test_e2e_vi_flows.py
git commit -m "test: cover semantic atlas user flows"
```

### Task 7: Final accessibility and regression audit

**Files:**
- Modify: `src/gyo/web/index.html`
- Modify: `src/gyo/web/style.css`
- Modify: `src/gyo/web/js/atlas-render.js`
- Modify: `src/gyo/web/js/main.js`

- [ ] **Step 1: Verify the supported user journeys manually**

At desktop 1440×900 and tablet 768×1024, verify:

- all representative images are large enough to compare;
- selected territory remains visible while the inspector scrolls;
- projection warning is legible and does not block navigation;
- map, back, breadcrumb, sample-mode, retry, and enter controls work with Tab, Enter, and Space;
- reduced-motion mode removes zoom transitions;
- failed thumbnails retain their slot and expose retry;
- no text or control overlaps at 200% browser zoom.

- [ ] **Step 2: Run final automated verification from a clean process**

```bash
uv run pytest -q
npm test
uv run python tests/test_e2e_vi_flows.py
git diff --check
```

Expected: all tests pass, browser flows exit 0, and `git diff --check` prints nothing.

- [ ] **Step 3: Commit any audit fixes**

If Step 1 required changes:

```bash
git add src/gyo/web
git commit -m "fix: polish semantic atlas accessibility"
```

If no changes were required, do not create an empty commit.
