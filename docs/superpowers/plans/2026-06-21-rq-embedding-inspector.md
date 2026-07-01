# RQ Embedding Inspector — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a tool that quantizes image embeddings with a residual quantizer (RQ) and shows the resulting code-prefix tree in a browser, coloring nodes by mean residual and sizing them by occupancy — surfacing geometric signals visually, never emitting a verdict.

**Architecture:** Five decoupled disk-backed stages (prep-data → extract → fit-rq → encode → serve). A swappable `Embedder` Protocol turns an image folder into an L2-normalized matrix. A hand-written `ResidualQuantizer` (sum reconstruction, EMA training, no neural decoder) produces semantic IDs and residuals. A code-prefix tree is built by O(N·L) aggregation. A FastAPI backend serves the tree JSON and bucket thumbnails to a vanilla-JS frontend.

**Tech Stack:** Python 3.12, NumPy, pandas + pyarrow (parquet), PyTorch + open_clip (MobileCLIP2-S0 image encoder), torchvision (Fashion-MNIST), Pillow, FastAPI + uvicorn, Typer (CLI), pytest. Frontend: vanilla JS + D3.

## Global Constraints

- **Python pinned to 3.12** via uv (`.python-version` = `3.12`); do not target 3.14.
- **RQ reconstruction is the SUM of codewords** (`x̂ = Σ codewords`). No neural decoder, ever. The optional `d→d'` projection must be **linear** and is **OFF by default**.
- **Codewords live in embedding space** — never transform them in a way that breaks `x̂ = Σ codewords`. The coarse-to-fine hierarchy depends on this.
- **RQ defaults:** `L=3` levels, `K=256` codewords per level.
- **Embeddings are L2-normalized** `(N, d)` float32.
- **Embedder extraction is mandatory-correct:** call `model.eval()` AND `reparameterize_model(model)` before inference (batchnorm / reparameterizable blocks). Skipping this yields inconsistent embeddings.
- **Labels are optional everywhere** — used only to color/inspect, never required by the RQ or the tree.
- **The tool emits NO verdict.** No per-node "data vs model" label, no "recommended action". `signals.py` does neutral aggregation only. The interpretation table is static help text, never applied to nodes.
- **Fashion-MNIST default subsample = 10000 images** (`--n`, configurable).
- **MobileCLIP2-S0 load:** `open_clip.create_model_and_transforms('MobileCLIP2-S0', pretrained='dfndr2b')`.

---

## File Structure

```
gyo/
  pyproject.toml                 # uv project, deps, console script
  .python-version                # "3.12"
  src/gyo/__init__.py
  src/gyo/io/__init__.py
  src/gyo/io/store.py            # save/load embeddings (.npy) and tables (.parquet)
  src/gyo/rq/__init__.py
  src/gyo/rq/quantizer.py        # ResidualQuantizer: fit (EMA), encode, reconstruct, save/load
  src/gyo/embedders/__init__.py
  src/gyo/embedders/base.py      # Embedder Protocol + DummyEmbedder (for tests)
  src/gyo/embedders/mobileclip.py# MobileCLIP2-S0 image encoder
  src/gyo/data/__init__.py
  src/gyo/data/fashion_mnist.py  # dump Fashion-MNIST -> images/*.png + labels.csv
  src/gyo/tree/__init__.py
  src/gyo/tree/build.py          # build_tree: code-prefix aggregation -> TreeNode
  src/gyo/tree/signals.py        # neutral per-node stats (occupancy, residual, dead codewords)
  src/gyo/api/__init__.py
  src/gyo/api/server.py          # FastAPI create_app: /tree, /node/{prefix}, /thumb/{idx}
  src/gyo/web/index.html         # tree UI shell
  src/gyo/web/app.js             # D3 tree, level slider, bucket panel, legend
  src/gyo/web/style.css
  src/gyo/cli.py                 # Typer app: prep-data | extract | fit-rq | encode | serve
  tests/test_store.py
  tests/test_quantizer.py
  tests/test_embedder_base.py
  tests/test_fashion_mnist.py
  tests/test_tree_build.py
  tests/test_signals.py
  tests/test_cli.py
  tests/test_server.py
```

**Artifact layout produced at runtime** (under a `--data-dir`, default `./run`):
```
run/
  images/*.png
  labels.csv                     # columns: path,label
  embeddings.npy                 # (N, d) float32 L2-normalized
  meta.parquet                   # columns: idx,path,label
  codebooks/v1/                  # level_0.npy ... level_{L-1}.npy + config.json
  codes.parquet                  # idx, c_0..c_{L-1}, j, r_0..r_{L-1}, final_residual
```

---

## Task 1: Project scaffold + io/store

**Files:**
- Create: `pyproject.toml`, `.python-version`, `src/gyo/__init__.py`, `src/gyo/io/__init__.py`, `src/gyo/io/store.py`
- Test: `tests/test_store.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `save_embeddings(path: str | Path, emb: np.ndarray) -> None`
  - `load_embeddings(path: str | Path) -> np.ndarray`
  - `save_table(path: str | Path, df: pd.DataFrame) -> None`  (parquet)
  - `load_table(path: str | Path) -> pd.DataFrame`

- [ ] **Step 1: Scaffold the uv project**

```bash
cd /home/red/repos/gyo
echo "3.12" > .python-version
uv init --no-workspace --name gyo --lib --python 3.12 .
```

Then overwrite `pyproject.toml` with:

```toml
[project]
name = "gyo"
version = "0.1.0"
description = "RQ-based embedding-space inspection tool (Semantic IDs)"
requires-python = ">=3.12,<3.13"
dependencies = [
    "numpy>=1.26",
    "pandas>=2.2",
    "pyarrow>=15",
    "pillow>=10",
    "torch>=2.2",
    "torchvision>=0.17",
    "open_clip_torch>=2.24",
    "timm>=0.9",
    "fastapi>=0.110",
    "uvicorn>=0.29",
    "typer>=0.12",
]

[project.scripts]
gyo = "gyo.cli:app"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/gyo"]

[dependency-groups]
dev = ["pytest>=8"]
```

- [ ] **Step 2: Write the failing test**

`tests/test_store.py`:

```python
import numpy as np
import pandas as pd
from gyo.io.store import save_embeddings, load_embeddings, save_table, load_table


def test_embeddings_roundtrip(tmp_path):
    emb = np.random.rand(5, 4).astype(np.float32)
    p = tmp_path / "emb.npy"
    save_embeddings(p, emb)
    loaded = load_embeddings(p)
    assert loaded.dtype == np.float32
    np.testing.assert_array_equal(loaded, emb)


def test_table_roundtrip(tmp_path):
    df = pd.DataFrame({"idx": [0, 1], "path": ["a.png", "b.png"], "label": ["t", "u"]})
    p = tmp_path / "meta.parquet"
    save_table(p, df)
    loaded = load_table(p)
    pd.testing.assert_frame_equal(loaded, df)
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/test_store.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'gyo.io.store'`

- [ ] **Step 4: Write minimal implementation**

`src/gyo/io/store.py`:

```python
from pathlib import Path
import numpy as np
import pandas as pd


def save_embeddings(path, emb: np.ndarray) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.save(path, emb.astype(np.float32, copy=False))


def load_embeddings(path) -> np.ndarray:
    return np.load(path).astype(np.float32, copy=False)


def save_table(path, df: pd.DataFrame) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def load_table(path) -> pd.DataFrame:
    return pd.read_parquet(path)
```

Note: `np.save` appends `.npy` if missing; tests use an explicit `.npy` path so the roundtrip path matches.

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_store.py -v`
Expected: PASS (2 passed)

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml .python-version uv.lock src/gyo tests/test_store.py
git commit -m "feat: project scaffold + io/store roundtrip"
```

---

## Task 2: RQ — encode, reconstruct, residuals, save/load

**Files:**
- Create: `src/gyo/rq/__init__.py`, `src/gyo/rq/quantizer.py`
- Test: `tests/test_quantizer.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `@dataclass EncodeResult: codes: np.ndarray  # (N, L) int64; residuals: np.ndarray  # (N, L) float32 per-level residual norm BEFORE subtracting that level; final_residual: np.ndarray  # (N,) float32 == ||r_L||; tie_index: np.ndarray  # (N,) int64`
  - `class ResidualQuantizer:`
    - `__init__(self, num_levels=3, codebook_size=256, dim=None, proj_dim=None, seed=0)`
    - attribute `codebooks: list[np.ndarray]` each `(K, dim)` float32 (created lazily in `fit`, or settable directly for tests)
    - `encode(self, x: np.ndarray) -> EncodeResult`
    - `reconstruct(self, codes: np.ndarray) -> np.ndarray  # (N, dim)`
    - `save(self, directory) -> None` / `@classmethod load(cls, directory) -> "ResidualQuantizer"`
- Later tasks rely on `EncodeResult.codes`, `EncodeResult.final_residual`, `EncodeResult.tie_index`.

- [ ] **Step 1: Write the failing test**

`tests/test_quantizer.py`:

```python
import numpy as np
from gyo.rq.quantizer import ResidualQuantizer


def _toy_quantizer():
    rq = ResidualQuantizer(num_levels=2, codebook_size=2, dim=2, seed=0)
    # level 0 separates along x; level 1 refines along y
    rq.codebooks = [
        np.array([[1.0, 0.0], [-1.0, 0.0]], dtype=np.float32),
        np.array([[0.0, 0.5], [0.0, -0.5]], dtype=np.float32),
    ]
    return rq


def test_encode_picks_nearest_and_residual_decreases():
    rq = _toy_quantizer()
    x = np.array([[1.0, 0.5]], dtype=np.float32)  # closest to cb0[0] then cb1[0]
    res = rq.encode(x)
    assert res.codes.tolist() == [[0, 0]]
    # residual norm before level 0 >= before level 1 >= final
    assert res.residuals[0, 0] >= res.residuals[0, 1] >= res.final_residual[0]
    assert res.final_residual[0] < 1e-6  # exact reconstruction for this point


def test_reconstruct_is_sum_of_codewords():
    rq = _toy_quantizer()
    codes = np.array([[0, 1]], dtype=np.int64)
    recon = rq.reconstruct(codes)
    expected = rq.codebooks[0][0] + rq.codebooks[1][1]
    np.testing.assert_allclose(recon[0], expected, atol=1e-6)


def test_tie_index_disambiguates_identical_tuples():
    rq = _toy_quantizer()
    x = np.array([[1.0, 0.5], [1.0, 0.5], [-1.0, -0.5]], dtype=np.float32)
    res = rq.encode(x)
    # first two share the same tuple -> j = 0,1 ; third is unique -> j = 0
    assert res.tie_index.tolist() == [0, 1, 0]


def test_save_load_roundtrip(tmp_path):
    rq = _toy_quantizer()
    rq.save(tmp_path / "cb")
    rq2 = ResidualQuantizer.load(tmp_path / "cb")
    assert rq2.num_levels == 2 and rq2.codebook_size == 2 and rq2.dim == 2
    for a, b in zip(rq.codebooks, rq2.codebooks):
        np.testing.assert_array_equal(a, b)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_quantizer.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'gyo.rq.quantizer'`

- [ ] **Step 3: Write minimal implementation**

`src/gyo/rq/quantizer.py`:

```python
import json
from dataclasses import dataclass
from pathlib import Path
import numpy as np


@dataclass
class EncodeResult:
    codes: np.ndarray        # (N, L) int64
    residuals: np.ndarray    # (N, L) float32: ||r_i|| before subtracting level i
    final_residual: np.ndarray  # (N,) float32: ||r_L||
    tie_index: np.ndarray    # (N,) int64: j within identical-tuple leaves


def _assign(residual: np.ndarray, codebook: np.ndarray) -> np.ndarray:
    # residual (N, d), codebook (K, d) -> nearest index (N,)
    d2 = ((residual[:, None, :] - codebook[None, :, :]) ** 2).sum(-1)
    return d2.argmin(1)


def _tie_index(codes: np.ndarray) -> np.ndarray:
    seen: dict[tuple, int] = {}
    out = np.empty(len(codes), dtype=np.int64)
    for i, row in enumerate(map(tuple, codes.tolist())):
        out[i] = seen.get(row, 0)
        seen[row] = out[i] + 1
    return out


class ResidualQuantizer:
    def __init__(self, num_levels=3, codebook_size=256, dim=None, proj_dim=None, seed=0):
        self.num_levels = num_levels
        self.codebook_size = codebook_size
        self.dim = dim
        self.proj_dim = proj_dim  # reserved; linear projection OFF by default
        self.seed = seed
        self.codebooks: list[np.ndarray] = []

    def encode(self, x: np.ndarray) -> EncodeResult:
        x = np.asarray(x, dtype=np.float32)
        n = x.shape[0]
        codes = np.empty((n, self.num_levels), dtype=np.int64)
        residuals = np.empty((n, self.num_levels), dtype=np.float32)
        r = x.copy()
        for i in range(self.num_levels):
            residuals[:, i] = np.linalg.norm(r, axis=1)
            idx = _assign(r, self.codebooks[i])
            codes[:, i] = idx
            r = r - self.codebooks[i][idx]
        final_residual = np.linalg.norm(r, axis=1).astype(np.float32)
        return EncodeResult(codes, residuals, final_residual, _tie_index(codes))

    def reconstruct(self, codes: np.ndarray) -> np.ndarray:
        codes = np.asarray(codes, dtype=np.int64)
        out = np.zeros((codes.shape[0], self.dim), dtype=np.float32)
        for i in range(self.num_levels):
            out += self.codebooks[i][codes[:, i]]
        return out

    def save(self, directory) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        for i, cb in enumerate(self.codebooks):
            np.save(d / f"level_{i}.npy", cb)
        (d / "config.json").write_text(json.dumps({
            "num_levels": self.num_levels,
            "codebook_size": self.codebook_size,
            "dim": self.dim,
            "proj_dim": self.proj_dim,
            "seed": self.seed,
        }))

    @classmethod
    def load(cls, directory) -> "ResidualQuantizer":
        d = Path(directory)
        cfg = json.loads((d / "config.json").read_text())
        rq = cls(**cfg)
        rq.codebooks = [np.load(d / f"level_{i}.npy") for i in range(cfg["num_levels"])]
        return rq
```

`dim` is set from config on load; for fresh instances in tests it is passed to `__init__`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_quantizer.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/gyo/rq tests/test_quantizer.py
git commit -m "feat: RQ encode/reconstruct (sum), residuals, tie index, save/load"
```

---

## Task 3: RQ — EMA k-means training (fit)

**Files:**
- Modify: `src/gyo/rq/quantizer.py` (add `fit`)
- Test: `tests/test_quantizer.py` (add training tests)

**Interfaces:**
- Consumes: `ResidualQuantizer` from Task 2.
- Produces: `fit(self, x: np.ndarray, iters=10, ema_decay=0.99) -> "ResidualQuantizer"` — initializes and trains `self.codebooks` (one EMA k-means per level over cascaded residuals), sets `self.dim` from `x`. Returns self.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_quantizer.py`:

```python
def test_fit_reduces_residual_and_sets_dim():
    rng = np.random.default_rng(0)
    centers = np.array([[3, 3], [-3, -3], [3, -3], [-3, 3]], dtype=np.float32)
    x = np.repeat(centers, 50, axis=0) + rng.normal(0, 0.1, (200, 2)).astype(np.float32)
    rq = ResidualQuantizer(num_levels=2, codebook_size=4, seed=0)
    rq.fit(x, iters=25)
    assert rq.dim == 2
    assert len(rq.codebooks) == 2
    res = rq.encode(x)
    # well-separated clusters -> tiny final residual after fit
    assert res.final_residual.mean() < 0.5
    # cascaded residual energy decreases on average
    assert res.residuals[:, 0].mean() >= res.residuals[:, 1].mean()


def test_fit_is_deterministic_with_seed():
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1, (120, 3)).astype(np.float32)
    a = ResidualQuantizer(num_levels=2, codebook_size=8, seed=7).fit(x, iters=10)
    b = ResidualQuantizer(num_levels=2, codebook_size=8, seed=7).fit(x, iters=10)
    for ca, cb in zip(a.codebooks, b.codebooks):
        np.testing.assert_allclose(ca, cb, atol=1e-5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_quantizer.py::test_fit_reduces_residual_and_sets_dim -v`
Expected: FAIL with `AttributeError: 'ResidualQuantizer' object has no attribute 'fit'`

- [ ] **Step 3: Write minimal implementation**

Add this method to `ResidualQuantizer` in `src/gyo/rq/quantizer.py`:

```python
    def fit(self, x: np.ndarray, iters: int = 10, ema_decay: float = 0.99) -> "ResidualQuantizer":
        x = np.asarray(x, dtype=np.float32)
        self.dim = x.shape[1]
        rng = np.random.default_rng(self.seed)
        self.codebooks = []
        r = x.copy()
        for _level in range(self.num_levels):
            # k-means++-ish init: random distinct points from current residuals
            init_idx = rng.choice(len(r), size=self.codebook_size, replace=len(r) < self.codebook_size)
            cb = r[init_idx].copy()
            counts = np.ones(self.codebook_size, dtype=np.float32)
            for _ in range(iters):
                assign = _assign(r, cb)
                for k in range(self.codebook_size):
                    members = r[assign == k]
                    if len(members) == 0:
                        continue
                    target = members.mean(0)
                    cb[k] = ema_decay * cb[k] + (1 - ema_decay) * target
                    counts[k] = ema_decay * counts[k] + (1 - ema_decay) * len(members)
            self.codebooks.append(cb.astype(np.float32))
            r = r - cb[_assign(r, cb)]
        return self
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_quantizer.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add src/gyo/rq/quantizer.py tests/test_quantizer.py
git commit -m "feat: RQ EMA k-means training (fit) over cascaded residuals"
```

---

## Task 4: Embedder Protocol + DummyEmbedder

**Files:**
- Create: `src/gyo/embedders/__init__.py`, `src/gyo/embedders/base.py`
- Test: `tests/test_embedder_base.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `class Embedder(Protocol): def embed_folder(self, folder: str | Path) -> tuple[np.ndarray, pd.DataFrame]: ...`
  - `def list_images(folder) -> list[Path]` — sorted `*.png/*.jpg/*.jpeg`.
  - `def l2_normalize(x: np.ndarray) -> np.ndarray`
  - `class DummyEmbedder` — deterministic per-file embeddings for tests; returns `(emb (N,d) L2-normalized, meta DataFrame[idx,path,label])`. Reads optional sibling `labels.csv` (columns `path,label`) to fill `label`, else empty string.

- [ ] **Step 1: Write the failing test**

`tests/test_embedder_base.py`:

```python
import numpy as np
import pandas as pd
from PIL import Image
from gyo.embedders.base import DummyEmbedder, l2_normalize, list_images


def _make_images(folder, n=3):
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        Image.new("L", (28, 28), color=i * 10).save(folder / f"img_{i}.png")


def test_l2_normalize_unit_rows():
    x = np.array([[3.0, 4.0], [0.0, 2.0]], dtype=np.float32)
    out = l2_normalize(x)
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), [1.0, 1.0], atol=1e-6)


def test_dummy_embedder_shapes_and_meta(tmp_path):
    _make_images(tmp_path, 3)
    pd.DataFrame({"path": ["img_0.png", "img_1.png"], "label": ["a", "b"]}).to_csv(
        tmp_path / "labels.csv", index=False)
    emb, meta = DummyEmbedder(dim=8).embed_folder(tmp_path)
    assert emb.shape == (3, 8)
    np.testing.assert_allclose(np.linalg.norm(emb, axis=1), np.ones(3), atol=1e-6)
    assert list(meta.columns) == ["idx", "path", "label"]
    assert meta.loc[0, "label"] == "a" and meta.loc[2, "label"] == ""


def test_list_images_sorted(tmp_path):
    _make_images(tmp_path, 3)
    files = list_images(tmp_path)
    assert [f.name for f in files] == ["img_0.png", "img_1.png", "img_2.png"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_embedder_base.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'gyo.embedders.base'`

- [ ] **Step 3: Write minimal implementation**

`src/gyo/embedders/base.py`:

```python
from pathlib import Path
from typing import Protocol
import hashlib
import numpy as np
import pandas as pd

_EXTS = {".png", ".jpg", ".jpeg"}


def list_images(folder) -> list[Path]:
    return sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in _EXTS)


def l2_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


def _load_labels(folder: Path) -> dict[str, str]:
    csv = folder / "labels.csv"
    if not csv.exists():
        return {}
    df = pd.read_csv(csv)
    return {str(p): str(l) for p, l in zip(df["path"], df["label"])}


class Embedder(Protocol):
    def embed_folder(self, folder) -> tuple[np.ndarray, pd.DataFrame]: ...


class DummyEmbedder:
    """Deterministic hash-based embeddings; for tests and pipeline wiring."""

    def __init__(self, dim: int = 8):
        self.dim = dim

    def embed_folder(self, folder) -> tuple[np.ndarray, pd.DataFrame]:
        folder = Path(folder)
        files = list_images(folder)
        labels = _load_labels(folder)
        vecs = []
        rows = []
        for idx, f in enumerate(files):
            seed = int(hashlib.md5(f.name.encode()).hexdigest(), 16) % (2**32)
            vecs.append(np.random.default_rng(seed).normal(0, 1, self.dim))
            rows.append({"idx": idx, "path": f.name, "label": labels.get(f.name, "")})
        emb = l2_normalize(np.array(vecs, dtype=np.float32)) if vecs else np.zeros((0, self.dim), np.float32)
        return emb, pd.DataFrame(rows, columns=["idx", "path", "label"])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_embedder_base.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/gyo/embedders/__init__.py src/gyo/embedders/base.py tests/test_embedder_base.py
git commit -m "feat: Embedder Protocol + DummyEmbedder + L2 normalize helpers"
```

---

## Task 5: MobileCLIP2-S0 embedder

**Files:**
- Create: `src/gyo/embedders/mobileclip.py`
- Test: `tests/test_embedder_base.py` (add a slow smoke test)

**Interfaces:**
- Consumes: `list_images`, `l2_normalize`, `_load_labels` patterns from Task 4.
- Produces: `class MobileCLIPEmbedder: __init__(self, model_name="MobileCLIP2-S0", pretrained="dfndr2b", device="cpu", batch_size=64); embed_folder(folder) -> tuple[np.ndarray, pd.DataFrame]` with the same return contract as `DummyEmbedder`.

- [ ] **Step 1: Write the failing (slow) test**

Append to `tests/test_embedder_base.py`:

```python
import pytest


@pytest.mark.slow
def test_mobileclip_smoke(tmp_path):
    from gyo.embedders.mobileclip import MobileCLIPEmbedder
    _make_images(tmp_path, 2)
    emb, meta = MobileCLIPEmbedder().embed_folder(tmp_path)
    assert emb.shape[0] == 2
    np.testing.assert_allclose(np.linalg.norm(emb, axis=1), np.ones(2), atol=1e-5)
    assert list(meta.columns) == ["idx", "path", "label"]
```

Register the marker — append to `pyproject.toml`:

```toml
[tool.pytest.ini_options]
markers = ["slow: requires model download / heavy compute"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_embedder_base.py::test_mobileclip_smoke -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'gyo.embedders.mobileclip'`

- [ ] **Step 3: Write minimal implementation**

`src/gyo/embedders/mobileclip.py`:

```python
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import open_clip
from PIL import Image

from gyo.embedders.base import list_images, l2_normalize, _load_labels


def _reparameterize(model):
    try:
        from timm.utils import reparameterize_model
    except Exception:  # pragma: no cover - fallback path
        from open_clip.model import reparameterize_model  # type: ignore
    return reparameterize_model(model)


class MobileCLIPEmbedder:
    def __init__(self, model_name="MobileCLIP2-S0", pretrained="dfndr2b",
                 device="cpu", batch_size=64):
        self.device = device
        self.batch_size = batch_size
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained)
        model.eval()                       # MANDATORY before reparameterize (batchnorm)
        model = _reparameterize(model)     # MANDATORY for reparameterizable blocks
        self.model = model.to(device)
        self.preprocess = preprocess

    @torch.no_grad()
    def embed_folder(self, folder) -> tuple[np.ndarray, pd.DataFrame]:
        folder = Path(folder)
        files = list_images(folder)
        labels = _load_labels(folder)
        out = []
        for start in range(0, len(files), self.batch_size):
            batch = files[start:start + self.batch_size]
            tensors = [self.preprocess(Image.open(f).convert("RGB")) for f in batch]
            x = torch.stack(tensors).to(self.device)
            feats = self.model.encode_image(x).cpu().numpy()
            out.append(feats)
        emb = np.concatenate(out, 0) if out else np.zeros((0, 512), np.float32)
        emb = l2_normalize(emb)
        meta = pd.DataFrame(
            [{"idx": i, "path": f.name, "label": labels.get(f.name, "")}
             for i, f in enumerate(files)],
            columns=["idx", "path", "label"])
        return emb, meta
```

- [ ] **Step 4: Run the slow test (verifies real load + reparameterize + L2)**

Run: `uv run pytest tests/test_embedder_base.py::test_mobileclip_smoke -v -m slow`
Expected: PASS (downloads model on first run; convert("RGB") handles grayscale Fashion-MNIST).
If the `MobileCLIP2-S0`/`dfndr2b` identifier is rejected by the installed open_clip, run `uv run python -c "import open_clip; print([m for m in open_clip.list_pretrained() if 'obileCLIP' in m[0]])"` and use the exact returned name/tag before continuing.

- [ ] **Step 5: Commit**

```bash
git add src/gyo/embedders/mobileclip.py tests/test_embedder_base.py pyproject.toml
git commit -m "feat: MobileCLIP2-S0 image embedder (eval + reparameterize, L2-norm)"
```

---

## Task 6: Fashion-MNIST prep

**Files:**
- Create: `src/gyo/data/__init__.py`, `src/gyo/data/fashion_mnist.py`
- Test: `tests/test_fashion_mnist.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `prepare_fashion_mnist(out_dir, n=10000, split="train", seed=0) -> Path` — downloads via torchvision, writes `<out_dir>/images/*.png` and `<out_dir>/labels.csv` (columns `path,label`, `path` is the bare filename, `label` is the class name), returns the images dir. The function accepts an injectable `dataset` iterable of `(PIL.Image, label_int)` for testing without a download.

- [ ] **Step 1: Write the failing test**

`tests/test_fashion_mnist.py`:

```python
import pandas as pd
from PIL import Image
from gyo.data.fashion_mnist import prepare_fashion_mnist, FASHION_CLASSES


def _fake_dataset(n=5):
    return [(Image.new("L", (28, 28), color=i * 5), i % 10) for i in range(n)]


def test_prepare_dumps_images_and_labels(tmp_path):
    img_dir = prepare_fashion_mnist(tmp_path, n=4, dataset=_fake_dataset(10))
    pngs = sorted(img_dir.glob("*.png"))
    assert len(pngs) == 4
    labels = pd.read_csv(tmp_path / "labels.csv")
    assert list(labels.columns) == ["path", "label"]
    assert len(labels) == 4
    assert labels.loc[0, "label"] in FASHION_CLASSES
    assert (img_dir / labels.loc[0, "path"]).exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_fashion_mnist.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'gyo.data.fashion_mnist'`

- [ ] **Step 3: Write minimal implementation**

`src/gyo/data/fashion_mnist.py`:

```python
from pathlib import Path
import pandas as pd

FASHION_CLASSES = [
    "T-shirt_top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle_boot",
]


def _default_dataset(out_dir: Path, split: str):
    from torchvision.datasets import FashionMNIST
    train = split == "train"
    return FashionMNIST(root=str(out_dir / "_torchvision"), train=train, download=True)


def prepare_fashion_mnist(out_dir, n=10000, split="train", seed=0, dataset=None) -> Path:
    out_dir = Path(out_dir)
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    if dataset is None:
        dataset = _default_dataset(out_dir, split)
    rows = []
    for i in range(min(n, len(dataset))):
        img, label_int = dataset[i]
        fname = f"{i:06d}.png"
        img.convert("L").save(img_dir / fname)
        rows.append({"path": fname, "label": FASHION_CLASSES[label_int]})
    pd.DataFrame(rows, columns=["path", "label"]).to_csv(out_dir / "labels.csv", index=False)
    return img_dir
```

`seed` is reserved for future random subsampling; current behavior takes the first `n` deterministically.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_fashion_mnist.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add src/gyo/data tests/test_fashion_mnist.py
git commit -m "feat: Fashion-MNIST prep -> images/*.png + labels.csv"
```

---

## Task 7: Tree build (code-prefix aggregation)

**Files:**
- Create: `src/gyo/tree/__init__.py`, `src/gyo/tree/build.py`
- Test: `tests/test_tree_build.py`

**Interfaces:**
- Consumes: `EncodeResult.codes`, `EncodeResult.final_residual` from Task 2.
- Produces:
  - `@dataclass TreeNode: prefix: tuple[int, ...]; level: int; occupancy: int; mean_residual: float; item_indices: list[int]; children: dict[int, "TreeNode"]`
  - `build_tree(codes: np.ndarray, final_residual: np.ndarray, labels: list[str] | None = None) -> TreeNode` — root has `prefix=()`, `level=0`; children keyed by `c_0`, grandchildren by `c_1`, down to `L` levels.
  - `node_at(root: TreeNode, prefix: tuple[int, ...]) -> TreeNode | None`

- [ ] **Step 1: Write the failing test**

`tests/test_tree_build.py`:

```python
import numpy as np
from gyo.tree.build import build_tree, node_at


def test_tree_aggregates_occupancy_children_and_residual():
    codes = np.array([[0, 0], [0, 1], [1, 0]], dtype=np.int64)
    final_res = np.array([0.2, 0.4, 0.6], dtype=np.float32)
    root = build_tree(codes, final_res)
    assert root.occupancy == 3
    assert set(root.children.keys()) == {0, 1}
    n0 = root.children[0]
    assert n0.prefix == (0,) and n0.level == 1 and n0.occupancy == 2
    assert abs(n0.mean_residual - 0.3) < 1e-6        # mean of 0.2, 0.4
    assert set(n0.children.keys()) == {0, 1}
    leaf = node_at(root, (0, 1))
    assert leaf.occupancy == 1 and leaf.item_indices == [1]


def test_node_at_missing_returns_none():
    codes = np.array([[0, 0]], dtype=np.int64)
    root = build_tree(codes, np.array([0.1], dtype=np.float32))
    assert node_at(root, (9, 9)) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_tree_build.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'gyo.tree.build'`

- [ ] **Step 3: Write minimal implementation**

`src/gyo/tree/build.py`:

```python
from dataclasses import dataclass, field
import numpy as np


@dataclass
class TreeNode:
    prefix: tuple
    level: int
    occupancy: int = 0
    mean_residual: float = 0.0
    item_indices: list = field(default_factory=list)
    children: dict = field(default_factory=dict)


def build_tree(codes: np.ndarray, final_residual: np.ndarray, labels=None) -> TreeNode:
    codes = np.asarray(codes, dtype=np.int64)
    n, num_levels = codes.shape
    root = TreeNode(prefix=(), level=0)
    root.item_indices = list(range(n))
    for i in range(n):
        node = root
        for lvl in range(num_levels):
            c = int(codes[i, lvl])
            child = node.children.get(c)
            if child is None:
                child = TreeNode(prefix=node.prefix + (c,), level=lvl + 1)
                node.children[c] = child
            child.item_indices.append(i)
            node = child
    _finalize(root, final_residual)
    return root


def _finalize(node: TreeNode, final_residual: np.ndarray) -> None:
    node.occupancy = len(node.item_indices)
    if node.occupancy:
        node.mean_residual = float(np.mean(final_residual[node.item_indices]))
    for child in node.children.values():
        _finalize(child, final_residual)


def node_at(root: TreeNode, prefix: tuple):
    node = root
    for c in prefix:
        node = node.children.get(int(c))
        if node is None:
            return None
    return node
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_tree_build.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/gyo/tree/__init__.py src/gyo/tree/build.py tests/test_tree_build.py
git commit -m "feat: code-prefix tree aggregation (occupancy, mean residual)"
```

---

## Task 8: Signals (neutral per-node stats — no verdict)

**Files:**
- Create: `src/gyo/tree/signals.py`
- Test: `tests/test_signals.py`

**Interfaces:**
- Consumes: `TreeNode` from Task 7.
- Produces:
  - `@dataclass NodeStat: prefix: tuple; level: int; occupancy: int; mean_residual: float; size_norm: float; residual_norm: float; is_dead: bool; purity: float | None`
  - `node_stats(root: TreeNode, labels: list[str] | None = None) -> list[NodeStat]` — flattens every node; `size_norm` = occupancy / max_occupancy; `residual_norm` = min-max scaled mean_residual across nodes (0..1, frozen when all equal → 0.0); `purity` = fraction of the node's most common label (None if `labels` is None); `is_dead` = occupancy == 0 (never produced by build_tree, but kept for codebook-coverage checks).
  - `dead_codeword_counts(codes: np.ndarray, codebook_size: int) -> list[int]` — per level, how many of the `K` codewords were never used.
  - **No function returns a hypothesis, action, or "data vs model" label.** This is enforced by test.

- [ ] **Step 1: Write the failing test**

`tests/test_signals.py`:

```python
import numpy as np
from gyo.tree.build import build_tree
from gyo.tree.signals import node_stats, dead_codeword_counts, NodeStat


def test_node_stats_normalizes_size_and_residual():
    codes = np.array([[0, 0], [0, 1], [1, 0]], dtype=np.int64)
    final_res = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    root = build_tree(codes, final_res)
    stats = node_stats(root)
    by_prefix = {s.prefix: s for s in stats}
    # root has max occupancy -> size_norm 1.0
    assert by_prefix[()].size_norm == 1.0
    # residual normalized 0..1 across nodes
    assert min(s.residual_norm for s in stats) == 0.0
    assert max(s.residual_norm for s in stats) == 1.0
    assert all(isinstance(s, NodeStat) for s in stats)


def test_purity_with_labels():
    codes = np.array([[0], [0], [0]], dtype=np.int64)
    root = build_tree(codes, np.array([0.1, 0.1, 0.1], dtype=np.float32))
    stats = node_stats(root, labels=["cat", "cat", "dog"])
    leaf = next(s for s in stats if s.prefix == (0,))
    assert abs(leaf.purity - 2 / 3) < 1e-6


def test_dead_codeword_counts():
    codes = np.array([[0, 0], [0, 1]], dtype=np.int64)  # level0 uses {0}, level1 uses {0,1}
    dead = dead_codeword_counts(codes, codebook_size=4)
    assert dead == [3, 2]  # level0: 4-1=3 dead ; level1: 4-2=2 dead


def test_signals_module_has_no_verdict_api():
    import gyo.tree.signals as s
    names = dir(s)
    for banned in ("hypothesis", "recommend", "action", "verdict", "route", "diagnose"):
        assert not any(banned in n.lower() for n in names), f"verdict-like API leaked: {banned}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_signals.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'gyo.tree.signals'`

- [ ] **Step 3: Write minimal implementation**

`src/gyo/tree/signals.py`:

```python
from collections import Counter
from dataclasses import dataclass
import numpy as np

from gyo.tree.build import TreeNode


@dataclass
class NodeStat:
    prefix: tuple
    level: int
    occupancy: int
    mean_residual: float
    size_norm: float
    residual_norm: float
    is_dead: bool
    purity: float | None


def _flatten(node: TreeNode, acc: list) -> None:
    acc.append(node)
    for child in node.children.values():
        _flatten(child, acc)


def node_stats(root: TreeNode, labels=None) -> list[NodeStat]:
    nodes: list[TreeNode] = []
    _flatten(root, nodes)
    max_occ = max((n.occupancy for n in nodes), default=1) or 1
    residuals = [n.mean_residual for n in nodes]
    lo, hi = min(residuals), max(residuals)
    span = (hi - lo) or 1.0
    out = []
    for n in nodes:
        purity = None
        if labels is not None and n.occupancy:
            counts = Counter(labels[i] for i in n.item_indices)
            purity = counts.most_common(1)[0][1] / n.occupancy
        out.append(NodeStat(
            prefix=n.prefix, level=n.level, occupancy=n.occupancy,
            mean_residual=n.mean_residual,
            size_norm=n.occupancy / max_occ,
            residual_norm=0.0 if hi == lo else (n.mean_residual - lo) / span,
            is_dead=n.occupancy == 0, purity=purity))
    return out


def dead_codeword_counts(codes: np.ndarray, codebook_size: int) -> list[int]:
    codes = np.asarray(codes, dtype=np.int64)
    return [codebook_size - len(np.unique(codes[:, lvl])) for lvl in range(codes.shape[1])]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_signals.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/gyo/tree/signals.py tests/test_signals.py
git commit -m "feat: neutral per-node signals (size/residual norm, purity, dead codewords)"
```

---

## Task 9: CLI wiring (prep-data | extract | fit-rq | encode)

**Files:**
- Create: `src/gyo/cli.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `prepare_fashion_mnist` (T6), `DummyEmbedder`/`MobileCLIPEmbedder` (T4/T5), `ResidualQuantizer` (T2/T3), `store` (T1).
- Produces a Typer `app` with commands writing the runtime artifact layout:
  - `prep-data --data-dir run --n 10000`
  - `extract --data-dir run --embedder mobileclip|dummy` → `embeddings.npy`, `meta.parquet`
  - `fit-rq --data-dir run --levels 3 --codebook-size 256` → `codebooks/v1/`
  - `encode --data-dir run` → `codes.parquet` (columns `idx, c_0..c_{L-1}, j, r_0..r_{L-1}, final_residual`)
  - `serve --data-dir run --port 8000` (wired in Task 10)

- [ ] **Step 1: Write the failing test**

`tests/test_cli.py`:

```python
import numpy as np
import pandas as pd
from PIL import Image
from typer.testing import CliRunner
from gyo.cli import app

runner = CliRunner()


def _seed_images(data_dir, n=6):
    img_dir = data_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        Image.new("L", (28, 28), color=i * 9).save(img_dir / f"{i:06d}.png")
    pd.DataFrame({"path": [f"{i:06d}.png" for i in range(n)],
                  "label": ["A" if i % 2 else "B" for i in range(n)]}).to_csv(
        data_dir / "labels.csv", index=False)


def test_extract_fit_encode_pipeline(tmp_path):
    _seed_images(tmp_path, 6)
    r1 = runner.invoke(app, ["extract", "--data-dir", str(tmp_path), "--embedder", "dummy"])
    assert r1.exit_code == 0, r1.output
    assert (tmp_path / "embeddings.npy").exists()
    assert (tmp_path / "meta.parquet").exists()

    r2 = runner.invoke(app, ["fit-rq", "--data-dir", str(tmp_path),
                             "--levels", "2", "--codebook-size", "3"])
    assert r2.exit_code == 0, r2.output
    assert (tmp_path / "codebooks" / "v1" / "config.json").exists()

    r3 = runner.invoke(app, ["encode", "--data-dir", str(tmp_path)])
    assert r3.exit_code == 0, r3.output
    codes = pd.read_parquet(tmp_path / "codes.parquet")
    assert {"idx", "c_0", "c_1", "j", "final_residual"} <= set(codes.columns)
    assert len(codes) == 6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cli.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'gyo.cli'`

- [ ] **Step 3: Write minimal implementation**

`src/gyo/cli.py`:

```python
from pathlib import Path
import pandas as pd
import typer

from gyo.io.store import save_embeddings, load_embeddings, save_table, load_table
from gyo.rq.quantizer import ResidualQuantizer

app = typer.Typer(help="RQ embedding-space inspector")


def _embedder(name: str):
    if name == "dummy":
        from gyo.embedders.base import DummyEmbedder
        return DummyEmbedder()
    from gyo.embedders.mobileclip import MobileCLIPEmbedder
    return MobileCLIPEmbedder()


@app.command("prep-data")
def prep_data(data_dir: str = "run", n: int = 10000, split: str = "train"):
    from gyo.data.fashion_mnist import prepare_fashion_mnist
    prepare_fashion_mnist(data_dir, n=n, split=split)
    typer.echo(f"prepared up to {n} images in {data_dir}/images")


@app.command()
def extract(data_dir: str = "run", embedder: str = "mobileclip"):
    emb, meta = _embedder(embedder).embed_folder(Path(data_dir) / "images")
    save_embeddings(Path(data_dir) / "embeddings.npy", emb)
    save_table(Path(data_dir) / "meta.parquet", meta)
    typer.echo(f"extracted {emb.shape[0]} embeddings dim={emb.shape[1]}")


@app.command("fit-rq")
def fit_rq(data_dir: str = "run", levels: int = 3, codebook_size: int = 256,
           iters: int = 10, seed: int = 0):
    emb = load_embeddings(Path(data_dir) / "embeddings.npy")
    rq = ResidualQuantizer(num_levels=levels, codebook_size=codebook_size, seed=seed)
    rq.fit(emb, iters=iters)
    rq.save(Path(data_dir) / "codebooks" / "v1")
    typer.echo(f"fit RQ L={levels} K={codebook_size} on {emb.shape[0]} points")


@app.command()
def encode(data_dir: str = "run"):
    emb = load_embeddings(Path(data_dir) / "embeddings.npy")
    rq = ResidualQuantizer.load(Path(data_dir) / "codebooks" / "v1")
    res = rq.encode(emb)
    cols = {"idx": range(len(emb))}
    for lvl in range(rq.num_levels):
        cols[f"c_{lvl}"] = res.codes[:, lvl]
    cols["j"] = res.tie_index
    for lvl in range(rq.num_levels):
        cols[f"r_{lvl}"] = res.residuals[:, lvl]
    cols["final_residual"] = res.final_residual
    save_table(Path(data_dir) / "codes.parquet", pd.DataFrame(cols))
    typer.echo(f"encoded {len(emb)} items")


@app.command()
def serve(data_dir: str = "run", port: int = 8000):
    import uvicorn
    from gyo.api.server import create_app
    uvicorn.run(create_app(data_dir), host="127.0.0.1", port=port)


if __name__ == "__main__":
    app()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_cli.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add src/gyo/cli.py tests/test_cli.py
git commit -m "feat: CLI (extract, fit-rq, encode) wiring with Typer"
```

---

## Task 10: API server + web UI

**Files:**
- Create: `src/gyo/api/__init__.py`, `src/gyo/api/server.py`, `src/gyo/web/index.html`, `src/gyo/web/app.js`, `src/gyo/web/style.css`
- Test: `tests/test_server.py`

**Interfaces:**
- Consumes: `load_table` (T1), `build_tree` (T7), `node_stats`/`dead_codeword_counts` (T8), the `codes.parquet`/`meta.parquet` artifacts (T9), `images/` (T6).
- Produces: `create_app(data_dir: str) -> FastAPI` with:
  - `GET /` → serves `web/index.html`
  - `GET /api/tree?level=L` → `{"nodes": [{"prefix": [...], "level": l, "occupancy": n, "mean_residual": r, "size_norm": s, "residual_norm": c, "purity": p|null}], "num_levels": L, "dead_codewords": [...]}` (nodes filtered to `level <= requested`)
  - `GET /api/node/{prefix}` (prefix = comma-joined ints, `root` for `()`) → `{"items": [{"idx": i, "path": "...", "label": "..."}], "occupancy": n}` (capped at 200 items)
  - `GET /thumb/{idx}` → the PNG bytes for item `idx`
  - Static legend text (the human-interpretation table) is rendered client-side from a constant in `app.js`, **not** computed per node.

- [ ] **Step 1: Write the failing test**

`tests/test_server.py`:

```python
import numpy as np
import pandas as pd
from PIL import Image
from fastapi.testclient import TestClient
from gyo.api.server import create_app


def _seed_run(data_dir):
    img_dir = data_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    for i in range(3):
        Image.new("L", (28, 28), color=i * 30).save(img_dir / f"{i:06d}.png")
    pd.DataFrame({"idx": [0, 1, 2],
                  "path": ["000000.png", "000001.png", "000002.png"],
                  "label": ["A", "A", "B"]}).to_parquet(data_dir / "meta.parquet", index=False)
    pd.DataFrame({"idx": [0, 1, 2], "c_0": [0, 0, 1], "c_1": [0, 1, 0],
                  "j": [0, 0, 0], "r_0": [1.0, 1.0, 1.0], "r_1": [0.5, 0.5, 0.5],
                  "final_residual": [0.1, 0.2, 0.9]}).to_parquet(
        data_dir / "codes.parquet", index=False)


def test_tree_endpoint(tmp_path):
    _seed_run(tmp_path)
    client = TestClient(create_app(str(tmp_path)))
    r = client.get("/api/tree?level=2")
    assert r.status_code == 200
    body = r.json()
    assert body["num_levels"] == 2
    prefixes = [tuple(n["prefix"]) for n in body["nodes"]]
    assert () in prefixes and (0,) in prefixes and (0, 1) in prefixes
    root = next(n for n in body["nodes"] if n["prefix"] == [])
    assert root["occupancy"] == 3


def test_node_items_and_thumb(tmp_path):
    _seed_run(tmp_path)
    client = TestClient(create_app(str(tmp_path)))
    r = client.get("/api/node/0")
    assert r.status_code == 200
    items = r.json()["items"]
    assert {it["idx"] for it in items} == {0, 1}
    t = client.get("/thumb/0")
    assert t.status_code == 200 and t.headers["content-type"].startswith("image/")


def test_root_serves_html(tmp_path):
    _seed_run(tmp_path)
    client = TestClient(create_app(str(tmp_path)))
    r = client.get("/")
    assert r.status_code == 200 and "<html" in r.text.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_server.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'gyo.api.server'`

- [ ] **Step 3: Write minimal implementation (server)**

`src/gyo/api/server.py`:

```python
from pathlib import Path
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse

from gyo.io.store import load_table
from gyo.tree.build import build_tree, node_at
from gyo.tree.signals import node_stats, dead_codeword_counts

WEB = Path(__file__).resolve().parent.parent / "web"


def create_app(data_dir: str) -> FastAPI:
    data_dir = Path(data_dir)
    app = FastAPI(title="gyo")

    def _load():
        codes_df = load_table(data_dir / "codes.parquet")
        meta_df = load_table(data_dir / "meta.parquet")
        level_cols = [c for c in codes_df.columns if c.startswith("c_")]
        codes = codes_df[sorted(level_cols)].to_numpy(np.int64)
        final_res = codes_df["final_residual"].to_numpy(np.float32)
        labels = meta_df["label"].astype(str).tolist()
        return codes, final_res, labels, meta_df

    @app.get("/", response_class=HTMLResponse)
    def index():
        return (WEB / "index.html").read_text()

    @app.get("/app.js")
    def appjs():
        return FileResponse(WEB / "app.js", media_type="application/javascript")

    @app.get("/style.css")
    def style():
        return FileResponse(WEB / "style.css", media_type="text/css")

    @app.get("/api/tree")
    def tree(level: int = 99):
        codes, final_res, labels, _ = _load()
        root = build_tree(codes, final_res, labels)
        stats = [s for s in node_stats(root, labels) if s.level <= level]
        return {
            "num_levels": codes.shape[1],
            "dead_codewords": dead_codeword_counts(codes, int(codes.max()) + 1),
            "nodes": [{
                "prefix": list(s.prefix), "level": s.level, "occupancy": s.occupancy,
                "mean_residual": s.mean_residual, "size_norm": s.size_norm,
                "residual_norm": s.residual_norm, "purity": s.purity,
            } for s in stats],
        }

    @app.get("/api/node/{prefix}")
    def node(prefix: str):
        codes, final_res, labels, meta_df = _load()
        root = build_tree(codes, final_res, labels)
        pfx = () if prefix == "root" else tuple(int(p) for p in prefix.split(","))
        target = node_at(root, pfx)
        if target is None:
            raise HTTPException(404, "prefix not found")
        items = [{"idx": int(i), "path": str(meta_df.loc[i, "path"]),
                  "label": str(meta_df.loc[i, "label"])} for i in target.item_indices[:200]]
        return {"items": items, "occupancy": target.occupancy}

    @app.get("/thumb/{idx}")
    def thumb(idx: int):
        _, _, _, meta_df = _load()
        if idx < 0 or idx >= len(meta_df):
            raise HTTPException(404, "idx out of range")
        path = data_dir / "images" / str(meta_df.loc[idx, "path"])
        if not path.exists():
            raise HTTPException(404, "image missing")
        return FileResponse(path, media_type="image/png")

    return app
```

- [ ] **Step 4: Run server tests to verify they pass**

Run: `uv run pytest tests/test_server.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Write the web UI**

`src/gyo/web/index.html`:

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
    <h1>gyo — embedding signal viewer</h1>
    <p class="disclaimer">Shows signals, does not judge. Size = occupancy, color = mean residual
      (hot = high). Any "data vs embedder" reading is your hypothesis, not the tool's.</p>
    <label>Level: <input id="level" type="range" min="0" max="3" value="1" /></label>
    <span id="levelVal">1</span>
    <button id="legendBtn">Legend (how you might read it)</button>
  </header>
  <main>
    <svg id="tree" width="900" height="560"></svg>
    <aside id="panel"><p>Click a node to inspect its bucket.</p></aside>
  </main>
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

`src/gyo/web/style.css`:

```css
body { font-family: system-ui, sans-serif; margin: 0; color: #1a1a1a; }
header { padding: 12px 18px; border-bottom: 1px solid #ddd; }
.disclaimer { color: #555; font-size: 13px; max-width: 760px; }
main { display: flex; }
#tree { flex: 1; }
#panel { width: 320px; border-left: 1px solid #ddd; padding: 12px; overflow-y: auto; height: 560px; }
#panel img { width: 48px; height: 48px; margin: 2px; image-rendering: pixelated; border: 1px solid #ccc; }
.node circle { stroke: #333; stroke-width: 1px; cursor: pointer; }
.node text { font-size: 10px; }
```

`src/gyo/web/app.js`:

```javascript
const svg = d3.select("#tree");
const panel = d3.select("#panel");
const levelInput = document.getElementById("level");
const levelVal = document.getElementById("levelVal");
document.getElementById("legendBtn").onclick = () => document.getElementById("legend").showModal();

// hot = high residual (red), cool = low (blue)
const color = d3.scaleSequential(d3.interpolateRdYlBu).domain([1, 0]);

async function loadTree(level) {
  const res = await fetch(`/api/tree?level=${level}`);
  const data = await res.json();
  levelInput.max = data.num_levels;
  render(data, level);
}

function render(data, level) {
  svg.selectAll("*").remove();
  const root = { prefix: [], children: [] };
  const byPrefix = new Map([["", root]]);
  for (const n of data.nodes) byPrefix.set(n.prefix.join(","), { ...n, children: [] });
  for (const n of data.nodes) {
    if (n.prefix.length === 0) continue;
    const parentKey = n.prefix.slice(0, -1).join(",");
    const parent = byPrefix.get(parentKey);
    if (parent) parent.children.push(byPrefix.get(n.prefix.join(",")));
  }
  const hierarchy = d3.hierarchy(byPrefix.get(""));
  const layout = d3.tree().size([540, 820]);
  layout(hierarchy);
  const g = svg.append("g").attr("transform", "translate(40,10)");
  g.selectAll("line.link").data(hierarchy.links()).join("line")
    .attr("class", "link").attr("stroke", "#bbb")
    .attr("x1", d => d.source.x).attr("y1", d => d.source.y)
    .attr("x2", d => d.target.x).attr("y2", d => d.target.y);
  const node = g.selectAll("g.node").data(hierarchy.descendants()).join("g")
    .attr("class", "node").attr("transform", d => `translate(${d.x},${d.y})`);
  node.append("circle")
    .attr("r", d => 4 + 22 * (d.data.size_norm || 1))
    .attr("fill", d => d.data.residual_norm == null ? "#ccc" : color(d.data.residual_norm))
    .on("click", (_e, d) => inspect(d.data.prefix));
  node.append("text").attr("dy", -10).attr("text-anchor", "middle")
    .text(d => d.data.occupancy != null ? d.data.occupancy : "");
}

async function inspect(prefix) {
  const key = prefix.length ? prefix.join(",") : "root";
  const res = await fetch(`/api/node/${key}`);
  const data = await res.json();
  panel.html(`<h3>Bucket [${prefix.join(",") || "root"}] — ${data.occupancy} items</h3>`);
  const box = panel.append("div");
  for (const it of data.items) {
    box.append("img").attr("src", `/thumb/${it.idx}`).attr("title", `${it.idx} ${it.label}`);
  }
}

levelInput.oninput = () => { levelVal.textContent = levelInput.value; loadTree(+levelInput.value); };
loadTree(+levelInput.value);
```

- [ ] **Step 6: Manually verify the UI renders**

```bash
cd /home/red/repos/gyo
# seed a tiny run with the dummy embedder so no model download is needed
uv run python -c "
from pathlib import Path
from PIL import Image
import pandas as pd
d = Path('run'); (d / 'images').mkdir(parents=True, exist_ok=True)
rows = []
for i in range(60):
    fn = f'{i:06d}.png'
    Image.new('L', (28, 28), color=(i * 4) % 255).save(d / 'images' / fn)
    rows.append({'path': fn, 'label': ['A', 'B', 'C'][i % 3]})
pd.DataFrame(rows).to_csv(d / 'labels.csv', index=False)
"
uv run gyo extract --data-dir run --embedder dummy
uv run gyo fit-rq --data-dir run --levels 3 --codebook-size 8 --iters 15
uv run gyo encode --data-dir run
uv run gyo serve --data-dir run --port 8000 &
sleep 3 && curl -s localhost:8000/api/tree?level=2 | head -c 300 ; echo
kill %1
```

Expected: the curl prints JSON with `nodes` and `num_levels`. (Open `http://localhost:8000` in a browser to see the tree: circles sized by occupancy, colored by residual; clicking shows thumbnails.)

- [ ] **Step 7: Commit**

```bash
git add src/gyo/api tests/test_server.py src/gyo/web
git commit -m "feat: FastAPI tree/node/thumb endpoints + D3 tree UI (signals, no verdict)"
```

---

## Task 11: End-to-end verification on Fashion-MNIST

**Files:**
- Create: `tests/test_e2e.py`

**Interfaces:**
- Consumes: the full CLI pipeline (T6–T10).
- Produces: one end-to-end test that runs prep-data (injected fake dataset to avoid a download in CI) → extract (dummy) → fit-rq → encode → tree endpoint, asserting the artifacts and tree are coherent. A separate documented manual command exercises the real MobileCLIP path on a 2000-image Fashion-MNIST subsample.

- [ ] **Step 1: Write the end-to-end test**

`tests/test_e2e.py`:

```python
import numpy as np
import pandas as pd
from PIL import Image
from typer.testing import CliRunner
from fastapi.testclient import TestClient
from gyo.cli import app
from gyo.data.fashion_mnist import prepare_fashion_mnist
from gyo.api.server import create_app

runner = CliRunner()


def _fake_fmnist(n=120):
    return [(Image.new("L", (28, 28), color=(i * 7) % 255), i % 10) for i in range(n)]


def test_full_pipeline_dummy(tmp_path):
    prepare_fashion_mnist(tmp_path, n=120, dataset=_fake_fmnist(120))
    assert (tmp_path / "labels.csv").exists()

    assert runner.invoke(app, ["extract", "--data-dir", str(tmp_path),
                               "--embedder", "dummy"]).exit_code == 0
    assert runner.invoke(app, ["fit-rq", "--data-dir", str(tmp_path), "--levels", "3",
                               "--codebook-size", "8", "--iters", "10"]).exit_code == 0
    assert runner.invoke(app, ["encode", "--data-dir", str(tmp_path)]).exit_code == 0

    codes = pd.read_parquet(tmp_path / "codes.parquet")
    assert len(codes) == 120
    assert {"c_0", "c_1", "c_2", "j", "final_residual"} <= set(codes.columns)

    client = TestClient(create_app(str(tmp_path)))
    body = client.get("/api/tree?level=3").json()
    assert body["num_levels"] == 3
    root = next(n for n in body["nodes"] if n["prefix"] == [])
    assert root["occupancy"] == 120
    # every reported node has the neutral signal fields and NO verdict field
    for n in body["nodes"]:
        assert set(n.keys()) == {"prefix", "level", "occupancy", "mean_residual",
                                 "size_norm", "residual_norm", "purity"}
```

- [ ] **Step 2: Run the whole suite**

Run: `uv run pytest -v -m "not slow"`
Expected: ALL PASS (every non-slow test across the suite).

- [ ] **Step 3: Run the real MobileCLIP path once (manual verification)**

```bash
cd /home/red/repos/gyo
uv run gyo prep-data --data-dir run_fmnist --n 2000
uv run gyo extract --data-dir run_fmnist --embedder mobileclip
uv run gyo fit-rq --data-dir run_fmnist --levels 3 --codebook-size 256 --iters 12
uv run gyo encode --data-dir run_fmnist
uv run gyo serve --data-dir run_fmnist --port 8000
# open http://localhost:8000 — confirm tree renders, nodes sized by occupancy,
# colored by residual; clicking a node shows Fashion-MNIST thumbnails.
```

Expected: pipeline completes on CPU in minutes; tree is navigable; thumbnails load. This satisfies the spec's end-to-end verification (tree renders with size=occupancy, color=residual; buckets browsable).

- [ ] **Step 4: Commit**

```bash
git add tests/test_e2e.py
git commit -m "test: end-to-end pipeline (prep->extract->fit->encode->tree) verification"
```

---

## Self-Review

**Spec coverage:**
- Embedding extraction (MobileCLIP2-S0, eval + reparameterize, L2-norm, swappable) → Tasks 4, 5. ✓
- RQ: EMA training, sum reconstruction, semantic IDs + per-level/final residuals, tie index `j`, save/load, linear projection OFF by default → Tasks 2, 3. ✓
- Codebook freeze (save/load) present; incremental expansion explicitly deferred per spec → Task 2 save/load. ✓
- Code-prefix tree (occupancy, mean residual, children) → Task 7. ✓
- Neutral signals, no verdict, legend as static help → Task 8 (+ enforced by `test_signals_module_has_no_verdict_api`) and Task 10 UI legend. ✓
- Level slider, size=occupancy, color=residual, click-to-inspect thumbnails → Task 10. ✓
- Fashion-MNIST test set, 10k default subsample → Task 6 + CLI default. ✓
- Python 3.12 pin → Task 1. ✓
- End-to-end verification on Fashion-MNIST → Task 11. ✓

**Placeholder scan:** No TBD/TODO; every code step contains full code. ✓

**Type consistency:** `ResidualQuantizer` (num_levels/codebook_size/dim/seed), `EncodeResult` (codes/residuals/final_residual/tie_index), `TreeNode` (prefix/level/occupancy/mean_residual/item_indices/children), `NodeStat` fields, and the `codes.parquet` columns (`c_*`, `j`, `r_*`, `final_residual`) are used identically across Tasks 2–11. ✓

**Note for the implementer:** the `dim` kwarg in `ResidualQuantizer.__init__` is required by Task 2 tests (codebooks set manually) and set by `fit`/`load` otherwise — both paths covered.
