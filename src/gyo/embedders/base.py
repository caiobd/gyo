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
        emb = (
            l2_normalize(np.array(vecs, dtype=np.float32))
            if vecs
            else np.zeros((0, self.dim), np.float32)
        )
        return emb, pd.DataFrame(rows, columns=["idx", "path", "label"])
