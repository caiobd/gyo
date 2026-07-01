import json
from dataclasses import dataclass
from pathlib import Path
import numpy as np


@dataclass
class EncodeResult:
    codes: np.ndarray  # (N, L) int64
    residuals: np.ndarray  # (N, L) float32: ||r_i|| before subtracting level i
    final_residual: np.ndarray  # (N,) float32: ||r_L||
    tie_index: np.ndarray  # (N,) int64: j within identical-tuple leaves


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
    def __init__(
        self, num_levels=3, codebook_size=256, dim=None, proj_dim=None, seed=0
    ):
        self.num_levels = num_levels
        self.codebook_size = codebook_size
        self.dim = dim
        self.proj_dim = proj_dim  # reserved; linear projection OFF by default
        self.seed = seed
        self.codebooks: list[np.ndarray] = []

    def fit(
        self, x: np.ndarray, iters: int = 10, ema_decay: float = 0.99
    ) -> "ResidualQuantizer":
        x = np.asarray(x, dtype=np.float32)
        self.dim = x.shape[1]
        rng = np.random.default_rng(self.seed)
        self.codebooks = []
        r = x.copy()
        for _level in range(self.num_levels):
            init_idx = rng.choice(
                len(r), size=self.codebook_size, replace=len(r) < self.codebook_size
            )
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
        (d / "config.json").write_text(
            json.dumps(
                {
                    "num_levels": self.num_levels,
                    "codebook_size": self.codebook_size,
                    "dim": self.dim,
                    "proj_dim": self.proj_dim,
                    "seed": self.seed,
                }
            )
        )

    @classmethod
    def load(cls, directory) -> "ResidualQuantizer":
        d = Path(directory)
        cfg = json.loads((d / "config.json").read_text())
        rq = cls(**cfg)
        rq.codebooks = [np.load(d / f"level_{i}.npy") for i in range(cfg["num_levels"])]
        return rq
