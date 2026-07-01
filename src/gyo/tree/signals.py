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
        out.append(
            NodeStat(
                prefix=n.prefix,
                level=n.level,
                occupancy=n.occupancy,
                mean_residual=n.mean_residual,
                size_norm=n.occupancy / max_occ,
                residual_norm=0.0 if hi == lo else (n.mean_residual - lo) / span,
                is_dead=n.occupancy == 0,
                purity=purity,
            )
        )
    return out


def dead_codeword_counts(codes: np.ndarray, codebook_size: int) -> list[int]:
    codes = np.asarray(codes, dtype=np.int64)
    return [
        codebook_size - len(np.unique(codes[:, lvl])) for lvl in range(codes.shape[1])
    ]
