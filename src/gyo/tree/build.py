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
