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
    assert abs(n0.mean_residual - 0.3) < 1e-6  # mean of 0.2, 0.4
    assert set(n0.children.keys()) == {0, 1}
    leaf = node_at(root, (0, 1))
    assert leaf.occupancy == 1 and leaf.item_indices == [1]


def test_node_at_missing_returns_none():
    codes = np.array([[0, 0]], dtype=np.int64)
    root = build_tree(codes, np.array([0.1], dtype=np.float32))
    assert node_at(root, (9, 9)) is None
