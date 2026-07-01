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
    codes = np.array(
        [[0, 0], [0, 1]], dtype=np.int64
    )  # level0 uses {0}, level1 uses {0,1}
    dead = dead_codeword_counts(codes, codebook_size=4)
    assert dead == [3, 2]  # level0: 4-1=3 dead ; level1: 4-2=2 dead


def test_signals_module_has_no_verdict_api():
    import gyo.tree.signals as s

    names = dir(s)
    for banned in ("hypothesis", "recommend", "action", "verdict", "route", "diagnose"):
        assert not any(banned in n.lower() for n in names), (
            f"verdict-like API leaked: {banned}"
        )
