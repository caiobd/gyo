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


def test_fit_reduces_residual_and_sets_dim():
    rng = np.random.default_rng(0)
    centers = np.array([[3, 3], [-3, -3], [3, -3], [-3, 3]], dtype=np.float32)
    x = np.repeat(centers, 50, axis=0) + rng.normal(0, 0.1, (200, 2)).astype(np.float32)
    rq = ResidualQuantizer(num_levels=2, codebook_size=4, seed=0)
    rq.fit(x, iters=25)
    assert rq.dim == 2
    assert len(rq.codebooks) == 2
    res = rq.encode(x)
    assert res.final_residual.mean() < 0.5
    assert res.residuals[:, 0].mean() >= res.residuals[:, 1].mean()


def test_fit_is_deterministic_with_seed():
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1, (120, 3)).astype(np.float32)
    a = ResidualQuantizer(num_levels=2, codebook_size=8, seed=7).fit(x, iters=10)
    b = ResidualQuantizer(num_levels=2, codebook_size=8, seed=7).fit(x, iters=10)
    for ca, cb in zip(a.codebooks, b.codebooks):
        np.testing.assert_allclose(ca, cb, atol=1e-5)
