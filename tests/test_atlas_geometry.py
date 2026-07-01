import numpy as np
import pytest

from gyo.atlas import MDSResult, metric_mds, prefix_vector, sibling_distance_matrix


@pytest.fixture
def codebooks():
    return np.array(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[0.5, 0.0], [0.0, 0.5]],
        ]
    )


def test_prefix_vector_reconstructs_codeword_path_in_float64(codebooks):
    result = prefix_vector((0, 1), codebooks)

    np.testing.assert_allclose(result, [1.0, 0.5])
    assert result.dtype == np.float64


def test_prefix_vector_requires_codebooks():
    with pytest.raises(ValueError, match="codebooks must not be empty"):
        prefix_vector((), [])


def test_sibling_distance_matrix_uses_reconstructed_vectors(codebooks):
    distances = sibling_distance_matrix([(0, 0), (0, 1)], codebooks)

    np.testing.assert_allclose(
        distances,
        [[0.0, np.sqrt(0.5)], [np.sqrt(0.5), 0.0]],
    )


def test_metric_mds_is_deterministic_and_fits_a_triangle():
    distances = np.array(
        [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]]
    )

    first = metric_mds(distances)
    second = metric_mds(distances)

    assert isinstance(first, MDSResult)
    np.testing.assert_array_equal(first.positions, second.positions)
    assert first.stress == second.stress
    np.testing.assert_allclose(first.positions.mean(axis=0), 0.0, atol=1e-12)
    assert np.linalg.norm(first.positions, axis=1).max() <= 1.0
    assert first.stress < 0.01


def test_metric_mds_normalizes_non_degenerate_layouts_and_is_scale_invariant():
    unit = np.array(
        [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]]
    )

    full_scale = metric_mds(unit)
    small_scale = metric_mds(unit * 0.1)

    assert np.linalg.norm(small_scale.positions, axis=1).max() == pytest.approx(1.0)
    full_distances = np.linalg.norm(
        full_scale.positions[:, None, :] - full_scale.positions[None, :, :], axis=2
    )
    small_distances = np.linalg.norm(
        small_scale.positions[:, None, :] - small_scale.positions[None, :, :], axis=2
    )
    np.testing.assert_allclose(small_distances, full_distances, atol=1e-12)
    assert small_scale.stress == pytest.approx(full_scale.stress, abs=1e-12)


@pytest.mark.parametrize("size", [0, 1])
def test_metric_mds_supports_zero_and_one_point(size):
    result = metric_mds(np.zeros((size, size)))

    assert result.positions.shape == (size, 2)
    assert result.stress == 0.0


def test_metric_mds_handles_duplicate_points_without_nan():
    distances = np.array(
        [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0]]
    )

    result = metric_mds(distances)

    assert np.isfinite(result.positions).all()
    assert np.isfinite(result.stress)
    np.testing.assert_allclose(result.positions[0], result.positions[1])


def test_metric_mds_rejects_non_square_matrices():
    with pytest.raises(ValueError, match="square"):
        metric_mds(np.zeros((2, 3)))


def test_metric_mds_rejects_nonzero_diagonal():
    with pytest.raises(ValueError, match="diagonal must be zero"):
        metric_mds([[0.01, 1.0], [1.0, 0.0]])


def test_metric_mds_rejects_material_asymmetry():
    with pytest.raises(ValueError, match="symmetric"):
        metric_mds([[0.0, 1.0], [1.01, 0.0]])


@pytest.mark.parametrize("max_iter", [-1, 1.5, True])
def test_metric_mds_rejects_invalid_max_iter(max_iter):
    with pytest.raises(ValueError, match="max_iter"):
        metric_mds(np.zeros((0, 0)), max_iter=max_iter)


@pytest.mark.parametrize("eps", [-1.0, np.nan, np.inf])
def test_metric_mds_rejects_invalid_eps(eps):
    with pytest.raises(ValueError, match="eps"):
        metric_mds(np.zeros((0, 0)), eps=eps)


def test_mds_result_is_immutable():
    result = MDSResult(np.zeros((1, 2)), 0.0)

    with pytest.raises(AttributeError):
        result.stress = 1.0
