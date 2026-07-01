import numpy as np
import pytest

from gyo.atlas import RankedSamples, ranked_samples


def test_ranks_nearest_representatives_and_farthest_outliers():
    embeddings = np.array([[0.0, 0.0], [0.1, 0.0], [2.0, 0.0], [9.0, 9.0]])

    result = ranked_samples(embeddings, [0, 1, 2], [0.0, 0.0], limit=2)

    assert result == RankedSamples(representative=(0, 1), outliers=(2, 1))


def test_single_member_is_not_padded_or_leaked_from_another_group():
    result = ranked_samples([[0, 0], [1, 0], [2, 0]], [2], [0, 0], limit=24)

    assert result.representative == (2,)
    assert result.outliers == (2,)


def test_empty_group_returns_empty_tuples():
    assert ranked_samples([[0, 0]], [], [0, 0]) == RankedSamples((), ())


def test_ties_preserve_member_order_for_both_rankings():
    embeddings = [[-1, 0], [1, 0], [0, -1], [0, 1]]

    result = ranked_samples(embeddings, [2, 0, 3, 1], [0, 0])

    assert result.representative == (2, 0, 3, 1)
    assert result.outliers == (2, 0, 3, 1)


def test_repeated_member_indices_do_not_duplicate_samples():
    result = ranked_samples([[0], [1]], [1, 1, 0], [0], limit=3)

    assert result.representative == (0, 1)
    assert result.outliers == (1, 0)


def test_ranked_samples_collections_are_immutable():
    result = ranked_samples([[0]], [0], [0])

    with pytest.raises(AttributeError):
        result.representative.append(1)


def test_non_member_nan_does_not_affect_ranking():
    result = ranked_samples([[0.0], [np.nan], [2.0]], [2, 0], [0.0])

    assert result == RankedSamples(representative=(0, 2), outliers=(2, 0))


def test_member_nan_is_rejected():
    with pytest.raises(
        ValueError, match="member embeddings must contain only finite values"
    ):
        ranked_samples([[0.0], [np.nan]], [1], [0.0])


@pytest.mark.parametrize(
    ("embeddings", "center", "message"),
    [
        ([[1 + 2j]], [0], "embeddings must contain only real values"),
        ([[0]], [1 + 2j], "center must contain only real values"),
    ],
)
def test_rejects_complex_embeddings_and_center(embeddings, center, message):
    with pytest.raises(ValueError, match=message):
        ranked_samples(embeddings, [0], center)


@pytest.mark.parametrize(
    ("embeddings", "members", "center", "limit", "message"),
    [
        ([0, 1], [0], [0], 1, "embeddings must be a two-dimensional array"),
        ([[0, 1]], [0], [0], 1, "center dimension must match embeddings"),
        ([[0]], [-1], [0], 1, "member indices must be within embeddings bounds"),
        ([[0]], [1], [0], 1, "member indices must be within embeddings bounds"),
        ([[0]], [0], [0], -1, "limit must be a non-negative integer"),
        ([[0]], [0], [0], 1.5, "limit must be a non-negative integer"),
        ([[0]], [0], [0], True, "limit must be a non-negative integer"),
    ],
)
def test_rejects_invalid_shapes_indices_and_limits(
    embeddings, members, center, limit, message
):
    with pytest.raises(ValueError, match=message):
        ranked_samples(embeddings, members, center, limit)


@pytest.mark.parametrize(
    ("embeddings", "center", "message"),
    [
        ([[np.nan]], [0], "member embeddings must contain only finite values"),
        ([[0]], [np.inf], "center must contain only finite values"),
    ],
)
def test_rejects_non_finite_values(embeddings, center, message):
    with pytest.raises(ValueError, match=message):
        ranked_samples(embeddings, [0], center)


@pytest.mark.parametrize("members", [[0.5], [True]])
def test_rejects_non_integer_member_indices(members):
    with pytest.raises(ValueError, match="member indices must be integers"):
        ranked_samples([[0]], members, [0])
