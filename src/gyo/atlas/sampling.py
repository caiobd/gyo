"""Representative and outlier sampling for Semantic Atlas groups."""

from dataclasses import dataclass
from numbers import Integral
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike


@dataclass(frozen=True)
class RankedSamples:
    """Ranked member indices for displaying a group."""

    representative: tuple[int, ...]
    outliers: tuple[int, ...]


def ranked_samples(
    embeddings: ArrayLike,
    member_indices: Sequence[int],
    center: ArrayLike,
    limit: int = 24,
) -> RankedSamples:
    """Rank unique group members by their distance from ``center``.

    Duplicate member indices keep their first occurrence. Distance ties preserve
    ``member_indices`` order for both representative samples and outliers.
    """

    if isinstance(limit, bool) or not isinstance(limit, Integral) or limit < 0:
        raise ValueError("limit must be a non-negative integer")

    vectors = np.asarray(embeddings)
    if vectors.ndim != 2:
        raise ValueError("embeddings must be a two-dimensional array")
    if np.iscomplexobj(vectors):
        raise ValueError("embeddings must contain only real values")

    centroid = np.asarray(center)
    if centroid.ndim != 1 or centroid.shape[0] != vectors.shape[1]:
        raise ValueError("center dimension must match embeddings")
    if np.iscomplexobj(centroid):
        raise ValueError("center must contain only real values")
    centroid = np.asarray(centroid, dtype=np.float64)
    if not np.isfinite(centroid).all():
        raise ValueError("center must contain only finite values")

    members: list[int] = []
    seen: set[int] = set()
    for index in member_indices:
        if isinstance(index, bool) or not isinstance(index, Integral):
            raise ValueError("member indices must be integers")
        normalized = int(index)
        if normalized < 0 or normalized >= vectors.shape[0]:
            raise ValueError("member indices must be within embeddings bounds")
        if normalized not in seen:
            seen.add(normalized)
            members.append(normalized)

    if not members or limit == 0:
        return RankedSamples((), ())

    member_vectors = np.asarray(vectors[members], dtype=np.float64)
    if not np.isfinite(member_vectors).all():
        raise ValueError("member embeddings must contain only finite values")
    distances = np.linalg.norm(member_vectors - centroid, axis=1)
    if not np.isfinite(distances).all():
        raise ValueError("member distances must contain only finite values")
    count = min(int(limit), len(members))
    representative_order = np.argsort(distances, kind="stable")[:count]
    outlier_order = np.argsort(-distances, kind="stable")[:count]
    representative = tuple(members[index] for index in representative_order)
    outliers = tuple(members[index] for index in outlier_order)
    return RankedSamples(representative, outliers)
