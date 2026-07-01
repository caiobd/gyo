"""Representative and outlier sampling for Semantic Atlas groups."""

from dataclasses import dataclass
from numbers import Integral
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike


@dataclass(frozen=True)
class RankedSamples:
    """Ranked member indices for displaying a group."""

    representative: list[int]
    outliers: list[int]


def ranked_samples(
    embeddings: ArrayLike,
    member_indices: Sequence[int],
    center: ArrayLike,
    limit: int = 24,
) -> RankedSamples:
    """Rank group members by their distance from ``center``."""

    if isinstance(limit, bool) or not isinstance(limit, Integral) or limit < 0:
        raise ValueError("limit must be a non-negative integer")

    vectors = np.asarray(embeddings, dtype=np.float64)
    if vectors.ndim != 2:
        raise ValueError("embeddings must be a two-dimensional array")
    if not np.isfinite(vectors).all():
        raise ValueError("embeddings must contain only finite values")

    centroid = np.asarray(center, dtype=np.float64)
    if centroid.ndim != 1 or centroid.shape[0] != vectors.shape[1]:
        raise ValueError("center dimension must match embeddings")
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
        return RankedSamples([], [])

    distances = np.linalg.norm(vectors[members] - centroid, axis=1)
    if not np.isfinite(distances).all():
        raise ValueError("member distances must contain only finite values")
    ranked = list(zip(members, distances, strict=True))
    count = min(int(limit), len(ranked))
    representative = [
        index for index, _ in sorted(ranked, key=lambda item: item[1])[:count]
    ]
    outliers = [
        index
        for index, _ in sorted(ranked, key=lambda item: item[1], reverse=True)[:count]
    ]
    return RankedSamples(representative, outliers)
