"""Deterministic geometry helpers for Semantic Atlas layouts."""

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class MDSResult:
    """A two-dimensional MDS layout and its normalized stress."""

    positions: NDArray[np.float64]
    stress: float


def prefix_vector(prefix: Sequence[int], codebooks: Sequence[ArrayLike]) -> NDArray[np.float64]:
    """Reconstruct a prefix vector by summing its path's codewords."""

    if len(codebooks) == 0:
        raise ValueError("codebooks must not be empty")

    first = np.asarray(codebooks[0], dtype=np.float64)
    if first.ndim != 2:
        raise ValueError("each codebook must be a two-dimensional array")
    result = np.zeros(first.shape[1], dtype=np.float64)
    if len(prefix) > len(codebooks):
        raise ValueError("prefix is deeper than the available codebooks")

    for depth, codeword_index in enumerate(prefix):
        codebook = np.asarray(codebooks[depth], dtype=np.float64)
        if codebook.ndim != 2 or codebook.shape[1] != result.size:
            raise ValueError("codebooks must be two-dimensional with matching vector sizes")
        result += codebook[codeword_index]
    return result


def sibling_distance_matrix(
    prefixes: Sequence[Sequence[int]], codebooks: Sequence[ArrayLike]
) -> NDArray[np.float64]:
    """Return pairwise Euclidean distances between reconstructed prefixes."""

    vectors = np.asarray(
        [prefix_vector(prefix, codebooks) for prefix in prefixes], dtype=np.float64
    )
    if not prefixes:
        return np.zeros((0, 0), dtype=np.float64)
    differences = vectors[:, None, :] - vectors[None, :, :]
    return np.linalg.norm(differences, axis=2)


def _classical_mds(distances: NDArray[np.float64]) -> NDArray[np.float64]:
    count = distances.shape[0]
    centering = np.eye(count) - np.full((count, count), 1.0 / count)
    gram = -0.5 * centering @ np.square(distances) @ centering
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1][:2]
    positive = np.maximum(eigenvalues[order], 0.0)
    positions = eigenvectors[:, order] * np.sqrt(positive)
    if positions.shape[1] < 2:
        positions = np.pad(positions, ((0, 0), (0, 2 - positions.shape[1])))

    # Resolve the arbitrary eigenvector signs so serialized layouts are stable.
    for axis in range(2):
        pivot = int(np.argmax(np.abs(positions[:, axis])))
        if positions[pivot, axis] < 0:
            positions[:, axis] *= -1
    return positions


def _pairwise_distances(positions: NDArray[np.float64]) -> NDArray[np.float64]:
    differences = positions[:, None, :] - positions[None, :, :]
    return np.linalg.norm(differences, axis=2)


def metric_mds(
    distances: ArrayLike, max_iter: int = 300, eps: float = 1e-7
) -> MDSResult:
    """Fit a deterministic two-dimensional metric MDS layout using SMACOF."""

    target = np.asarray(distances, dtype=np.float64)
    if target.ndim != 2 or target.shape[0] != target.shape[1]:
        raise ValueError("distances must be a square matrix")
    if not np.isfinite(target).all():
        raise ValueError("distances must contain only finite values")
    if np.any(target < 0):
        raise ValueError("distances must be non-negative")
    if not np.allclose(target, target.T):
        raise ValueError("distances must be symmetric")

    count = target.shape[0]
    if count <= 1:
        return MDSResult(np.zeros((count, 2), dtype=np.float64), 0.0)

    positions = _classical_mds(target)
    denominator = float(np.sum(np.triu(np.square(target), k=1)))
    previous_stress = np.inf

    for _ in range(max(0, max_iter)):
        fitted = _pairwise_distances(positions)
        ratios = np.divide(target, fitted, out=np.zeros_like(target), where=fitted > 0)
        b_matrix = -ratios
        np.fill_diagonal(b_matrix, 0.0)
        b_matrix[np.diag_indices(count)] = -b_matrix.sum(axis=1)
        updated = b_matrix @ positions / count

        updated_distances = _pairwise_distances(updated)
        raw_stress = float(
            np.sum(np.triu(np.square(target - updated_distances), k=1))
        )
        positions = updated
        if np.isfinite(previous_stress) and previous_stress - raw_stress <= eps * max(
            previous_stress, 1.0
        ):
            break
        previous_stress = raw_stress

    # A zero target distance denotes the same point. Collapse each connected
    # duplicate component exactly, eliminating harmless eigensolver roundoff.
    unseen = set(range(count))
    while unseen:
        component = {unseen.pop()}
        frontier = list(component)
        while frontier:
            current = frontier.pop()
            neighbors = {index for index in unseen if target[current, index] == 0.0}
            unseen.difference_update(neighbors)
            component.update(neighbors)
            frontier.extend(neighbors)
        indices = list(component)
        positions[indices] = positions[indices].mean(axis=0)

    positions -= positions.mean(axis=0)
    radius = float(np.max(np.linalg.norm(positions, axis=1)))
    if radius > 1.0:
        positions /= radius

    fitted = _pairwise_distances(positions)
    if denominator == 0.0:
        stress = 0.0
    else:
        # Optimize away global scale because output normalization is presentational.
        fitted_upper = fitted[np.triu_indices(count, k=1)]
        target_upper = target[np.triu_indices(count, k=1)]
        fitted_norm = float(fitted_upper @ fitted_upper)
        scale = float((target_upper @ fitted_upper) / fitted_norm) if fitted_norm else 0.0
        stress = float(np.sqrt(np.sum(np.square(target_upper - scale * fitted_upper)) / denominator))
    return MDSResult(positions.astype(np.float64, copy=False), stress)
