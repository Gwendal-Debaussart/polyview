"""Shared helpers for the multi-view spectral clustering estimators."""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import scipy.linalg as sla
from scipy.sparse.linalg import LinearOperator, eigsh
from sklearn.metrics.pairwise import pairwise_distances, pairwise_kernels

# Below this size a dense (subset) eigendecomposition is cheap and more robust
# than ARPACK; above it only the k leading eigenvectors are computed with eigsh.
_DENSE_EIGH_MAX_SAMPLES = 500


def compute_affinity(
    X: np.ndarray, affinity="rbf", gamma: Union[float, str, None] = None
) -> np.ndarray:
    """Compute a non-negative affinity (similarity) matrix for one view.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Data of one view (or an ``(n_samples, n_samples)`` affinity matrix
        when ``affinity='precomputed'``).
    affinity : str or callable, default='rbf'
        Kernel passed to :func:`sklearn.metrics.pairwise.pairwise_kernels`.
    gamma : float, 'median' or None, default=None
        Kernel coefficient. ``None`` uses the sklearn default of the kernel
        (``1 / n_features`` for 'rbf'). ``'median'`` sets the bandwidth to the
        median pairwise distance: ``gamma = 1 / (2 * median(||x_i - x_j||^2))``
        for 'rbf' and ``gamma = 1 / median(||x_i - x_j||_1)`` for 'laplacian'.

    Returns
    -------
    K : ndarray of shape (n_samples, n_samples)
    """
    if isinstance(gamma, str):
        if gamma != "median":
            raise ValueError(f"gamma must be a float, 'median' or None, got {gamma!r}.")
        if affinity == "rbf":
            dists, factor = pairwise_distances(X, metric="sqeuclidean"), 2.0
        elif affinity == "laplacian":
            dists, factor = pairwise_distances(X, metric="manhattan"), 1.0
        else:
            raise ValueError(
                "gamma='median' is only supported for affinity='rbf' or "
                f"'laplacian', got affinity={affinity!r}."
            )
        median = np.median(dists[np.triu_indices_from(dists, k=1)])
        gamma = 1.0 / (factor * median) if median > 0 else 1.0

    params = {} if gamma is None else {"gamma": gamma}
    K = pairwise_kernels(X, metric=affinity, **params)
    if np.any(K < 0):
        raise ValueError(
            f"The affinity {affinity!r} produced negative similarities; spectral "
            "clustering requires a non-negative affinity such as 'rbf'."
        )
    return (K + K.T) / 2.0


def normalized_similarity(S: np.ndarray) -> np.ndarray:
    """Return the symmetric normalised similarity ``D^{-1/2} S D^{-1/2}``.

    Its leading eigenvectors are the trailing eigenvectors of the normalised
    Laplacian ``I - D^{-1/2} S D^{-1/2}``.
    """
    d = S.sum(axis=1)
    d_inv_sqrt = 1.0 / np.sqrt(np.maximum(d, 1e-12 * max(np.abs(d).max(), 1e-300)))
    L = d_inv_sqrt[:, None] * S * d_inv_sqrt[None, :]
    return (L + L.T) / 2.0


def top_eigenvectors(
    A: np.ndarray,
    k: int,
    random_state: np.random.RandomState,
    W: Optional[np.ndarray] = None,
    coef: float = 0.0,
) -> np.ndarray:
    """Orthonormal eigenvectors of the ``k`` largest eigenvalues of ``A + coef * W W^T``.

    ``A`` must be symmetric. Only the ``k`` leading eigenpairs are computed:
    with a dense subset ``eigh`` for small problems and ARPACK (``eigsh``)
    otherwise, the low-rank term being applied implicitly.

    Returns
    -------
    U : ndarray of shape (n, k)
    """
    n = A.shape[0]
    if not 1 <= k <= n:
        raise ValueError(f"n_clusters must be in [1, n_samples={n}], got {k}.")
    has_low_rank = W is not None and coef != 0.0

    if n <= max(_DENSE_EIGH_MAX_SAMPLES, 5 * k):
        M = A + coef * (W @ W.T) if has_low_rank else A
        _, U = sla.eigh(M, subset_by_index=[n - k, n - 1])
        return U

    if has_low_rank:

        def matvec(x):
            return A @ x + coef * (W @ (W.T @ x))

        M = LinearOperator((n, n), matvec=matvec, matmat=matvec, dtype=A.dtype)
    else:
        M = A
    v0 = random_state.uniform(-1.0, 1.0, n)
    _, U = eigsh(M, k=k, which="LA", v0=v0)
    return U


def subspace_agreement(U: np.ndarray, V: np.ndarray) -> float:
    """``tr(U U^T V V^T) = ||U^T V||_F^2`` for orthonormal ``U`` and ``V``."""
    return float(np.linalg.norm(U.T @ V) ** 2)


def row_normalize(U: np.ndarray) -> np.ndarray:
    """Scale every row of ``U`` to unit Euclidean norm (zero rows are left as is)."""
    norms = np.linalg.norm(U, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return U / norms


def check_spectral_params(n_clusters, max_iter, n_samples: int) -> None:
    """Validate the parameters shared by the spectral estimators."""
    if (
        not isinstance(n_clusters, (int, np.integer))
        or not 1 <= n_clusters <= n_samples
    ):
        raise ValueError(
            f"n_clusters must be an integer in [1, n_samples={n_samples}], got {n_clusters!r}."
        )
    if not isinstance(max_iter, (int, np.integer)) or max_iter < 0:
        raise ValueError(f"max_iter must be a non-negative integer, got {max_iter!r}.")
