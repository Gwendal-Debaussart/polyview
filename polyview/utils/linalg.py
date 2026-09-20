"""Truncated symmetric eigensolvers shared by the spectral and kernel methods."""

from __future__ import annotations

from typing import List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.linalg import cho_factor, cho_solve, eigh
from scipy.sparse.linalg import LinearOperator, eigsh
from sklearn.utils import check_random_state

EigenSolver = Literal["auto", "dense", "arpack"]


def _resolve_eigen_solver(eigen_solver: str, n: int, k: int) -> str:
    if eigen_solver == "auto":
        # Same heuristic as sklearn's KernelPCA: Lanczos pays off when only a
        # few eigenpairs of a large matrix are needed.
        return "arpack" if n > 200 and k < 10 else "dense"
    if eigen_solver == "dense":
        return eigen_solver
    if eigen_solver == "arpack":
        if k >= n:
            raise ValueError(
                f"eigen_solver='arpack' requires k < n, got k={k} and n={n}."
            )
        return eigen_solver
    raise ValueError(
        f"eigen_solver must be 'auto', 'dense' or 'arpack', got {eigen_solver!r}."
    )


def resolve_smoother_solver(
    eigen_solver: str, n: int, total_features: int, k: int
) -> str:
    """Choose the eigensolver for a sum of ridge smoothers.

    'auto' selects Lanczos ('arpack') when the views have fewer features in
    total than there are samples: the operator then has rank at most
    ``total_features``, each product costs O(n * total_features) and the
    leading eigenvalues are well separated. Otherwise each smoother is close
    to the identity, the leading eigenvalues cluster and Lanczos needs many
    products, so forming the (n, n) matrix and calling LAPACK ('dense') is
    faster.
    """
    if eigen_solver == "auto":
        return "arpack" if n > 200 and total_features < n and k < n else "dense"
    return _resolve_eigen_solver(eigen_solver, n, k)


def truncated_eigh(
    A: Union[np.ndarray, LinearOperator],
    k: int,
    *,
    largest: bool = True,
    eigen_solver: EigenSolver = "auto",
    random_state=None,
    v0: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the ``k`` largest or smallest eigenpairs of a symmetric matrix.

    Parameters
    ----------
    A : ndarray or LinearOperator of shape (n, n)
        Symmetric matrix. A ``LinearOperator`` avoids materialising ``A``
        when ``eigen_solver='arpack'``.
    k : int
        Number of eigenpairs to return, ``1 <= k <= n``.
    largest : bool, default=True
        Return the largest (True) or smallest (False) algebraic eigenvalues.
    eigen_solver : {'auto', 'dense', 'arpack'}, default='auto'
        - 'dense'  : LAPACK ``eigh`` restricted to the requested index range.
        - 'arpack' : implicitly restarted Lanczos (``scipy.sparse.linalg.eigsh``),
          which only needs matrix-vector products. Requires ``k < n``.
        - 'auto'   : 'arpack' if ``n > 200`` and ``k < 10``, else 'dense'.
    random_state : int, RandomState instance or None, default=None
        Seeds the ARPACK starting vector when ``v0`` is None.
    v0 : ndarray of shape (n,) or None, default=None
        ARPACK starting vector, e.g. to warm-start from a previous solution.

    Returns
    -------
    eigenvalues : ndarray of shape (k,)
        Sorted from most to least extreme (descending if ``largest``, else
        ascending).
    eigenvectors : ndarray of shape (n, k)
        Orthonormal eigenvectors, with signs fixed so that the entry of
        largest magnitude in each column is positive. This makes the output
        independent of the solver used.
    """
    n = A.shape[0]
    if not 1 <= k <= n:
        raise ValueError(f"k must be in [1, {n}], got {k}.")
    solver = _resolve_eigen_solver(eigen_solver, n, k)

    if solver == "dense":
        if isinstance(A, LinearOperator):
            A = A @ np.eye(n)
            A = (A + A.T) / 2.0
        lo, hi = (n - k, n - 1) if largest else (0, k - 1)
        vals, vecs = eigh(A, subset_by_index=[lo, hi])
    else:
        if v0 is None:
            v0 = check_random_state(random_state).uniform(-1, 1, n)
        vals, vecs = eigsh(A, k=k, which="LA" if largest else "SA", v0=v0)

    order = np.argsort(vals)
    if largest:
        order = order[::-1]
    vals, vecs = vals[order], vecs[:, order]

    signs = np.sign(vecs[np.argmax(np.abs(vecs), axis=0), np.arange(k)])
    signs[signs == 0] = 1.0
    return vals, vecs * signs


def smoother_sum_operator(
    views: Sequence[np.ndarray],
    regs: Sequence[float],
    scale: float = 1.0,
) -> Tuple[LinearOperator, List[tuple]]:
    r"""Matrix-free sum of ridge smoothers.

    Represents :math:`M = \sum_v X_v (X_v^T X_v / s + r_v I)^{-1} X_v^T / s`
    without forming any (n, n) matrix: each product costs
    :math:`O(n \sum_v d_v)` instead of the :math:`O(n^2 \sum_v d_v)` time and
    :math:`O(n^2)` memory needed to build ``M`` explicitly.

    Parameters
    ----------
    views : sequence of ndarray of shape (n, d_v)
    regs : sequence of float
        Ridge term :math:`r_v` for each view.
    scale : float, default=1.0
        Covariance normalisation :math:`s`.

    Returns
    -------
    op : LinearOperator of shape (n, n)
    factors : list of Cholesky factorisations
        One per view, of :math:`X_v^T X_v / s + r_v I`, for use with
        ``scipy.linalg.cho_solve``.
    """
    n = views[0].shape[0]
    factors = [
        cho_factor(X.T @ X / scale + r * np.eye(X.shape[1]))
        for X, r in zip(views, regs)
    ]

    def matmat(U: np.ndarray) -> np.ndarray:
        U = U.reshape(n, -1)
        out = np.zeros((n, U.shape[1]))
        for X, fac in zip(views, factors):
            out += X @ cho_solve(fac, X.T @ U / scale)
        return out

    op = LinearOperator(
        (n, n), matvec=matmat, rmatvec=matmat, matmat=matmat, dtype=float
    )
    return op, factors
