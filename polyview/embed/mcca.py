from __future__ import annotations

from typing import List, Literal, Optional, Sequence, Union, cast

import numpy as np
from scipy.linalg import cho_solve
from sklearn.utils.validation import check_is_fitted

from polyview.base import BaseMultiViewTransformer
from polyview.utils.linalg import (
    EigenSolver,
    resolve_smoother_solver,
    smoother_sum_operator,
    truncated_eigh,
)


OutputMode = Literal["concat", "mean", "list"]
ObjectiveMode = Literal["sumcor", "maxvar"]


def _make_output(
    projections: List[np.ndarray],
    mode: OutputMode,
) -> Union[np.ndarray, List[np.ndarray]]:
    if mode == "concat":
        return np.concatenate(projections, axis=1)
    if mode == "mean":
        return np.mean(projections, axis=0)
    if mode == "list":
        return projections
    raise ValueError(f"output must be 'concat', 'mean', or 'list', got {mode!r}.")


def _center_columns(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = X.mean(axis=0)
    return X - mu, mu


class MCCA(BaseMultiViewTransformer):
    """Multi-set CCA (Kettenring, 1971) with selectable objective.

    Supported objectives:
    - ``"sumcor"``: SUMCOR criterion, solved via generalized eigenproblem over concatenated feature-space covariance blocks.
    - ``"maxvar"``: MAXVAR-style shared latent criterion, solved in sample space using summed smoother matrices (GCCA-like formulation).

    Parameters
    ----------
    n_components : int or None, default=None
      Number of canonical components. If ``None``, use the smallest feature dimension across views.
    regularisation : float or list of float, default=1e-6
      Ridge term added to each within-view covariance block.
    objective : {"sumcor", "maxvar"}, default="sumcor"
      Kettenring objective variant used during fitting.
    output : {"concat", "mean", "list"}, default="concat"
      How to combine per-view projections in ``transform``.
    centre : bool, default=True
      Whether to center columns of each view before fitting.
    eigen_solver : {"auto", "dense", "arpack"}, default="auto"
      Solver for the top ``n_components`` eigenvectors. "dense" uses LAPACK restricted to the requested eigenpairs; "arpack" uses an iterative Lanczos solver (matrix-free for ``"maxvar"``, so the (n, n) smoother matrix is never formed). "auto" picks "arpack" when the problem size exceeds 200 and ``n_components < 10`` for ``"sumcor"``, and, for ``"maxvar"``, when ``n_samples > 200`` and the views have fewer features in total than there are samples; "dense" otherwise.
    n_views : int or None, default=None
      Expected number of views.

    Attributes
    ----------
    weights_ : list of ndarray
      Per-view projection matrices.
    eigenvalues_ : ndarray
      Top generalized eigenvalues.
    means_ : list of ndarray
      Per-view means used for centering.

    .. rubric:: References

    - Kettenring, J. R. (1971). Canonical analysis of several sets of variables.
      Biometrika, 58(3), 433-451.
    - Guo, C., & Wu, D. (2021). Canonical correlation analysis (CCA) based multi-view learning: An overview.
      arXiv preprint arXiv:1907.01693.
    """

    def __init__(
        self,
        n_components: Optional[int] = None,
        regularisation: Union[float, List[float]] = 1e-6,
        objective: ObjectiveMode = "sumcor",
        output: OutputMode = "concat",
        centre: bool = True,
        eigen_solver: EigenSolver = "auto",
        n_views: Optional[int] = None,
    ) -> None:
        super().__init__(n_views=n_views)
        self.n_components = n_components
        self.regularisation = regularisation
        self.objective = objective
        self.output = output
        self.centre = centre
        self.eigen_solver = eigen_solver

    def _resolve_regularisation(self, n_views: int) -> List[float]:
        r = self.regularisation
        if isinstance(r, (int, float)):
            return [float(r)] * n_views
        r_seq = list(cast(Sequence[float], r))
        if len(r_seq) != n_views:
            raise ValueError(
                f"regularisation has {len(r_seq)} entries but there are {n_views} views."
            )
        return [float(v) for v in r_seq]

    def _resolve_n_components(self, views: List[np.ndarray]) -> int:
        d_min = min(X.shape[1] for X in views)
        if self.n_components is None:
            return d_min
        k = int(self.n_components)
        if k <= 0:
            raise ValueError("n_components must be a positive integer.")
        if k > d_min:
            raise ValueError(
                f"n_components={k} exceeds smallest view dimension ({d_min})."
            )
        return k

    def _fit_sumcor(
        self,
        centred: List[np.ndarray],
        regs: List[float],
        k: int,
    ) -> None:
        n = self.n_samples_
        blocks = [X.shape[1] for X in centred]
        offsets = np.cumsum([0] + blocks)

        Xcat = np.concatenate(centred, axis=1)
        C = (Xcat.T @ Xcat) / max(1, n - 1)

        # B = blockdiag(C_vv + r_v I): its inverse square root is computed
        # block by block rather than by eigendecomposing the full matrix.
        B_inv_sqrt = np.zeros_like(C)
        for i, reg in enumerate(regs):
            a, b = offsets[i], offsets[i + 1]
            Bii = C[a:b, a:b] + reg * np.eye(b - a)
            evals, evecs = np.linalg.eigh(Bii)
            B_inv_sqrt[a:b, a:b] = (evecs / np.sqrt(np.maximum(evals, 1e-12))) @ evecs.T
        M = B_inv_sqrt @ C @ B_inv_sqrt
        M = (M + M.T) / 2.0

        vals, vecs = truncated_eigh(M, k, largest=True, eigen_solver=self.eigen_solver)

        A = B_inv_sqrt @ vecs
        self.eigenvalues_ = vals
        self.n_components_ = k

        self.weights_ = []
        for i in range(self.n_views_in_):
            a, b = offsets[i], offsets[i + 1]
            self.weights_.append(A[a:b, :])

    def _fit_maxvar(
        self,
        centred: List[np.ndarray],
        regs: List[float],
        k: int,
    ) -> None:
        M_agg, factors = smoother_sum_operator(centred, regs)

        solver = resolve_smoother_solver(
            self.eigen_solver, self.n_samples_, sum(X.shape[1] for X in centred), k
        )
        self.eigenvalues_, self.G_ = truncated_eigh(
            M_agg, k, largest=True, eigen_solver=solver
        )
        self.n_components_ = k

        self.weights_ = [
            cho_solve(fac, X.T @ self.G_) for X, fac in zip(centred, factors)
        ]

    def fit(self, views: List[np.ndarray], y=None) -> "MCCA":
        views = self._validate_views(views, reset=True)
        regs = self._resolve_regularisation(self.n_views_in_)
        k = self._resolve_n_components(views)

        if self.centre:
            centred, self.means_ = zip(*[_center_columns(X) for X in views])
            centred = list(centred)
            self.means_ = list(self.means_)
        else:
            centred = views
            self.means_ = [np.zeros(X.shape[1]) for X in views]

        objective = cast(ObjectiveMode, self.objective)
        if objective == "sumcor":
            self._fit_sumcor(centred, regs, k)
        elif objective == "maxvar":
            self._fit_maxvar(centred, regs, k)
        else:
            raise ValueError(
                f"objective must be 'sumcor' or 'maxvar', got {self.objective!r}."
            )

        self._centred_views_ = centred
        return self

    def transform(self, views: List[np.ndarray]) -> Union[np.ndarray, List[np.ndarray]]:
        check_is_fitted(self, ["weights_", "means_", "n_components_"])
        views = self._validate_views(views, reset=False)

        projections = [
            (X - mu) @ W for X, mu, W in zip(views, self.means_, self.weights_)
        ]
        return _make_output(projections, cast(OutputMode, self.output))

    def canonical_correlations(self) -> np.ndarray:
        """Return pairwise per-component correlations on the fitted data."""
        check_is_fitted(self, ["weights_", "_centred_views_", "n_components_"])
        M = self.n_views_in_
        k = self.n_components_
        out = np.zeros((M, M, k))

        Zs = [X @ W for X, W in zip(self._centred_views_, self.weights_)]
        for i in range(M):
            for j in range(i + 1, M):
                for c in range(k):
                    zi = Zs[i][:, c]
                    zj = Zs[j][:, c]
                    denom = (np.std(zi) * np.std(zj)) + 1e-12
                    r = float(np.dot(zi, zj) / (len(zi) * denom))
                    out[i, j, c] = out[j, i, c] = r
        return out
