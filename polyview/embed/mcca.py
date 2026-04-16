from __future__ import annotations

from typing import List, Literal, Optional, Sequence, Union, cast

import numpy as np
from sklearn.utils.validation import check_is_fitted

from polyview.base import BaseMultiViewTransformer


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

    Reference
    ---------
    Kettenring, J. R. (1971). Canonical analysis of several sets of variables.
    Biometrika, 58(3), 433-451.

    Guo, Chenfeng and Dongrui Wu. Canonical Correlation Analysis (CCA) Based Multi-View Learning: An Overview.
    ArXiv preprint arXiv:1907.01693 (2021).
    """

    def __init__(
        self,
        n_components: Optional[int] = None,
        regularisation: Union[float, List[float]] = 1e-6,
        objective: ObjectiveMode = "sumcor",
        output: OutputMode = "concat",
        centre: bool = True,
        n_views: Optional[int] = None,
    ) -> None:
        super().__init__(n_views=n_views)
        self.n_components = n_components
        self.regularisation = regularisation
        self.objective = objective
        self.output = output
        self.centre = centre

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

    @staticmethod
    def _smoother(X: np.ndarray, reg: float) -> np.ndarray:
        d = X.shape[1]
        XtX_reg = X.T @ X + reg * np.eye(d)
        return X @ np.linalg.solve(XtX_reg, X.T)

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

        B = np.zeros_like(C)
        for i, (X, reg) in enumerate(zip(centred, regs)):
            a, b = offsets[i], offsets[i + 1]
            Cii = (X.T @ X) / max(1, n - 1)
            B[a:b, a:b] = Cii + reg * np.eye(Cii.shape[0])

        evals_B, evecs_B = np.linalg.eigh(B)
        inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(evals_B, 1e-12)))
        B_inv_sqrt = evecs_B @ inv_sqrt @ evecs_B.T
        M = B_inv_sqrt @ C @ B_inv_sqrt
        M = (M + M.T) / 2.0

        vals, vecs = np.linalg.eigh(M)
        idx = np.argsort(vals)[::-1]
        vals = vals[idx]
        vecs = vecs[:, idx]

        A = B_inv_sqrt @ vecs[:, :k]
        self.eigenvalues_ = vals[:k]
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
        n = self.n_samples_

        M_agg = np.zeros((n, n))
        for X, reg in zip(centred, regs):
            M_agg += self._smoother(X, reg)

        vals, vecs = np.linalg.eigh(M_agg)
        idx = np.argsort(vals)[::-1]
        self.G_ = vecs[:, idx[:k]]
        self.eigenvalues_ = vals[idx[:k]]
        self.n_components_ = k

        self.weights_ = []
        for X, reg in zip(centred, regs):
            XtX_reg = X.T @ X + reg * np.eye(X.shape[1])
            W = np.linalg.solve(XtX_reg, X.T @ self.G_)
            self.weights_.append(W)

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
