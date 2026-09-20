from __future__ import annotations

from typing import List, Literal, Optional, Union

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


OutputMode = Literal["concat", "mean", "list", "shared"]


def _center_columns(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Zero-centre each column of X. Returns (X_centred, col_means)."""
    mu = X.mean(axis=0)
    return X - mu, mu


class GCCA(BaseMultiViewTransformer):
    """
    Generalised Canonical Correlation Analysis (GCCA).

    Finds a shared low-dimensional embedding G that maximises linear agreement across all views simultaneously (MAXVAR criterion). Works with M >= 2 views.  When M = 2 this recovers classical CCA.

    Parameters
    ----------
    n_components : int, default=2
        Number of shared dimensions k.
    regularisation : float or list of float, default=1e-4
        Ridge regularisation added to each view's covariance before
        inversion.  A single float applies the same value to all views; a list gives per-view values.  Larger values = stronger regularisation (useful when d_v > n or features are collinear).
    output : str {"concat", "mean", "list", "shared"}, default="shared"
        How to combine per-view projections in transform():
        - "concat" : [Z1 | Z2 | ... | ZM]  shape (n, M*k)
        - "mean"   : (Z1 + Z2 + ... + ZM) / M  shape (n, k)
        - "list"   : [Z1, Z2, ..., ZM]  list of (n, k) arrays
        - "shared" : ``G_``  shape (n, n_components)
    centre : bool, default=True
        Subtract column means from each view before fitting.
    eigen_solver : {"auto", "dense", "arpack"}, default="auto"
        Solver for the top ``n_components`` eigenvectors of the aggregated smoother matrix.
        - "dense"  : LAPACK, restricted to the requested eigenpairs.
        - "arpack" : iterative Lanczos solver applied matrix-free, so the (n, n) smoother matrix is never formed. Much faster and lighter in memory for large ``n_samples``.
        - "auto"   : "arpack" when ``n_samples > 200`` and the views have fewer features in total than there are samples, "dense" otherwise (the leading eigenvalues then cluster and Lanczos converges slowly).

    Attributes
    ----------
    G_ : ndarray of shape (n_train, n_components)
        Shared embedding of the training data.
    weights_ : list of ndarray, shape (n_features_v, n_components)
        Per-view projection matrices W(v).
    means_ : list of ndarray, shape (n_features_v,)
        Per-view column means (used to centre test data).
    eigenvalues_ : ndarray of shape (n_components,)
        Top-k eigenvalues of the aggregated smoother matrix.

    Examples
    --------
    >>> import numpy as np
    >>> from polyview.embed.gcca import GCCA
    >>> X1, X2, X3 = (np.random.rand(100, 20) for _ in range(3))
    >>> gcca = GCCA(n_components=10, output="concat")
    >>> Z_train = gcca.fit_transform([X1, X2, X3])
    >>> Z_train.shape
    (100, 30)
    >>> T1, T2, T3 = (np.random.rand(20, 20) for _ in range(3))
    >>> Z_test = gcca.transform([T1, T2, T3])
    >>> Z_test.shape
    (20, 30)

    .. rubric:: References

    - Guo, C., & Wu, D. (2021). Canonical correlation analysis (CCA) based multi-view learning: An overview.
      arXiv preprint arXiv:1907.01693.
    """

    def __init__(
        self,
        n_components: int = 2,
        regularisation: Union[float, List[float]] = 1e-4,
        output: OutputMode = "shared",
        centre: bool = True,
        eigen_solver: EigenSolver = "auto",
        n_views: Optional[int] = None,
    ) -> None:
        super().__init__(n_views=n_views)
        self.n_components = n_components
        self.regularisation = regularisation
        self.output = output
        self.centre = centre
        self.eigen_solver = eigen_solver

    def _get_reg(self, n_views: int) -> List[float]:
        r = self.regularisation
        if np.isscalar(r):
            return [float(r)] * n_views
        r = list(r)
        if len(r) != n_views:
            raise ValueError(
                f"regularisation list has {len(r)} entries but there are {n_views} views."
            )
        return [float(x) for x in r]

    def _make_output(
        self,
        projections: List[np.ndarray],
        mode: OutputMode,
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """Combine per-view projections according to output mode.

        Parameters
        ----------
        projections : list of (n, k) arrays
        mode : "concat" | "mean" | "list" | "shared"

        Returns
        -------
        ndarray of shape (n, M*k) for "concat",
        ndarray of shape (n, k)   for "mean",
        list of (n, k) arrays     for "list",
        ndarray of shape (n, k)   for "shared".
        """
        if mode == "concat":
            return np.concatenate(projections, axis=1)
        if mode == "mean":
            return np.mean(projections, axis=0)
        if mode == "shared":
            return self.G_
        if mode == "list":
            return projections
        raise ValueError(
            f"output must be 'concat', 'mean', 'shared', or 'list', got {mode!r}."
        )

    def fit(self, views: List[np.ndarray], y=None) -> "GCCA":
        """
        Fit the GCCA model to the training data.

        Parameters
        ----------
        views : list of (n, d_v) arrays
            Training data from each view.
        y : ignored

        Returns
        -------
        self : GCCA
            The fitted GCCA model.
        """

        views = self._validate_views(views, reset=True)
        n = self.n_samples_
        regs = self._get_reg(self.n_views_in_)

        # Centre each view, store means for transform()
        if self.centre:
            centred, self.means_ = zip(*[_center_columns(X) for X in views])
            centred = list(centred)
            self.means_ = list(self.means_)
        else:
            centred = views
            self.means_ = [np.zeros(X.shape[1]) for X in views]

        # Aggregated smoother M = sum_v X (X^T X / n + rI)^{-1} X^T / n,
        # kept matrix-free
        M_agg, factors = smoother_sum_operator(centred, regs, scale=n)

        # Shared embedding: top-k eigenvectors of M_agg
        solver = resolve_smoother_solver(
            self.eigen_solver, n, sum(self.n_features_in_), self.n_components
        )
        self.eigenvalues_, self.G_ = truncated_eigh(
            M_agg, self.n_components, largest=True, eigen_solver=solver
        )

        # Per-view projection matrices W(v) = (X^T X / n + rI)^{-1} X^T G / n
        self.weights_ = [
            cho_solve(fac, X.T @ self.G_ / n) for X, fac in zip(centred, factors)
        ]

        self._centred_views_ = centred

        return self

    def transform(self, views: List) -> Union[np.ndarray, List[np.ndarray]]:
        """Project views into the shared embedding space.

        Parameters
        ----------
        views : list of array-like of shape (n_samples, n_features_v)

        Returns
        -------
        Depends on ``output`` parameter — see class docstring.
        """
        check_is_fitted(self, "weights_")
        views = self._validate_views(views, reset=False)

        projections = [
            (X - mu) @ W for X, mu, W in zip(views, self.means_, self.weights_)
        ]
        if self.output == "shared":
            # sum_v X_v W_v = M G = G diag(eigenvalues) on the training data, so
            # this recovers G_ there and extends it to new samples.
            return np.sum(projections, axis=0) / self.eigenvalues_
        return self._make_output(projections, self.output)

    def canonical_correlations(self) -> np.ndarray:
        """Pairwise canonical correlations between all view pairs.

        Returns
        -------
        ``ndarray of shape (n_views, n_views, n_components)``
          corrs[v1, v2, :] = per-component correlation between
          projections of view v1 and view v2.
        """
        check_is_fitted(self, "weights_")
        M = self.n_views_in_
        k = self.n_components
        out = np.zeros((M, M, k))
        Zs = [
            (X - mu) @ W
            for X, mu, W in zip(self._centred_views_, self.means_, self.weights_)
        ]
        for v1 in range(M):
            for v2 in range(v1 + 1, M):
                for c in range(k):
                    z1, z2 = Zs[v1][:, c], Zs[v2][:, c]
                    denom = (np.std(z1) * np.std(z2)) + 1e-10
                    r = float(np.dot(z1, z2) / (len(z1) * denom))
                    out[v1, v2, c] = out[v2, v1, c] = r
        return out
