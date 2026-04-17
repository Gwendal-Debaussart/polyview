from __future__ import annotations

from typing import List, Literal, Optional, Union

import numpy as np
from sklearn.metrics.pairwise import rbf_kernel, linear_kernel, polynomial_kernel
from sklearn.utils.validation import check_is_fitted

from polyview.base import BaseMultiViewTransformer
from polyview.fusion.kernel_fusion import center_kernel, normalize_kernel


OutputMode = Literal["concat", "mean", "list"]


def _make_output(
    projections: List[np.ndarray],
    mode: OutputMode,
) -> Union[np.ndarray, List[np.ndarray]]:
    """Combine per-view projections according to output mode.

    Parameters
    ----------
    projections : list of (n, k) arrays
    mode : "concat" | "mean" | "list"

    Returns
    -------
    ndarray of shape (n, M*k) for "concat",
    ndarray of shape (n, k)   for "mean",
    list of (n, k) arrays     for "list".
    """
    if mode == "concat":
        return np.concatenate(projections, axis=1)
    if mode == "mean":
        return np.mean(projections, axis=0)
    if mode == "list":
        return projections
    raise ValueError(f"output must be 'concat', 'mean', or 'list', got {mode!r}.")


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
    output : str {"concat", "mean", "list"}, default="concat"
        How to combine per-view projections in transform():
        - "concat" : [Z1 | Z2 | ... | ZM]  shape (n, M*k)
        - "mean"   : (Z1 + Z2 + ... + ZM) / M  shape (n, k)
        - "list"   : [Z1, Z2, ..., ZM]  list of (n, k) arrays
    centre : bool, default=True
        Subtract column means from each view before fitting.

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
    >>> from polyview.embed.cca import GCCA
    >>> gcca = GCCA(n_components=10, output="concat")
    >>> Z_train = gcca.fit_transform([X1, X2, X3])
    >>> Z_test  = gcca.transform([T1, T2, T3])

    .. rubric:: References

    - Guo, C., & Wu, D. (2021). Canonical correlation analysis (CCA) based multi-view learning: An overview.
      arXiv preprint arXiv:1907.01693.
    """

    def __init__(
        self,
        n_components: int = 2,
        regularisation: Union[float, List[float]] = 1e-4,
        output: OutputMode = "concat",
        centre: bool = True,
        n_views: Optional[int] = None,
    ) -> None:
        super().__init__(n_views=n_views)
        self.n_components = n_components
        self.regularisation = regularisation
        self.output = output
        self.centre = centre

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

    def _smoother(self, X: np.ndarray, reg: float) -> np.ndarray:
        """Compute hat matrix S(v) = X (X^T X + r I)^{-1} X^T  shape (n, n)."""
        d = X.shape[1]
        XtX_reg = X.T @ X + reg * np.eye(d)
        return X @ np.linalg.solve(XtX_reg, X.T)

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

        # Build aggregated smoother M = sum_v S(v)
        M_agg = np.zeros((n, n))
        for X, reg in zip(centred, regs):
            M_agg += self._smoother(X, reg)

        # Shared embedding: top-k eigenvectors of M_agg
        vals, vecs = np.linalg.eigh(M_agg)
        idx = np.argsort(vals)[::-1]
        k = self.n_components
        self.G_ = vecs[:, idx[:k]]  # (n, k)
        self.eigenvalues_ = vals[idx[:k]]

        # Per-view projection matrices W(v) = (X^T X + rI)^{-1} X^T G
        self.weights_ = []
        for X, reg in zip(centred, regs):
            XtX_reg = X.T @ X + reg * np.eye(X.shape[1])
            W = np.linalg.solve(XtX_reg, X.T @ self.G_)
            self.weights_.append(W)

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
        return _make_output(projections, self.output)

    def canonical_correlations(self) -> np.ndarray:
        """Pairwise canonical correlations between all view pairs.

        Returns
        -------
        ndarray of shape (n_views, n_views, n_components)
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
