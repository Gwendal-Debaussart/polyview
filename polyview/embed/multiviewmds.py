from __future__ import annotations
from typing import Dict, List, Optional, Sequence
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.utils.validation import check_is_fitted
from polyview.base import BaseMultiViewEmbedder


class MultiViewMDS(BaseMultiViewEmbedder):
    r"""Multi-view MDS with adaptive view weighting (Bai et al., 2017).

    The method minimizes a weighted multi-view stress objective:

    .. math::

        J(X, \alpha) = \sum_{v=1}^{M} \alpha_v^{\gamma} J^{(v)}(X),

    where

    .. math::

        J^{(v)}(X) = \frac{1}{2}\sum_{i \neq j} w^{(v)}_{ij}
        \left(\delta^{(v)}_{ij} - d_{ij}(X)\right)^2,

    with :math:`d_{ij}(X)=\|x_i-x_j\|_2` and :math:`\delta^{(v)}` the input
    dissimilarities for view :math:`v`.

    Parameters
    ----------
    n_components : int, default=2
        Embedding dimension.
    gamma : float, default=2.0
        View-weight concentration parameter. Must be > 1.
    dissimilarity : {"euclidean", "precomputed"}, default="euclidean"
        If "euclidean", each view is treated as a feature matrix and converted
        to pairwise distances. If "precomputed", each view is a square
        dissimilarity matrix.
    metric : str, default="euclidean"
        Metric passed to :func:`sklearn.metrics.pairwise_distances` when
        ``dissimilarity='euclidean'``.
    metric_params : dict or None, default=None
        Extra keyword arguments for ``pairwise_distances``.
    max_iter : int, default=200
        Maximum number of MM iterations.
    tol : float, default=1e-6
        Relative objective decrease tolerance.
    eps : float, default=1e-12
        Numerical stability constant.
    random_state : int or None, default=None
        Random seed for initialization.
    init : ndarray of shape (n_samples, n_components) or None, default=None
        Optional initial embedding.
    n_views : int or None, default=None
        Expected number of views.

    Attributes
    ----------
    embedding_ : ndarray of shape (n_samples, n_components)
        Learned embedding.
    stress_ : float
        Final objective value :math:`J(X, \alpha)`.
    view_stress_ : ndarray of shape (n_views,)
        Final per-view stress terms :math:`J^{(v)}(X)`.
    view_weights_ : ndarray of shape (n_views,)
        Learned view weights :math:`\alpha`.
    dissimilarities_ : list of ndarray
        Per-view input dissimilarity matrices used in fitting.
    n_iter_ : int
        Number of optimization iterations.

    References
    ----------
    - Bai, S., Bai, X., Latecki, L. J., & Tian, Q. (2017).
      Multidimensional Scaling on Multiple Input Distance Matrices. arXiv:1605.00286.
    """

    def __init__(
        self,
        n_components: int = 2,
        gamma: float = 2.0,
        dissimilarity: str = "euclidean",
        metric: str = "euclidean",
        metric_params: Optional[Dict] = None,
        max_iter: int = 200,
        tol: float = 1e-6,
        eps: float = 1e-12,
        random_state: Optional[int] = None,
        init: Optional[np.ndarray] = None,
        n_views: Optional[int] = None,
    ) -> None:
        super().__init__(n_views=n_views)
        self.n_components = n_components
        self.gamma = gamma
        self.dissimilarity = dissimilarity
        self.metric = metric
        self.metric_params = metric_params
        self.max_iter = max_iter
        self.tol = tol
        self.eps = eps
        self.random_state = random_state
        self.init = init

    def _compute_view_dissimilarities(self, views: List[np.ndarray]) -> List[np.ndarray]:
        mode = str(self.dissimilarity).lower()
        if mode not in {"euclidean", "precomputed"}:
            raise ValueError(
                "dissimilarity must be 'euclidean' or 'precomputed', "
                f"got {self.dissimilarity!r}."
            )

        out: List[np.ndarray] = []
        if mode == "precomputed":
            n = views[0].shape[0]
            for i, D in enumerate(views):
                D = np.asarray(D, dtype=float)
                if D.shape != (n, n):
                    raise ValueError(
                        f"Precomputed dissimilarity for view {i} must have shape ({n}, {n}), got {D.shape}."
                    )
                if not np.allclose(D, D.T, atol=1e-10):
                    raise ValueError(
                        f"Precomputed dissimilarity for view {i} must be symmetric."
                    )
                if np.any(D < 0):
                    raise ValueError(
                        f"Precomputed dissimilarity for view {i} contains negative values."
                    )
                out.append(D)
            return out

        metric_params = self.metric_params or {}
        for X in views:
            D = pairwise_distances(X, metric=self.metric, **metric_params)
            out.append(np.asarray(D, dtype=float))
        return out

    def _compute_v_matrix(
        self,
        weight_mats: List[np.ndarray],
        alpha: np.ndarray,
    ) -> np.ndarray:
        n = weight_mats[0].shape[0]
        v_off = np.zeros((n, n), dtype=float)
        for a, Wv in zip(alpha, weight_mats):
            v_off -= (a**self.gamma) * Wv
        np.fill_diagonal(v_off, 0.0)
        diag = -np.sum(v_off, axis=1)
        V = v_off
        np.fill_diagonal(V, diag)
        return V

    def _compute_b_matrix(
        self,
        Z: np.ndarray,
        deltas: List[np.ndarray],
        weight_mats: List[np.ndarray],
        alpha: np.ndarray,
    ) -> np.ndarray:
        n = Z.shape[0]
        Dz = pairwise_distances(Z, metric="euclidean")
        ratio = np.zeros((n, n), dtype=float)

        for a, Delta_v, Wv in zip(alpha, deltas, weight_mats):
            ratio += (a**self.gamma) * Wv * Delta_v

        off = np.zeros((n, n), dtype=float)
        mask = ~np.eye(n, dtype=bool)
        denom = np.maximum(Dz, self.eps)
        off[mask] = -(ratio[mask] / denom[mask])
        diag = -np.sum(off, axis=1)
        B = off
        np.fill_diagonal(B, diag)
        return B

    def _stress_per_view(
        self,
        X: np.ndarray,
        deltas: List[np.ndarray],
        weight_mats: List[np.ndarray],
    ) -> np.ndarray:
        Dx = pairwise_distances(X, metric="euclidean")
        J = np.zeros(len(deltas), dtype=float)
        for v, (Delta_v, Wv) in enumerate(zip(deltas, weight_mats)):
            diff = Delta_v - Dx
            J[v] = 0.5 * float(np.sum(Wv * (diff**2)))
        return J

    def _update_alpha(self, view_stress: np.ndarray) -> np.ndarray:
        if abs(self.gamma - 1.0) < 1e-12:
            alpha = np.zeros_like(view_stress)
            alpha[np.argmin(view_stress)] = 1.0
            return alpha

        if self.gamma <= 1.0:
            raise ValueError("gamma must be > 1.0 for the Bai et al. update rule.")

        power = 1.0 / (1.0 - self.gamma)
        safe = np.maximum(view_stress, self.eps)
        num = safe**power
        total = float(np.sum(num))
        if total <= 0:
            return np.full_like(view_stress, 1.0 / len(view_stress))
        return num / total

    def fit(self, views: List[np.ndarray], y=None) -> "MultiViewMDS":
        """Fit the multi-view MDS model.

        Parameters
        ----------
        views : list of ndarray
            Input views as feature matrices or precomputed dissimilarities.
        y : ignored

        Returns
        -------
        self : MultiViewMDS
        """
        views = self._validate_views(views, reset=True)

        if self.n_components <= 0:
            raise ValueError("n_components must be a positive integer.")
        if self.max_iter <= 0:
            raise ValueError("max_iter must be a positive integer.")

        deltas = self._compute_view_dissimilarities(views)
        n = deltas[0].shape[0]
        m = len(deltas)

        weight_mats = [np.ones((n, n), dtype=float) for _ in range(m)]
        for Wv in weight_mats:
            np.fill_diagonal(Wv, 0.0)

        if self.init is None:
            rng = np.random.RandomState(self.random_state)
            X = rng.normal(size=(n, self.n_components))
        else:
            X = np.asarray(self.init, dtype=float)
            if X.shape != (n, self.n_components):
                raise ValueError(
                    f"init must have shape ({n}, {self.n_components}), got {X.shape}."
                )

        X -= X.mean(axis=0, keepdims=True)
        alpha = np.full(m, 1.0 / m, dtype=float)

        prev_obj = np.inf
        objective_history: List[float] = []

        for iteration in range(1, self.max_iter + 1):
            Z = X.copy()
            B = self._compute_b_matrix(Z, deltas, weight_mats, alpha)
            V = self._compute_v_matrix(weight_mats, alpha)
            X = np.linalg.pinv(V) @ B @ Z
            X -= X.mean(axis=0, keepdims=True)

            view_stress = self._stress_per_view(X, deltas, weight_mats)
            alpha = self._update_alpha(view_stress)

            obj = float(np.sum((alpha**self.gamma) * view_stress))
            objective_history.append(obj)

            if np.isfinite(prev_obj):
                rel = abs(prev_obj - obj) / max(abs(prev_obj), self.eps)
                if rel < self.tol:
                    break
            prev_obj = obj

        self.embedding_ = X
        self.view_weights_ = alpha
        self.view_stress_ = view_stress
        self.stress_ = objective_history[-1]
        self.objective_history_ = np.asarray(objective_history, dtype=float)
        self.dissimilarities_ = deltas
        self.n_iter_ = iteration
        return self

    def fit_transform(self, views: List[np.ndarray], y=None) -> np.ndarray:
        """Fit the model and return the learned embedding."""
        return self.fit(views, y).embedding_

    def transform(self, views: List[np.ndarray]) -> np.ndarray:
        """Return embedding of the training samples.

        This algorithm is non-parametric and does not provide an out-of-sample
        extension. The method validates input consistency and returns
        ``embedding_``.
        """
        check_is_fitted(self, ["embedding_", "n_views_in_", "n_features_in_"])
        self._validate_views(views, reset=False)
        return self.embedding_


MVMDS = MultiViewMDS