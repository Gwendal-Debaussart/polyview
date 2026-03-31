from __future__ import annotations

import warnings
from typing import List, Optional

import numpy as np
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning

from polyview.base import BaseMultiViewClusterer


class MultiViewKMeans(BaseMultiViewClusterer):
    """
    Multi-view K-Means clustering.

    Parameters
    ----------
    n_clusters : int, default=2
        Number of clusters.
    gamma : float, default=2.0
        Controls the distribution of view weights alpha(v).
    max_iter : int, default=50
        Maximum number of alternating-update iterations.
    n_init : int, default=10
        Number of random restarts. Best result (lowest objective) kept.
    tol : float, default=1e-6
        Stop when the relative change in objective falls below this.
    learn_weights : bool, default=True
        If True, run RMKMC (adaptive view weights). If False, run SMKMC (equal view weights, simpler).
    eps : float, default=1e-10
        Small constant added to row norms before inversion (D update) to avoid division by zero on zero-residual samples.
    random_state : int or None, default=None
        Random seed.

    Attributes
    ----------
    labels_ : ndarray of shape (n_samples,)
        Consensus cluster assignment.
    centroids_ : list of ndarray, shape (n_clusters, n_features_v)
        Per-view cluster centroid matrices F(v).
    weights_ : ndarray of shape (n_views,)
        Learned view importance weights alpha(v). All equal to 1/n_views when learn_weights=False.
    objective_ : float
        Final value of the objective function.
    n_iter_ : int
        Number of iterations performed in the best run.

    Notes
    -----
    The gamma parameter controls how the view weights alpha(v) are distributed:
        - gamma -> inf gives equal weights.
        - gamma -> 1 collapses weight onto the single best view.
    The paper recommends searching log10(gamma) in [0.1, 2.0].
    Setting ``learn_weights=False`` recovers the Simple Multi-view K-Means (SMKMC) variant from the same paper, which uses equal view weights throughout.

    Examples
    --------
    >>> import numpy as np
    >>> from polyview.cluster.kmeans import MultiviewKMeans
    >>> X1 = np.random.rand(100, 4)
    >>> X2 = np.random.rand(100, 6)
    >>> model = MultiviewKMeans(n_clusters=3, random_state=0)
    >>> labels = model.fit_predict([X1, X2])
    >>> labels.shape
    (100,)

    Reference
    ---------
    Cai, X. et al. (2013). Multi-view K-means Clustering on Big Data. IEEE Transactions on Knowledge and Data Engineering.
    """

    def __init__(
        self,
        n_clusters: int = 2,
        gamma: float = 2.0,
        max_iter: int = 50,
        n_init: int = 10,
        tol: float = 1e-6,
        learn_weights: bool = True,
        eps: float = 1e-10,
        random_state: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.max_iter = max_iter
        self.n_init = n_init
        self.tol = tol
        self.learn_weights = learn_weights
        self.eps = eps
        self.random_state = random_state

    def _objective(
        self,
        views: List[np.ndarray],
        G: np.ndarray,
        F: List[np.ndarray],
        alpha: np.ndarray,
    ) -> float:
        r"""
        Computes the objective function value for given cluster assignment G, centroids F, and view weights alpha.

        $\sum_v \alpha(v)^{\gamma} \|X(v) - G F(v)\|_{2,1}$

        Parameters
        ----------
        views : list of (n, d_v)
            List of view data arrays.
        G : (n, K)
            Cluster indicator matrix.
        F : list of (K, d_v)
            List of centroid matrices for each view.
        alpha : (n_views,)
            View weights.

        Returns
        -------
        obj : float
            Objective function value.
        """
        obj = 0.0
        for v, (X, Fv) in enumerate(zip(views, F)):
            E = X - G @ Fv
            row_norms = np.linalg.norm(E, axis=1)
            obj += (alpha[v] ** self.gamma) * row_norms.sum()
        return float(obj)

    def _update_D(
        self,
        X: np.ndarray,
        G: np.ndarray,
        F: np.ndarray,
    ) -> np.ndarray:
        r"""
        Updates D(v) diagonal entries.

        $D(v)_{ii} = 1 / (2 \|\|e(v)_i\|\|_2)$

        Parameters
        ----------
        X : (n, d_v)
                View data.
        G : (n, K)
                Cluster indicator.
        F : (K, d_v)
                Centroids (one per row).

        Returns
        -------
        diag : (n,)
                Diagonal entries of D(v).
        """
        E = X - G @ F
        row_norms = np.linalg.norm(E, axis=1)
        return 1.0 / (2.0 * np.maximum(row_norms, self.eps))

    def _update_F(
        self,
        X: np.ndarray,
        G: np.ndarray,
        De: np.ndarray,
    ) -> np.ndarray:
        r"""
        Updates F(v) centroids for view v, given current cluster assignment G and D(v) diagonal De.

        $F^{(v)} = (G^T De G)^{-1} G^T D^{(v)} X$

        Parameters
        ----------
        X : (n, d_v)
            View data.
        G : (n, K)
            Cluster indicator matrix.
        De : (n,)
            Diagonal of De(v) = alpha(v)^gamma * D(v).

        Returns
        -------
        F_new : (K, d_v)
            Centroids as rows.
        """
        DeX = De[:, None] * X
        GtDeG = G.T @ (De[:, None] * G)
        GtDeX = G.T @ DeX
        reg = self.eps * np.eye(self.n_clusters)
        return np.linalg.inv(GtDeG + reg) @ GtDeX

    def _update_G(
        self,
        views: List[np.ndarray],
        F: List[np.ndarray],
        De: List[np.ndarray],
    ) -> np.ndarray:
        r"""
        Updates cluster assignment G by assigning each sample to the cluster with minimum weighted distance.

        For each sample i:
            $k^* = \arg\min_k \sum_v De(v)_i \|x^{(v)}_i - F^{(v)}_k\|^2$

        Parameters
        ----------
        views : list of (n, d_v)
            List of view data arrays.
        F : list of (K, d_v)
            List of centroid matrices for each view.
        De : list of (n,)
            List of diagonals of De(v) for each view.

        Returns
        -------
        G_new : (n, K)
            One-hot cluster indicator matrix.
        """
        n, K = self.n_samples_, self.n_clusters
        cost = np.zeros((n, K))
        for X, Fv, dev in zip(views, F, De):
            diff = X[:, None, :] - Fv[None, :, :]
            sq_dist = np.sum(diff**2, axis=2)
            cost += dev[:, None] * sq_dist

        labels = np.argmin(cost, axis=1)
        G = np.zeros((n, K))
        G[np.arange(n), labels] = 1.0
        return G

    def _update_alpha(
        self,
        views: List[np.ndarray],
        G: np.ndarray,
        F: List[np.ndarray],
        D: List[np.ndarray],
    ) -> np.ndarray:
        r"""
        Updates view weights alpha.

        $\alpha^{(v)} = (\gamma H^{(v)})^{1/(1-\gamma)} / \sum_v (\gamma H^{(v)})^{1/(1-\gamma)}$

        where $H^{(v)} = \sum_i D^{(v)}_{ii} \|x^{(v)}_i - F^{(v)}_{k_i}\|^2$

        Parameters
        ----------
        views : list of (n, d_v)
            List of view data arrays.
        G : (n, K)
            Cluster indicator matrix.
        F : list of (K, d_v)
            List of centroid matrices for each view.
        D : list of (n,)
            List of diagonals of D(v) for each view (not De — no alpha factor).

        Returns
        -------
        alpha : (n_views,)
            Updated view weights.
        """
        H = np.zeros(self.n_views_in_)
        for v, (X, Fv, Dv) in enumerate(zip(views, F, D)):
            E = X - G @ Fv
            row_sq_norms = np.sum(E**2, axis=1)
            H[v] = float(Dv @ row_sq_norms)

        if abs(self.gamma - 1.0) < 1e-10:
            return np.full(self.n_views_in_, 1.0 / self.n_views_in_)

        exponent = 1.0 / (1.0 - self.gamma)
        H_safe = np.maximum(H, self.eps)
        scores = (self.gamma * H_safe) ** exponent
        total = scores.sum()
        if total <= 0 or not np.isfinite(total):
            return np.full(self.n_views_in_, 1.0 / self.n_views_in_)
        return scores / total

    def _run_once(
        self,
        views: List[np.ndarray],
        rng: np.random.RandomState,
    ) -> tuple:
        r"""
        Runs one full alternating-optimisation of the multi-view k-means algorithm.

        Returns
        -------
        labels : ndarray
            Cluster labels for each sample.
        F : list of ndarray
            List of centroid matrices for each view.
        alpha : ndarray
            View weights.
        objective : float
            Final objective value.
        n_iter : int
            Number of iterations performed.
        """
        n, K, M = self.n_samples_, self.n_clusters, self.n_views_in_

        G = np.zeros((n, K))
        rand_labels = rng.randint(0, K, size=n)
        for k in range(K):
            if k not in rand_labels:
                rand_labels[rng.randint(0, n)] = k
        G[np.arange(n), rand_labels] = 1.0
        D = [np.ones(n) for _ in range(M)]
        alpha = np.full(M, 1.0 / M)
        F = []
        for X in views:
            GtG = G.T @ G + self.eps * np.eye(K)
            F.append(np.linalg.inv(GtG) @ (G.T @ X))

        prev_obj = np.inf

        for iteration in range(self.max_iter):
            De = [(alpha[v] ** self.gamma) * D[v] for v in range(M)]
            F = [self._update_F(X, G, dev) for X, dev in zip(views, De)]
            G = self._update_G(views, F, De)
            D = [self._update_D(X, G, Fv) for X, Fv in zip(views, F)]

            if self.learn_weights:
                alpha = self._update_alpha(views, G, F, D)

            obj = self._objective(views, G, F, alpha)
            rel_change = abs(prev_obj - obj) / (abs(prev_obj) + self.eps)
            if rel_change < self.tol:
                iteration += 1
                break
            prev_obj = obj

        labels = np.argmax(G, axis=1)
        return labels, F, alpha, obj, iteration + 1

    def fit(self, views: List, y=None) -> "RMKMC":
        """
        Fit the multi-view k-means model.

        Parameters
        ----------
        views : list of array-like of shape (n_samples, n_features_v)
            List of view data arrays.
        y : None
            Ignored.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        views_validated = self._validate_views(views, reset=True)

        rng = np.random.RandomState(self.random_state)

        best_obj = np.inf
        best_labels = None
        best_F = None
        best_alpha = None
        best_n_iter = 0

        for _ in range(self.n_init):
            labels, F, alpha, obj, n_iter = self._run_once(views_validated, rng)
            if obj < best_obj:
                best_obj = obj
                best_labels = labels
                best_F = F
                best_alpha = alpha
                best_n_iter = n_iter

        self.labels_ = best_labels
        self.centroids_ = best_F
        self.weights_ = best_alpha
        self.objective_ = best_obj
        self.n_iter_ = best_n_iter

        return self

    def predict(self, views: List) -> np.ndarray:
        """
        Assign cluster labels to new samples (nearest centroid per view, weighted by learned alpha).

        Parameters
        ----------
        views : list of array-like
            List of view data arrays for prediction.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Predicted cluster labels.
        """
        self._check_is_fitted()
        views_arr = self._validate_views(views, reset=False)
        n = views_arr[0].shape[0]
        K = self.n_clusters

        cost = np.zeros((n, K))
        for X, Fv, av in zip(views_arr, self.centroids_, self.weights_):
            diff = X[:, None, :] - Fv[None, :, :]
            sq_dist = np.sum(diff**2, axis=2)
            cost += (av**self.gamma) * sq_dist

        return np.argmin(cost, axis=1)

    def fit_predict(self, views: List, y=None) -> np.ndarray:
        """
        Fit the model and return cluster labels.

        Parameters
        ----------
        views : list of array-like
            List of view data arrays.
        y : None
            Ignored.

        Returns
        -------
        labels : ndarray
            Cluster labels for each sample.
        """
        return self.fit(views, y).labels_
