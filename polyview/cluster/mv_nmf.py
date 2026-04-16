from __future__ import annotations

from typing import List, Optional, Union

import numpy as np
from sklearn.utils.validation import check_is_fitted

from polyview.base import BaseMultiViewClusterer


class MultiViewNMF(BaseMultiViewClusterer):
    """Multi-view Non-negative Matrix Factorisation clustering.

    Finds a shared non-negative coefficient matrix H (the consensus
    representation) and per-view basis matrices W(v) by jointly
    minimising weighted Frobenius reconstruction error across all views.

    The row argmax of H gives cluster labels.  H itself is a soft
    assignment matrix useful as a low-dimensional embedding.

    Parameters
    ----------
    n_components : int, default=2
        Rank k of the factorisation — number of clusters / latent dims.
    max_iter : int, default=200
        Maximum number of multiplicative update iterations.
    n_init : int, default=10
        Number of random initialisations; best (lowest objective) is kept.
    tol : float, default=1e-4
        Stop when relative change in objective falls below this.
    learn_weights : bool, default=False
        If True, adapt per-view weights lambda(v) from reconstruction
        quality.  If False (default), use equal weights 1/M.
    gamma : float, default=2.0
        Controls weight concentration when learn_weights=True.
        Higher gamma -> more uniform weights (approaches equal weighting).
        Only used when learn_weights=True.
    eps : float, default=1e-10
        Small floor added to denominators to prevent division by zero,
        and used to clip H and W away from exact zero.
    random_state : int or None, default=None

    Attributes
    ----------
    H_ : ndarray of shape (n_samples, n_components)
        Shared non-negative coefficient matrix (soft cluster assignments).
    W_ : list of ndarray, shape (n_components, n_features_v)
        Per-view basis matrices.
    weights_ : ndarray of shape (n_views,)
        Final per-view reconstruction weights lambda(v).
        Equal to 1/M when learn_weights=False.
    labels_ : ndarray of shape (n_samples,)
        Hard cluster labels = argmax(``H_``, axis=1).
    reconstruction_errors_ : ndarray of shape (n_views,)
        Per-view Frobenius reconstruction error at convergence.
    objective_ : float
        Weighted sum of reconstruction errors at convergence.
    n_iter_ : int
        Number of iterations performed in the best run.

    Examples
    --------
    >>> from polyview.cluster.mv_nmf import MultiViewNMF
    >>> model = MultiViewNMF(n_components=3, random_state=0)
    >>> labels = model.fit_predict([X1, X2])
    >>> model.H_.shape
    (n_samples, 3)

    Use ``H_`` as a soft embedding downstream:

    >>> from sklearn.cluster import KMeans
    >>> labels = KMeans(n_clusters=3).fit_predict(model.H_)

    Or use ``H_`` as cluster probabilities for soft clustering:

    >>> labels = np.argmax(model.H_, axis=1)

    References
    ----------
    - Gao, J., He, L., Zhang, X., Zhou, J., & Wu, D. (2013).
        Multi-view clustering via joint nonnegative matrix factorization.
        In Proceedings of the 2013 SIAM International Conference on Data Mining (SDM).
    """

    def __init__(
        self,
        n_components: int = 2,
        max_iter: int = 200,
        n_init: int = 10,
        tol: float = 1e-4,
        learn_weights: bool = False,
        gamma: float = 2.0,
        eps: float = 1e-10,
        random_state: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.n_components = n_components
        self.max_iter = max_iter
        self.n_init = n_init
        self.tol = tol
        self.learn_weights = learn_weights
        self.gamma = gamma
        self.eps = eps
        self.random_state = random_state

    def _objective(
        self,
        views: List[np.ndarray],
        H: np.ndarray,
        W: List[np.ndarray],
        lam: np.ndarray,
    ) -> float:
        """Weighted sum of Frobenius reconstruction errors.

        sum_v  lambda(v) * ||X(v) - H W(v)||^2_F
        """
        obj = 0.0
        for X, Wv, lv in zip(views, W, lam):
            diff = X - H @ Wv
            obj += lv * float(np.sum(diff * diff))
        return obj

    def _per_view_errors(
        self,
        views: List[np.ndarray],
        H: np.ndarray,
        W: List[np.ndarray],
    ) -> np.ndarray:
        """Frobenius error per view, unweighted."""
        errs = np.zeros(len(views))
        for v, (X, Wv) in enumerate(zip(views, W)):
            diff = X - H @ Wv
            errs[v] = float(np.sum(diff * diff))
        return errs

    def _update_weights(self, errs: np.ndarray) -> np.ndarray:
        """Adaptive view weights from reconstruction quality.

        lambda(v) proportional to (err(v))^{1/(1-gamma)}.
        Lower error -> higher weight.
        Falls back to equal weights when gamma is near 1 or errs are zero.
        """
        if abs(self.gamma - 1.0) < 1e-10:
            return np.full(len(errs), 1.0 / len(errs))

        exponent = 1.0 / (1.0 - self.gamma)
        safe_errs = np.maximum(errs, self.eps)
        scores = safe_errs**exponent
        total = scores.sum()
        if total <= 0 or not np.isfinite(total):
            return np.full(len(errs), 1.0 / len(errs))
        return scores / total

    def _update_H(
        self,
        views: List[np.ndarray],
        H: np.ndarray,
        W: List[np.ndarray],
        lam: np.ndarray,
    ) -> np.ndarray:
        """Update H via multiplicative rule.

        H <- H * numerator / denominator   (element-wise)

        numerator   = sum_v lam(v) * X(v) W(v)^T       shape (n, k)
        denominator = sum_v lam(v) * H W(v) W(v)^T     shape (n, k)
        """
        numer = np.zeros_like(H)
        denom = np.zeros_like(H)
        for X, Wv, lv in zip(views, W, lam):
            numer += lv * (X @ Wv.T)  # (n,d) @ (d,k) -> (n,k)
            denom += lv * (H @ (Wv @ Wv.T))  # (n,k) @ (k,k) -> (n,k)
        return H * numer / (denom + self.eps)

    def _update_W(
        self,
        X: np.ndarray,
        H: np.ndarray,
        Wv: np.ndarray,
    ) -> np.ndarray:
        """Update W(v) via multiplicative rule.

        W(v) <- W(v) * (H^T X(v)) / (H^T H W(v))   (element-wise)

        shape: (k, d_v)
        """
        HtX = H.T @ X  # (k, d_v)
        HtHW = (H.T @ H) @ Wv  # (k, d_v)
        return Wv * HtX / (HtHW + self.eps)

    def _init_factors(
        self,
        views: List[np.ndarray],
        rng: np.random.RandomState,
    ):
        """Random non-negative initialisation for H and W(v).

        Initialise near the data mean so the factors start with a reasonable magnitude.
        """
        n = self.n_samples_
        k = self.n_components

        # Scale H so HW ~ X on average
        H = rng.rand(n, k) + self.eps

        W = []
        for X in views:
            d = X.shape[1]
            scale = np.sqrt(X.mean() / k) if X.mean() > 0 else 1.0
            W.append(rng.rand(k, d) * scale + self.eps)

        return H, W

    def _run_once(
        self,
        views: List[np.ndarray],
        rng: np.random.RandomState,
    ) -> tuple:
        """One full multiplicative-update run from random init.

        Returns
        -------
        (H, W, lam, objective, n_iter)
        """
        M = self.n_views_in_
        H, W = self._init_factors(views, rng)

        # Uniform weights to start
        lam = np.full(M, 1.0 / M)

        prev_obj = np.inf

        for iteration in range(self.max_iter):
            # Update H (shared across all views)
            H = self._update_H(views, H, W, lam)

            # Update each W(v) independently
            W = [self._update_W(X, H, Wv) for X, Wv in zip(views, W)]

            # Update view weights
            if self.learn_weights:
                errs = self._per_view_errors(views, H, W)
                lam = self._update_weights(errs)

            obj = self._objective(views, H, W, lam)
            if np.isinf(prev_obj):
                rel_change = np.inf
            else:
                rel_change = abs(prev_obj - obj) / (abs(prev_obj) + self.eps)
            if rel_change < self.tol:
                iteration += 1
                break
            prev_obj = obj

        return H, W, lam, obj, iteration + 1

    def fit(self, views: List, y=None) -> "MultiViewNMF":
        """Fit the model.

        Parameters
        ----------
        views : list of array-like of shape (n_samples, n_features_v)
            All values should be non-negative.  Negative values are
            clipped to zero before factorisation with a warning.
        y : ignored

        Returns
        -------
        self
        """
        views = self._validate_views(views, reset=True)

        # Clip negative values — NMF requires non-negative input
        clipped = False
        clean_views = []
        for v, X in enumerate(views):
            if np.any(X < 0):
                clipped = True
                X = np.maximum(X, 0.0)
            clean_views.append(X)

        if clipped:
            import warnings

            warnings.warn(
                "Some views contain negative values. "
                "They have been clipped to zero for NMF. "
                "Consider using a non-negative transformation "
                "(e.g. abs, log1p, or TF-IDF) before fitting.",
                UserWarning,
                stacklevel=2,
            )

        rng = np.random.RandomState(self.random_state)

        best_obj = np.inf
        best_H = None
        best_W = None
        best_lam = None
        best_iter = 0

        for _ in range(self.n_init):
            H, W, lam, obj, n_iter = self._run_once(clean_views, rng)
            if obj < best_obj:
                best_obj = obj
                best_H = H
                best_W = W
                best_lam = lam
                best_iter = n_iter

        self.H_ = best_H
        self.W_ = best_W
        self.weights_ = best_lam
        self.objective_ = best_obj
        self.n_iter_ = best_iter
        self.reconstruction_errors_ = self._per_view_errors(clean_views, best_H, best_W)

        # Hard cluster labels
        self.labels_ = np.argmax(self.H_, axis=1)

        return self

    def transform(self, views: List) -> np.ndarray:
        """Project new samples into the shared H space.

        Solves the NNLS problem for H given fixed W(v):

            min_H  sum_v lambda(v) * ||X(v) - H W(v)||^2_F   s.t. H >= 0

        via the same multiplicative update rule, starting from a
        uniform initialisation, with W(v) held fixed.

        Parameters
        ----------
        views : list of array-like of shape (n_samples_new, n_features_v)

        Returns
        -------
        H_new : ndarray of shape (n_samples_new, n_components)
        """
        check_is_fitted(self, "W_")
        views = self._validate_views(views, reset=False)

        # Clip negatives
        views = [np.maximum(X, 0.0) for X in views]

        n = views[0].shape[0]
        k = self.n_components
        lam = self.weights_

        H = np.ones((n, k)) / k + self.eps

        for _ in range(self.max_iter):
            H_new = self._update_H(views, H, self.W_, lam)
            rel = np.linalg.norm(H_new - H, "fro") / (
                np.linalg.norm(H, "fro") + self.eps
            )
            H = H_new
            if rel < self.tol:
                break

        return H

    def fit_transform(self, views: List, y=None) -> np.ndarray:
        """Fit and return ``H_`` directly (no re-projection needed)."""
        self.fit(views, y)
        return self.H_

    def fit_predict(self, views: List, y=None) -> np.ndarray:
        """Fit and return cluster labels."""
        return self.fit(views, y).labels_
