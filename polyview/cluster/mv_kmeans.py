from __future__ import annotations

import warnings
from typing import List, Optional

import numpy as np
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning

from polyview.base import BaseMultiViewClusterer


class MultiviewKMeans(BaseMultiViewClusterer):
    r"""Co-regularized multi-view K-Means clustering.

    Runs one K-Means instance per view but couples them through a shared
    soft assignment matrix :math:`H` that is updated after each round of
    per-view fits.  The coupling term penalizes disagreement between views,
    pushing them towards a consensus partition.

    Parameters
    ----------
    n_clusters : int, default=2
        Number of clusters.
    lambda_ : float, default=1.0
        Co-regularization strength.  :math:`0` → independent per-view K-Means.
        Larger values force stronger consensus.
    max_iter : int, default=10
        Maximum number of alternating-optimization rounds.
    n_init : int, default=10
        Number of random restarts for the *initial* (uncoupled) K-Means.
    tol : float, default=1e-4
        Stop early when the fraction of samples that change cluster
        assignment falls below this threshold.
    temperature : float, default=1.0
        Temperature for the soft-assignment softmax.  Lower → harder
        assignments.
    random_state : int or None, default=None
        Seed for reproducibility.
    verbose : bool, default=False

    Attributes
    ----------
    labels_ : ndarray of shape (n_samples,)
        Final consensus cluster labels.
    labels_per_view_ : list of ndarray
        Per-view cluster labels at convergence (may disagree slightly).
    inertia_ : float
        Sum of inertias across all views at convergence.
    n_iter_ : int
        Number of alternating-optimization rounds performed.
    centroids_ : list of ndarray of shape (n_clusters, n_features_i)
        Per-view cluster centroids.


    Algorithm (one iteration)
    -------------------------
    - For each view :math:`v`, build augmented data
        :math:`X_v^{aug} = [X_v \mid \lambda H]`.
    - Run K-Means on each augmented view.
    - Update :math:`H_v = softmax(-dist(X_v, centroids_v) / T)`.
    - Update shared :math:`H = \frac{1}{V} \sum_{v=1}^V H_v`.
    - Repeat until :math:`y` stops changing or ``max_iter`` is reached.

    The augmented-data trick avoids modifying the K-Means objective
    directly — the regularization weight :math:`\lambda` controls how strongly
    views pull each other's assignments.

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
    Kumar, A., Rai, P., & Daumé, H. (2011).
    Co-regularized multi-view spectral clustering.
    Advances in Neural Information Processing Systems, 24.

    """

    def __init__(
        self,
        n_clusters: int = 2,
        lambda_: float = 1.0,
        max_iter: int = 10,
        n_init: int = 10,
        tol: float = 1e-4,
        temperature: float = 1.0,
        random_state: Optional[int] = None,
        verbose: bool = False,
    ) -> None:
        super().__init__()
        self.n_clusters = n_clusters
        self.lambda_ = lambda_
        self.max_iter = max_iter
        self.n_init = n_init
        self.tol = tol
        self.temperature = temperature
        self.random_state = random_state
        self.verbose = verbose

    def _soft_assignments(
        self, X: np.ndarray, centroids: np.ndarray
    ) -> np.ndarray:
        """Compute soft cluster assignments via softmax over neg-distances.

        Parameters
        ----------
        X : ndarray (n_samples, n_features)
        centroids : ndarray (n_clusters, n_features)

        Returns
        -------
        H : ndarray (n_samples, n_clusters)
            Rows sum to 1.
        """

        diffs = X[:, np.newaxis, :] - centroids[np.newaxis, :, :]
        dists = np.sum(diffs ** 2, axis=2)
        logits = -dists / max(self.temperature, 1e-10)
        logits -= logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        return exp / exp.sum(axis=1, keepdims=True)

    def _kmeans_on_view(
        self,
        X: np.ndarray,
        H_shared: np.ndarray,
        *,
        init: str | np.ndarray = "k-means++",
        n_init: int = 1,
    ) -> KMeans:
        """Run K-Means on the augmented view  [X | lambda * H_shared]."""
        aug = np.concatenate(
            [X, self.lambda_ * H_shared], axis=1
        )
        km = KMeans(
            n_clusters=self.n_clusters,
            init=init,
            n_init=n_init,
            max_iter=300,
            random_state=self.random_state,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            km.fit(aug)
        return km

    @staticmethod
    def _hard_labels(H: np.ndarray) -> np.ndarray:
        return np.argmax(H, axis=1)

    def _label_change_fraction(
        self, old: np.ndarray, new: np.ndarray
    ) -> float:
        return np.mean(old != new)

    def fit(self, views: List, y=None) -> "MultiviewKMeans":
        """Fit the model.

        Parameters
        ----------
        views : list of array-like of shape (n_samples, n_features_i)
        y : ignored

        Returns
        -------
        self
        """
        views = self._validate_views(views, reset=True)
        n, k = self.n_samples_, self.n_clusters

        H_shared = np.full((n, k), 1.0 / k)

        km_list: list[KMeans] = []
        H_views: list[np.ndarray] = []

        for v in views:
            km = self._kmeans_on_view(
                v, H_shared, init="k-means++", n_init=self.n_init
            )
            km_list.append(km)
            centroids_orig = km.cluster_centers_[:, : v.shape[1]]
            H_views.append(self._soft_assignments(v, centroids_orig))

        H_shared = np.mean(H_views, axis=0)
        prev_labels = self._hard_labels(H_shared)

        if self.verbose:
            print(f"[MultiviewKMeans] init done, starting co-reg loop")

        for iteration in range(self.max_iter):
            H_views = []
            km_list_new = []

            for v, km_prev in zip(views, km_list):
                centroids_init = km_prev.cluster_centers_[:, : v.shape[1]]
                aug_centroids = np.concatenate(
                    [centroids_init, self.lambda_ * H_shared[: k]], axis=1
                )
                km = self._kmeans_on_view(
                    v, H_shared, init="k-means++", n_init=1
                )
                km_list_new.append(km)
                centroids_orig = km.cluster_centers_[:, : v.shape[1]]
                H_views.append(self._soft_assignments(v, centroids_orig))

            km_list = km_list_new
            H_shared = np.mean(H_views, axis=0)
            curr_labels = self._hard_labels(H_shared)

            change = self._label_change_fraction(prev_labels, curr_labels)
            if self.verbose:
                print(
                    f"[MultiviewKMeans] iter {iteration + 1:3d} | "
                    f"label change: {change:.4f}"
                )

            if change < self.tol:
                if self.verbose:
                    print(
                        f"[MultiviewKMeans] converged at iter {iteration + 1}"
                    )
                break

            prev_labels = curr_labels

        self.n_iter_ = iteration + 1
        self.labels_ = curr_labels
        self.labels_per_view_ = [
            km.labels_[: n] for km in km_list
        ]
        self.centroids_ = [
            km.cluster_centers_[:, : v.shape[1]]
            for km, v in zip(km_list, views)
        ]
        self.inertia_ = sum(km.inertia_ for km in km_list)

        return self

    def predict(self, views: List) -> np.ndarray:
        """Assign cluster labels to new samples.

        Uses the nearest centroid in each view and takes a majority vote
        across views.

        Parameters
        ----------
        views : list of array-like

        Returns
        -------
        labels : ndarray of shape (n_samples,)
        """
        self._check_is_fitted()
        views = self._validate_views(views, reset=False)

        per_view_labels = []
        for v, centroids in zip(views, self.centroids_):
            diffs = v[:, np.newaxis, :] - centroids[np.newaxis, :, :]
            dists = np.sum(diffs ** 2, axis=2)
            per_view_labels.append(np.argmin(dists, axis=1))

        label_matrix = np.stack(per_view_labels, axis=1)  # (n, n_views)
        labels = np.apply_along_axis(
            lambda row: np.bincount(row, minlength=self.n_clusters).argmax(),
            axis=1,
            arr=label_matrix,
        )
        return labels

    def fit_predict(self, views: List, y=None) -> np.ndarray:
        """Fit and return cluster labels (uses training assignments)."""
        return self.fit(views, y).labels_