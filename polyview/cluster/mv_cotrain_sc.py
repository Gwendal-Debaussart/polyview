import warnings
from typing import List

import numpy as np
from sklearn.cluster import KMeans
from sklearn.utils import check_random_state

from polyview.base import BaseMultiViewClusterer
from polyview.cluster._spectral import (
    check_spectral_params,
    compute_affinity,
    normalized_similarity,
    row_normalize,
    subspace_agreement,
    top_eigenvectors,
)


class MultiViewCoTrainSpectralClustering(BaseMultiViewClusterer):
    """
    Multi-view co-training spectral clustering algorithm (Kumar & Daumé, 2011).

    Each view starts from its own spectral embedding ``U_v``: the ``n_clusters``
    leading eigenvectors of the normalised similarity ``D_v^{-1/2} K_v D_v^{-1/2}``.
    At every iteration, the affinity of each view is projected onto the
    eigenspaces found in the other views, which keeps the within-cluster
    similarities the views agree on and removes the rest::

        S_v = sym((sum_{w != v} U_w U_w^T) K_v),   sym(S) = (S + S^T) / 2
        U_v <- leading eigenvectors of D_{S_v}^{-1/2} S_v D_{S_v}^{-1/2}

    All views are updated from the previous iteration's embeddings. The
    row-normalised concatenation of the final ``U_v`` is clustered with k-means.
    The paper states the algorithm for two views; with more views, the
    projections onto the other views' eigenspaces are summed.

    Parameters
    ----------
    n_clusters : int, default=2
        The number of clusters to form.
    n_init : int, default=10
        Number of time the k-means algorithm will be run with different centroid seeds.
    max_iter : int, default=50
        Number of co-training iterations. ``0`` gives the concatenation of the
        per-view spectral embeddings.
    affinity : str or callable, default='rbf'
        Kernel to use for computing the affinity matrix. Should be a valid metric
        for sklearn.metrics.pairwise.pairwise_kernels, and must produce
        non-negative similarities.
    lambda_reg : deprecated
        Has no effect. Kumar & Daumé's co-training has no regularisation
        parameter; it is kept only so that existing code keeps running and
        will be removed in a future release.
    random_state : int, RandomState instance or None, default=None
        Determines random number generation for the eigensolver start vector and
        the k-means centroid initialization. Use an int to make the randomness
        deterministic.
    gamma : float, 'median' or None, default=None
        Kernel coefficient passed to the affinity. ``None`` uses the sklearn
        default (``1 / n_features`` for 'rbf'). ``'median'`` uses the median
        heuristic: ``gamma = 1 / (2 * median(||x_i - x_j||^2))`` for 'rbf'
        (``1 / median(||x_i - x_j||_1)`` for 'laplacian').

    Attributes
    ----------
    embedding_ : ``np.ndarray of shape (n_samples, n_clusters * n_views)``
        Row-normalised concatenation of the co-trained spectral embeddings of all
        views, on which k-means is run.
    objective_ : list of float
        Co-training does not optimise an explicit objective. This records, after
        each iteration, the mean cross-view agreement
        ``tr(U_v U_v^T U_w U_w^T) / n_clusters`` over all pairs of views, which lies
        in ``[0, 1]`` (1 means all views span the same eigenspace). It has
        ``max_iter`` entries.
    labels_ : np.ndarray of shape (n_samples,)
        Cluster labels for each sample after fitting.

    Notes
    -----
    ``S_v`` has rank at most ``2 (n_views - 1) n_clusters``, so its leading
    eigenvectors are computed exactly in that low-dimensional subspace, without
    any ``n_samples x n_samples`` eigendecomposition after the initialisation.

    References
    ----------
    - Kumar, A., & Daumé, H. (2011). A co-training approach for multi-view spectral clustering.
      In Proceedings of the 28th International Conference on Machine Learning (ICML).
    """

    def __init__(
        self,
        n_clusters=2,
        n_init=10,
        max_iter=50,
        affinity="rbf",
        lambda_reg="deprecated",
        random_state=None,
        gamma=None,
    ) -> None:
        super().__init__()
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.affinity = affinity
        self.lambda_reg = lambda_reg
        self.random_state = random_state
        self.gamma = gamma

    @staticmethod
    def _cotrain_embedding(
        K: np.ndarray, W: np.ndarray, k: int, random_state
    ) -> np.ndarray:
        """
        Leading ``k`` eigenvectors of the normalised ``S = sym(W W^T K)``.

        ``W`` stacks the other views' embeddings, so ``W W^T`` is the summed
        projection onto their eigenspaces. With ``B = K W``,
        ``S = (W B^T + B W^T) / 2`` lies in the span of ``[W, B]``; the problem is
        solved exactly in an orthonormal basis of that span.
        """
        B = K @ W
        degrees = (W @ B.sum(axis=0) + B @ W.sum(axis=0)) / 2.0
        d_inv_sqrt = 1.0 / np.sqrt(np.maximum(degrees, 1e-12 * np.abs(degrees).max()))
        Wn = d_inv_sqrt[:, None] * W
        Bn = d_inv_sqrt[:, None] * B

        Q, _ = np.linalg.qr(np.hstack([Wn, Bn]))
        A, C = Q.T @ Wn, Q.T @ Bn
        vals, vecs = np.linalg.eigh((A @ C.T + C @ A.T) / 2.0)
        if Q.shape[1] < Q.shape[0] and vals[-k] < 0:
            # Fewer than k positive eigenvalues: the leading eigenspace reaches
            # into the null space of S, outside span(Q). Solve the full problem.
            S = normalized_similarity((W @ B.T + B @ W.T) / 2.0)
            return top_eigenvectors(S, k, random_state)
        return Q @ vecs[:, -k:]

    @staticmethod
    def _mean_agreement(embeddings: List[np.ndarray]) -> float:
        """Mean of ``tr(U_v U_v^T U_w U_w^T) / k`` over all pairs of views."""
        k = embeddings[0].shape[1]
        pairs = [
            subspace_agreement(embeddings[v], embeddings[w]) / k
            for v in range(len(embeddings))
            for w in range(v + 1, len(embeddings))
        ]
        return float(np.mean(pairs))

    def fit(
        self, views: List[np.ndarray], y=None
    ) -> "MultiViewCoTrainSpectralClustering":
        """
        Fit the co-training spectral clustering model to the provided views.

        Parameters
        ----------
        views : list of np.ndarray
            List of data matrices for each view, where each matrix has shape
            (n_samples, n_features_v). At least two views are required.
        y : ignored

        Returns
        -------
        self
        """
        views = self._validate_views(views, reset=True)
        if self.n_views_in_ < 2:
            raise ValueError(
                f"Co-training requires at least 2 views, got {self.n_views_in_}."
            )
        if not (isinstance(self.lambda_reg, str) and self.lambda_reg == "deprecated"):
            warnings.warn(
                "lambda_reg has no effect: Kumar & Daumé (2011) co-training has no "
                "regularisation parameter. It will be removed in a future release.",
                FutureWarning,
                stacklevel=2,
            )
        k = self.n_clusters
        check_spectral_params(k, self.max_iter, self.n_samples_)
        rng = check_random_state(self.random_state)

        kernels = [compute_affinity(X, self.affinity, self.gamma) for X in views]
        embeddings = [
            top_eigenvectors(normalized_similarity(K), k, rng) for K in kernels
        ]

        agreement = []
        n_views = len(kernels)
        for _ in range(self.max_iter):
            embeddings = [
                self._cotrain_embedding(
                    kernels[v],
                    np.hstack([embeddings[w] for w in range(n_views) if w != v]),
                    k,
                    rng,
                )
                for v in range(n_views)
            ]
            agreement.append(self._mean_agreement(embeddings))

        self.objective_ = agreement
        self.embedding_ = row_normalize(np.hstack(embeddings))
        kmeans = KMeans(
            n_clusters=k,
            n_init=self.n_init,
            random_state=self.random_state,
        )
        self.labels_ = kmeans.fit_predict(self.embedding_)
        return self

    def fit_predict(self, views: List[np.ndarray], y=None) -> np.ndarray:
        return super().fit_predict(views, y)
