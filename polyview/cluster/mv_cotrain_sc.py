from ast import List
from typing import List
import numpy as np
from sklearn.metrics import pairwise_kernels
from sklearn.cluster import KMeans
from sklearn.manifold import SpectralEmbedding
from polyview.base import BaseMultiViewClusterer


class MultiViewCoTrainSpectralClustering(BaseMultiViewClusterer):
    """
    Multi-view co-training spectral clustering algorithm.

    Parameters
    ----------
    n_clusters : int, default=2
        The number of clusters to form.
    n_init : int, default=10
        Number of time the k-means algorithm will be run with different centroid seeds.
    max_iter : int, default=50
        Maximum number of iterations of the alternating optimization.
    affinity : str, default='rbf'
        Kernel to use for computing the affinity matrix. Should be a valid metric for sklearn.metrics.pairwise.pairwise_kernels.
    lambda_reg : float, default=1.0
        Regularization parameter for co-training terms.
    random_state : int or None, default=None
        Determines random number generation for centroid initialization. Use an int to make the randomness deterministic.

    Attributes
    ----------
    embedding_ : np.ndarray of shape (n_samples, n_clusters * n_views)
        The concatenated spectral embeddings from all views after fitting.
    objective_ : list of float
        The objective function values at each iteration of the optimization process.
    labels_ : np.ndarray of shape (n_samples,)
        Cluster labels for each sample after fitting.

    References
    ----------
    - Kumar, A., & Daumé, H. (2011). A co-training approach for multi-view spectral clustering.
      In Proceedings of the 28th International Conference on Machine Learning (ICML 2011).
    """

    def __init__(
        self,
        n_clusters=2,
        n_init=10,
        max_iter=50,
        affinity="rbf",
        lambda_reg=1.0,
        random_state=None,
    ) -> None:
        super().__init__()
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.affinity = affinity
        self.lambda_reg = lambda_reg
        self.random_state = random_state

    def _update_spectral_embedding(
        self,
        laplacians: List[np.ndarray],
        embeddings: List[np.ndarray],
        U_consensus: np.ndarray,
        lambda_reg: float,
    ) -> List[np.ndarray]:
        """
        Update spectral embeddings for all views
        """
        for v in range(len(laplacians)):
            L_aug = laplacians[v] + lambda_reg * np.eye(laplacians[v].shape[0])
            B = lambda_reg * U_consensus
            embeddings[v] = np.linalg.solve(L_aug, B)
            embeddings[v] /= np.linalg.norm(embeddings[v], axis=1, keepdims=True) + 1e-8
        U_consensus = np.mean(embeddings, axis=0)
        return embeddings, U_consensus

    def fit(self, views: List[np.ndarray]) -> None:
        """ """
        embeddings = []
        laplacians = []
        for X in views:
            A = pairwise_kernels(X, metric=self.affinity)
            embedding = SpectralEmbedding(
                n_components=self.n_clusters, affinity="precomputed"
            )
            embeddings.append(embedding.fit_transform(A))
            D = np.diag(A.sum(axis=1))
            laplacians.append(D - A)
        U_consensus = np.mean(embeddings, axis=0)

        for it in range(self.max_iter):
            embeddings, U_consensus = self._update_spectral_embedding(
                laplacians, embeddings, U_consensus, self.lambda_reg
            )

        kmeans = KMeans(
            n_clusters=self.n_clusters,
            n_init=self.n_init,
            random_state=self.random_state,
        )
        self.labels_ = kmeans.fit_predict(U_consensus)
        self.embedding_ = U_consensus
        self.objective_ = kmeans.inertia_
        return self

    def fit_predict(self, views: List[np.ndarray], y=None) -> np.ndarray:
        return super().fit_predict(views, y)
