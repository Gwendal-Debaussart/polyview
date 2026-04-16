from typing import List

from sklearn.cluster import KMeans
from sklearn.manifold import SpectralEmbedding
import numpy as np
from polyview.base import BaseMultiViewClusterer
from sklearn.metrics.pairwise import pairwise_kernels


class MultiViewCoRegSpectralClustering(BaseMultiViewClusterer):
    """
    Multi-view co-regularized spectral clustering algorithm.

    Parameters
    ----------
    n_clusters : int, default=2
        The number of clusters to form.
    n_init : int, default=10
        Number of time the k-means algorithm will be run with different centroid seeds.
    max_iter : int, default=50
        Maximum number of iterations of the alternating optimization.
    v_lambda : float, default=1.0
        Regularization parameter for co-regularization terms.
    affinity : str, default='rbf'
        Kernel to use for computing the affinity matrix. Should be a valid metric for sklearn.metrics.pairwise.pairwise_kernels.
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

    Notes
    -----
    This implementation follows the approach of co-regularized spectral clustering, where spectral embeddings for each view are learned jointly with a regularization term that encourages the embeddings to be similar across views. The algorithm alternates between updating the spectral embeddings for each view and updating the cluster assignments based on the combined embeddings. The objective function includes both the spectral clustering objective for each view and the co-regularization terms that penalize differences between the embeddings of different views.
    """

    def __init__(
        self,
        n_clusters=2,
        n_init=10,
        max_iter=50,
        v_lambda=1.0,
        affinity="rbf",
        random_state=None,
    ) -> None:
        super().__init__()
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.v_lambda = v_lambda
        self.affinity = affinity
        self.random_state = random_state

    def _compute_graph_laplacians(self, views: List[np.ndarray]) -> List[np.ndarray]:
        """
        Compute unnormalized graph Laplacians for each view using RBF kernel.
        """

        laplacians = []
        for X in views:
            S = pairwise_kernels(X, metric=self.affinity)
            D = np.diag(S.sum(axis=1))
            L = D - S
            laplacians.append(L)
        return laplacians

    def _update_single_embedding(
        self, laplacian: np.ndarray, reg_matrix: np.ndarray
    ) -> np.ndarray:
        """
        Update spectral embedding for a single view given its Laplacian and regularization matrix.
        """
        L_reg = laplacian + self.v_lambda * reg_matrix
        vals, vecs = np.linalg.eigh(L_reg)
        idx = np.argsort(vals)[: self.n_clusters]
        return vecs[:, idx]

    def _objective(
        self, laplacians: List[np.ndarray], embeddings: List[np.ndarray]
    ) -> float:
        """
        Compute the sum of trace(U^T L U) for all views plus co-regularization terms.
        """
        obj = 0.0
        n_views = len(laplacians)
        for v in range(n_views):
            U = embeddings[v]
            obj += np.trace(U.T @ laplacians[v] @ U)
            for w in range(n_views):
                if v != w:
                    obj += (
                        self.v_lambda
                        * np.linalg.norm(
                            U @ U.T - embeddings[w] @ embeddings[w].T, "fro"
                        )
                        ** 2
                    )
        return obj

    def _update_spectral_embedding(
        self, laplacians: List[np.ndarray], embeddings: List[np.ndarray]
    ) -> List[np.ndarray]:
        """
        Updates spectral embeddings for each view based on the current embeddings of other views.
        """
        n_views = len(laplacians)
        n_samples = embeddings[0].shape[0]
        new_embeddings = []
        for v in range(n_views):
            reg = np.zeros((n_samples, n_samples))
            for w in range(n_views):
                if v != w:
                    reg += embeddings[w] @ embeddings[w].T
            reg = (reg + reg.T) / 2
            new_U = self._update_single_embedding(laplacians[v], reg)
            new_embeddings.append(new_U)
        return new_embeddings

    def fit(self, views: list) -> None:
        """
        Fits the multi-view co-regularized spectral clustering model to the provided views.

        Parameters
        ----------
        views : list of np.ndarray
            List of data matrices for each view, where each matrix has shape (n_samples, n_features_v).
        """

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

        obj_vals = []
        for it in range(self.max_iter):
            embeddings = self._update_spectral_embedding(laplacians, embeddings)
            obj_vals.append(self._objective(laplacians, embeddings))
        self.objective_ = obj_vals
        V_mat = np.hstack(embeddings)
        norm_v = np.sqrt(np.diag(V_mat @ V_mat.T))
        norm_v[norm_v == 0] = 1
        self.embedding_ = np.linalg.inv(np.diag(norm_v)) @ V_mat
        kmeans = KMeans(
            n_clusters=self.n_clusters,
            n_init=self.n_init,
            random_state=self.random_state,
        )
        self.labels_ = kmeans.fit_predict(self.embedding_)
        return self

    def fit_predict(self, views: List, y=None) -> np.ndarray:
        return super().fit_predict(views, y)
