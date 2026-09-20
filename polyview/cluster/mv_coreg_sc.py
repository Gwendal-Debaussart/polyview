from typing import List

from sklearn.cluster import KMeans
from sklearn.manifold import SpectralEmbedding
import numpy as np
from polyview.base import BaseMultiViewClusterer
from polyview.utils.linalg import truncated_eigh
from scipy.sparse.linalg import LinearOperator
from polyview.cluster._spectral import check_spectral_params, compute_affinity


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
    v_lambda : float, default=0.01
        Weight of the agreement term between the embeddings of different views. Its suitable scale depends on ``laplacian``: values around 0.01 suit the normalized Laplacian, whose eigenvalues lie in [0, 2], whereas the unnormalized Laplacian scales with the node degrees.
    affinity : str, default='rbf'
        Kernel to use for computing the affinity matrix. Should be a valid metric for sklearn.metrics.pairwise.pairwise_kernels.
    eigen_solver : {'auto', 'dense', 'arpack'}, default='auto'
        Solver for the ``n_clusters`` smallest eigenvectors computed at each iteration. The co-regularized Laplacians are applied matrix-free, without forming the (n, n) co-regularization terms. 'dense' materializes them and uses LAPACK restricted to the requested eigenpairs; 'arpack' uses an iterative Lanczos solver warm-started from the previous embedding. 'auto' picks 'arpack' when ``n_samples > 200``, 'dense' otherwise.
    random_state : int or None, default=None
        Determines random number generation for centroid initialization. Use an int to make the randomness deterministic.
    tol : float, default=1e-6
        Stop when the change of the objective between two iterations is below ``tol`` times its initial value.
    laplacian : {"normalized", "unnormalized"}, default="normalized"
        Graph Laplacian of each view: the symmetric normalized Laplacian I - D^{-1/2} A D^{-1/2} used by Kumar et al. (2011), or the unnormalized Laplacian D - A. The eigenvectors of the unnormalized Laplacian can concentrate on a few low-degree samples, which degrades the clustering on some datasets.
    gamma : float, 'median' or None, default=None
        Kernel coefficient passed to the affinity. ``None`` uses the sklearn
        default (``1 / n_features`` for 'rbf'). ``'median'`` uses the median
        heuristic: ``gamma = 1 / (2 * median(||x_i - x_j||^2))`` for 'rbf'
        (``1 / median(||x_i - x_j||_1)`` for 'laplacian').

    Attributes
    ----------
    embedding_ : ``np.ndarray of shape (n_samples, n_clusters * n_views)``
        The concatenated spectral embeddings from all views after fitting.
    view_embeddings_ : list of np.ndarray of shape (n_samples, n_clusters)
        The orthonormal spectral embedding ``U_v`` learned for each view.
    objective_ : list of float
        The objective function values at each iteration of the optimization process.
    n_iter_ : int
        Number of iterations run.
    labels_ : np.ndarray of shape (n_samples,)
        Cluster labels for each sample after fitting.

    Notes
    -----
    Each view v has a spectral embedding U_v with orthonormal columns. Following the pairwise co-regularization of Kumar et al. (2011), the method minimizes

    J = sum_v tr(U_v^T L_v U_v) - v_lambda * sum_{v<w} tr(U_v U_v^T U_w U_w^T),

    i.e. the spectral clustering objective of each view minus the agreement between the embeddings of each pair of views, which is maximal when they span the same subspace. The views are updated one at a time: given the other embeddings, the optimal U_v is formed by the eigenvectors of L_v - v_lambda * sum_{w != v} U_w U_w^T with the smallest eigenvalues, so that J never increases. The final clusters are obtained by k-means on the row-normalized concatenation of the embeddings. By default, L_v = I - D_v^{-1/2} A_v D_v^{-1/2} is the symmetric normalized Laplacian of the affinity matrix A_v of view v, as in Kumar et al.; the unnormalized Laplacian D_v - A_v is available with ``laplacian="unnormalized"``.

    References
    ----------
    - Kumar, A. et al. (2011). Co-regularized Multi-view Spectral Clustering.
      Advances in Neural Information Processing Systems 24.
    """

    def __init__(
        self,
        n_clusters=2,
        n_init=10,
        max_iter=50,
        v_lambda=0.01,
        affinity="rbf",
        eigen_solver="auto",
        random_state=None,
        tol=1e-6,
        laplacian="normalized",
        gamma=None,
    ) -> None:
        super().__init__()
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.v_lambda = v_lambda
        self.affinity = affinity
        self.eigen_solver = eigen_solver
        self.random_state = random_state
        self.tol = tol
        self.laplacian = laplacian
        self.gamma = gamma

    def _graph_laplacian(self, A: np.ndarray) -> np.ndarray:
        """Graph Laplacian of an affinity matrix, as selected by ``laplacian``."""
        if self.laplacian == "unnormalized":
            return np.diag(A.sum(axis=1)) - A
        if self.laplacian == "normalized":
            A = A.copy()
            np.fill_diagonal(A, 0.0)
            d_inv_sqrt = 1.0 / np.sqrt(np.maximum(A.sum(axis=1), np.finfo(float).tiny))
            return np.eye(A.shape[0]) - d_inv_sqrt[:, None] * A * d_inv_sqrt[None, :]
        raise ValueError(
            f"laplacian must be 'normalized' or 'unnormalized', got {self.laplacian!r}."
        )

    def _compute_graph_laplacians(self, views: List[np.ndarray]) -> List[np.ndarray]:
        """
        Compute the graph Laplacian of each view from its affinity matrix.
        """
        return [
            self._graph_laplacian(compute_affinity(X, self.affinity, self.gamma))
            for X in views
        ]

    def _resolve_eigen_solver(self, n_samples: int) -> str:
        # The co-regularized Laplacians are applied matrix-free, so Lanczos
        # only needs cheap products whatever the number of clusters.
        if self.eigen_solver == "auto":
            return (
                "arpack"
                if n_samples > 200 and self.n_clusters < n_samples - 1
                else "dense"
            )
        return self.eigen_solver

    def _regularized_laplacian(
        self, laplacian: np.ndarray, others: List[np.ndarray]
    ) -> LinearOperator:
        """L - v_lambda * sum_w U_w U_w^T, without forming the (n, n) sum."""
        n = laplacian.shape[0]

        def matmat(X: np.ndarray) -> np.ndarray:
            X = X.reshape(n, -1)
            out = laplacian @ X
            for U in others:
                out -= self.v_lambda * (U @ (U.T @ X))
            return out

        return LinearOperator(
            (n, n), matvec=matmat, rmatvec=matmat, matmat=matmat, dtype=float
        )

    def _update_single_embedding(
        self,
        laplacian: np.ndarray,
        others: List[np.ndarray],
        previous: np.ndarray,
        solver: str,
    ) -> np.ndarray:
        """
        Update spectral embedding for a single view given its Laplacian and the embeddings of the other views.
        """
        # Warm-start Lanczos from the previous embedding of this view
        v0 = previous.sum(axis=1)
        _, vecs = truncated_eigh(
            self._regularized_laplacian(laplacian, others),
            self.n_clusters,
            largest=False,
            eigen_solver=solver,
            random_state=self.random_state,
            v0=v0 if np.linalg.norm(v0) > 0 else None,
        )
        return vecs

    def _objective(
        self, laplacians: List[np.ndarray], embeddings: List[np.ndarray]
    ) -> float:
        """
        Compute sum_v tr(U_v^T L_v U_v) - v_lambda * sum_{v<w} tr(U_v U_v^T U_w U_w^T).
        """
        obj = sum(np.trace(U.T @ L @ U) for U, L in zip(embeddings, laplacians))
        n_views = len(laplacians)
        for v in range(n_views):
            for w in range(v + 1, n_views):
                # tr(U_v U_v^T U_w U_w^T) = ||U_v^T U_w||_F^2, without (n, n) matrices
                obj -= (
                    self.v_lambda * np.linalg.norm(embeddings[v].T @ embeddings[w]) ** 2
                )
        return obj

    def _update_spectral_embedding(
        self, laplacians: List[np.ndarray], embeddings: List[np.ndarray], solver: str
    ) -> List[np.ndarray]:
        """
        Update the spectral embedding of each view in turn, given the latest embeddings of the other views.
        """
        embeddings = list(embeddings)
        n_views = len(laplacians)
        for v in range(n_views):
            embeddings[v] = self._update_single_embedding(
                laplacians[v],
                [embeddings[w] for w in range(n_views) if w != v],
                embeddings[v],
                solver,
            )
        return embeddings

    def fit(self, views: List[np.ndarray], y=None) -> None:
        """
        Fits the multi-view co-regularized spectral clustering model to the provided views.

        Parameters
        ----------
        views : list of np.ndarray
            List of data matrices for each view, where each matrix has shape (n_samples, n_features_v).
        """
        views = self._validate_views(views, reset=True)
        check_spectral_params(self.n_clusters, self.max_iter, self.n_samples_)
        if self.v_lambda < 0:
            raise ValueError(f"v_lambda must be non-negative, got {self.v_lambda}.")

        embeddings = []
        laplacians = []
        for X in views:
            A = compute_affinity(X, self.affinity, self.gamma)
            embedding = SpectralEmbedding(
                n_components=self.n_clusters, affinity="precomputed"
            )
            embeddings.append(embedding.fit_transform(A))
            laplacians.append(self._graph_laplacian(A))

        solver = self._resolve_eigen_solver(self.n_samples_)
        obj_vals = []
        for it in range(self.max_iter):
            embeddings = self._update_spectral_embedding(laplacians, embeddings, solver)
            obj_vals.append(self._objective(laplacians, embeddings))
            # Measured against the initial objective, which stays meaningful when
            # the objective itself converges to (numerically) zero
            if it > 0 and abs(obj_vals[-2] - obj_vals[-1]) < self.tol * abs(
                obj_vals[0]
            ):
                break
        self.objective_ = obj_vals
        self.n_iter_ = len(obj_vals)
        self.view_embeddings_ = embeddings
        V_mat = np.hstack(embeddings)
        norm_v = np.linalg.norm(V_mat, axis=1)
        norm_v[norm_v == 0] = 1
        self.embedding_ = V_mat / norm_v[:, None]
        kmeans = KMeans(
            n_clusters=self.n_clusters,
            n_init=self.n_init,
            random_state=self.random_state,
        )
        self.labels_ = kmeans.fit_predict(self.embedding_)
        return self

    def fit_predict(self, views: List[np.ndarray], y=None) -> np.ndarray:
        return super().fit_predict(views, y)
