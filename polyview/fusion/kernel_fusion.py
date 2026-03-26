"""
Kernel-space fusion via weighted sum of per-view kernel matrices.

Classes / functions
-------------------

center_kernel(K)        remove mean in RKHS
normalize_kernel(K)     scale so K[i,i] = 1 for all i
is_valid_kernel(K)      check symmetry + PSD
KernelSpec              per-view kernel configuration
KernelFusion            fit/transform estimator producing a fused kernel

References
----------

Gönen & Alpaydin (2011). Multiple kernel learning algorithms.
JMLR 12, 2211-2268.
"""

from __future__ import annotations

from typing import Callable, List, Literal, Optional, Union

import numpy as np
from sklearn.metrics.pairwise import (
    linear_kernel as _sk_linear,
    rbf_kernel as _sk_rbf,
    polynomial_kernel as _sk_poly,
    pairwise_distances,
)

from polyview.base import BaseMultiViewTransformer

KernelFn   = Callable[[np.ndarray], np.ndarray]
KernelName = Literal["linear", "rbf", "polynomial", "precomputed"]


def center_kernel(K: np.ndarray) -> np.ndarray:
    """Center a kernel matrix in the RKHS.

    Equivalent to centering the implicit feature map phi(x) so its empirical
    mean is zero.  Strongly recommended before fusing kernels from different
    views — removes the constant bias term and makes kernels comparable.

    K_c = K - 1_n K - K 1_n + 1_n K 1_n

    Parameters
    ----------
    K : ndarray of shape (n_samples, n_samples)

    Returns
    -------
    K_centered : ndarray of shape (n_samples, n_samples), symmetric
    """
    K = np.asarray(K, dtype=float)
    row_mean   = K.mean(axis=1, keepdims=True)
    col_mean   = K.mean(axis=0, keepdims=True)
    grand_mean = K.mean()
    K_c = K - row_mean - col_mean + grand_mean
    return (K_c + K_c.T) / 2.0


def normalize_kernel(K: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Normalize a kernel so that K[i, i] = 1 for all i.

    K_n[i, j] = K[i, j] / sqrt(K[i, i] * K[j, j])

    Prevents views with larger-magnitude kernels from dominating the
    weighted sum in KernelFusion.

    Parameters
    ----------
    K : ndarray of shape (n_samples, n_samples)
    eps : float, default=1e-10
        Guards against zero-norm samples.

    Returns
    -------
    K_normalized : ndarray of shape (n_samples, n_samples), values in [-1, 1]
    """
    K = np.asarray(K, dtype=float)
    diag = np.sqrt(np.maximum(np.diag(K), eps))
    return np.clip(K / np.outer(diag, diag), -1.0, 1.0)


def is_valid_kernel(K: np.ndarray, tol: float = 1e-6) -> bool:
    """Return True if K is a valid symmetric PSD kernel matrix.

    Parameters
    ----------
    K : ndarray
    tol : float
        Tolerance for symmetry check and minimum eigenvalue.

    Returns
    -------
    bool
    """
    K = np.asarray(K, dtype=float)
    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        return False
    if not np.allclose(K, K.T, atol=tol):
        return False
    return bool(np.linalg.eigvalsh(K).min() >= -tol)


class KernelSpec:
    """Configuration for one view's kernel.

    Parameters
    ----------
    kernel : str or callable
        ``"linear"``      — dot-product kernel (sklearn linear_kernel)
        ``"rbf"``         — RBF/Gaussian (sklearn rbf_kernel, median heuristic when gamma=None)
        ``"polynomial"``  — polynomial kernel (sklearn polynomial_kernel)
        ``"precomputed"`` — the view array IS already a kernel matrix
        callable          — any ``(X: ndarray) -> ndarray`` function

    weight : float, default=1.0
        Scalar weight in the weighted sum.  Must be >= 0.

    center : bool, default=True
        Center the kernel in the RKHS before fusion.

    normalize : bool, default=True
        Normalize K so diagonals equal 1 before fusion.

    kernel_params : dict, optional
        Extra kwargs forwarded to the kernel function, e.g.
        ``{"gamma": 0.1}`` for RBF or ``{"degree": 2}`` for polynomial.

    **kwargs
        Shorthand for kernel_params — ``KernelSpec("rbf", gamma=0.5)``
        is equivalent to ``KernelSpec("rbf", kernel_params={"gamma": 0.5})``.

    Examples
    --------
    >>> KernelSpec("rbf")
    >>> KernelSpec("rbf", gamma=0.5)
    >>> KernelSpec("polynomial", kernel_params={"degree": 2, "coef0": 0.0})
    >>> KernelSpec("precomputed", weight=2.0)
    >>> KernelSpec(lambda X: np.tanh(X @ X.T), weight=0.5)
    """

    def __init__(
        self,
        kernel: Union[KernelName, KernelFn] = "rbf",
        weight: float = 1.0,
        center: bool = True,
        normalize: bool = True,
        kernel_params: Optional[dict] = None,
        **kwargs,
    ) -> None:
        if weight < 0:
            raise ValueError(f"weight must be >= 0, got {weight}.")
        self.kernel        = kernel
        self.weight        = weight
        self.center        = center
        self.normalize     = normalize
        self.kernel_params = {**(kernel_params or {}), **kwargs}

    def build(self, X: np.ndarray) -> np.ndarray:
        """Compute (and preprocess) the kernel matrix for view X.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features), or (n_samples, n_samples)
            when ``kernel="precomputed"``.

        Returns
        -------
        K : ndarray of shape (n_samples, n_samples)
        """
        X = np.asarray(X, dtype=float)
        k = self.kernel

        if k == "precomputed":
            if X.ndim != 2 or X.shape[0] != X.shape[1]:
                raise ValueError(
                    "When kernel='precomputed' the view must be a square "
                    f"(n x n) kernel matrix, got shape {X.shape}."
                )
            K = X.copy()

        elif k == "linear":
            K = _sk_linear(X)

        elif k == "rbf":
            gamma = self.kernel_params.get("gamma", None)
            if gamma is None:
                sq_dists  = pairwise_distances(X, metric="sqeuclidean")
                upper     = sq_dists[np.triu_indices_from(sq_dists, k=1)]
                median_sq = np.median(upper)
                gamma     = 1.0 / (2.0 * median_sq) if median_sq > 0 else 1.0
            K = _sk_rbf(X, gamma=gamma)

        elif k == "polynomial":
            K = _sk_poly(
                X,
                degree=self.kernel_params.get("degree", 3),
                gamma=self.kernel_params.get("gamma", None),
                coef0=self.kernel_params.get("coef0", 1.0),
            )

        elif callable(k):
            K = np.asarray(k(X, **self.kernel_params), dtype=float)
            if K.ndim != 2 or K.shape != (X.shape[0], X.shape[0]):
                raise ValueError(
                    f"Custom kernel must return an (n x n) matrix, "
                    f"got shape {K.shape}."
                )

        else:
            raise ValueError(
                f"Unknown kernel {k!r}. Use 'linear', 'rbf', 'polynomial', "
                "'precomputed', or a callable."
            )

        if self.center:
            K = center_kernel(K)
        if self.normalize:
            K = normalize_kernel(K)

        return K

    def __repr__(self) -> str:
        k = self.kernel if isinstance(self.kernel, str) else self.kernel.__name__
        parts = [f"kernel={k!r}", f"weight={self.weight}"]
        if self.kernel_params:
            parts.append(f"params={self.kernel_params}")
        return f"KernelSpec({', '.join(parts)})"


class KernelFusion(BaseMultiViewTransformer):
    """Fuse views via a weighted sum of per-view kernel matrices.

    Each view is mapped to a kernel matrix by its ``KernelSpec``, then:

        K_fused = sum_i (w_i * K_i)          normalize_weights=False
        K_fused = sum_i (w_i / W * K_i)      normalize_weights=True

    The output is a raw (n_samples, n_samples) kernel matrix for use with
    any kernel method: spectral clustering, kernel SVM, kernel PCA, etc.

    Parameters
    ----------
    specs : KernelSpec or list of KernelSpec or None
        One spec per view.  ``None`` or a single spec is broadcast:
        - ``None``         -> RBF, weight=1, center+normalize on each view
        - single KernelSpec -> same spec applied to every view

    normalize_weights : bool, default=False
        Divide weights by their sum (convex combination).

    Attributes
    ----------
    kernels_ : list of ndarray (n_samples, n_samples)
    weights_ : ndarray (n_views,)
    K_fused_ : ndarray (n_samples, n_samples)
    specs_   : list of KernelSpec

    Examples
    --------
    >>> kf = KernelFusion()
    >>> K  = kf.fit_transform([X1, X2])        # RBF on each view

    >>> specs = [KernelSpec("rbf", weight=2.0, gamma=0.1),
    ...          KernelSpec("linear", weight=1.0)]
    >>> K = KernelFusion(specs).fit_transform([X1, X2])

    >>> specs = [KernelSpec("precomputed"), KernelSpec("precomputed")]
    >>> K = KernelFusion(specs).fit_transform([A1, A2])   # A* are n x n
    """

    def __init__(
        self,
        specs: Optional[Union[KernelSpec, List[KernelSpec]]] = None,
        normalize_weights: bool = False,
        n_views: Optional[int] = None,
    ) -> None:
        super().__init__(n_views=n_views)
        self.specs             = specs
        self.normalize_weights = normalize_weights

    def _resolve_specs(self, n_views: int) -> List[KernelSpec]:
        if self.specs is None:
            return [KernelSpec("rbf") for _ in range(n_views)]
        if isinstance(self.specs, KernelSpec):
            return [self.specs for _ in range(n_views)]
        specs = list(self.specs)
        if len(specs) != n_views:
            raise ValueError(
                f"Expected {n_views} KernelSpecs (one per view), "
                f"got {len(specs)}."
            )
        return specs

    def _resolve_weights(self, specs: List[KernelSpec]) -> np.ndarray:
        w = np.array([s.weight for s in specs], dtype=float)
        if self.normalize_weights:
            total = w.sum()
            if total <= 0:
                raise ValueError("All weights are zero — cannot normalize.")
            w = w / total
        return w

    def fit(self, views: List, y=None) -> "KernelFusion":
        """Compute per-view kernels and fuse them.

        Parameters
        ----------
        views : list of array-like
            Feature arrays (n_samples, n_features_i), or square kernel
            matrices (n_samples, n_samples) paired with
            ``KernelSpec("precomputed")``.
        y : ignored

        Returns
        -------
        self
        """
        views = self._validate_views(views, reset=True)

        self.specs_   = self._resolve_specs(self.n_views_in_)
        self.weights_ = self._resolve_weights(self.specs_)
        self.kernels_ = [spec.build(v) for spec, v in zip(self.specs_, views)]

        n = self.n_samples_
        for i, K in enumerate(self.kernels_):
            if K.shape != (n, n):
                raise ValueError(
                    f"Kernel {i} has shape {K.shape}, expected ({n}, {n})."
                )

        self.K_fused_ = sum(w * K for w, K in zip(self.weights_, self.kernels_))
        return self

    def transform(self, views: List) -> np.ndarray:
        """Return the fused kernel for a (possibly new) set of views.

        Parameters
        ----------
        views : list of array-like

        Returns
        -------
        K_fused : ndarray of shape (n_samples_new, n_samples_new)
        """
        self._check_is_fitted()
        views_arr = [np.asarray(v, dtype=float) for v in views]
        if len(views_arr) != self.n_views_in_:
            raise ValueError(
                f"Fitted on {self.n_views_in_} views, got {len(views_arr)}."
            )
        for i, (v, n_feat) in enumerate(zip(views_arr, self.n_features_in_)):
            if self.specs_[i].kernel != "precomputed" and v.shape[1] != n_feat:
                raise ValueError(
                    f"View {i} has {v.shape[1]} features, expected {n_feat}."
                )
        kernels_new = [spec.build(v) for spec, v in zip(self.specs_, views_arr)]
        return sum(w * K for w, K in zip(self.weights_, kernels_new))


    def kernel_matrix(self) -> np.ndarray:
        """Return a copy of the fused kernel matrix."""
        self._check_is_fitted()
        return self.K_fused_.copy()

    def view_contributions(self) -> List[dict]:
        """Per-view contribution fractions to the fused kernel (by Frobenius norm)."""
        self._check_is_fitted()
        norms          = [np.linalg.norm(K, "fro") for K in self.kernels_]
        weighted_norms = [w * n for w, n in zip(self.weights_, norms)]
        total          = sum(weighted_norms) or 1.0
        return [
            {
                "spec":                    spec,
                "weight":                  float(w),
                "kernel_frobenius_norm":   float(n),
                "contribution_fraction":   float(wn / total),
            }
            for spec, w, n, wn in zip(
                self.specs_, self.weights_, norms, weighted_norms
            )
        ]