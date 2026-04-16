"""
References
----------

Gönen & Alpaydin (2011). Multiple kernel learning algorithms.
JMLR 12, 2211-2268.
"""

from __future__ import annotations

from typing import List, Optional, Union

import numpy as np

from polyview.base import BaseMultiViewTransformer
from polyview.utils.kernels import KernelSpec, center_kernel, is_valid_kernel, normalize_kernel


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
        self.specs = specs
        self.normalize_weights = normalize_weights

    def _resolve_specs(self, n_views: int) -> List[KernelSpec]:
        if self.specs is None:
            return [KernelSpec("rbf") for _ in range(n_views)]
        if isinstance(self.specs, KernelSpec):
            return [self.specs for _ in range(n_views)]
        specs = list(self.specs)
        if len(specs) != n_views:
            raise ValueError(
                f"Expected {n_views} KernelSpecs (one per view), " f"got {len(specs)}."
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

        self.specs_ = self._resolve_specs(self.n_views_in_)
        self.weights_ = self._resolve_weights(self.specs_)
        self.kernels_ = [spec.build(v) for spec, v in zip(self.specs_, views)]

        n = self.n_samples_
        for i, K in enumerate(self.kernels_):
            if K.shape != (n, n):
                raise ValueError(
                    f"Kernel {i} has shape {K.shape}, expected ({n}, {n})."
                )

        self.K_fused_ = np.zeros((n, n), dtype=float)
        for w, K in zip(self.weights_, self.kernels_):
            self.K_fused_ += w * K
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
        K_fused = np.zeros((views_arr[0].shape[0], views_arr[0].shape[0]), dtype=float)
        for w, K in zip(self.weights_, kernels_new):
            K_fused += w * K
        return K_fused

    def kernel_matrix(self) -> np.ndarray:
        """Return a copy of the fused kernel matrix."""
        self._check_is_fitted()
        return self.K_fused_.copy()

    def view_contributions(self) -> List[dict]:
        """Per-view contribution fractions to the fused kernel (by Frobenius norm)."""
        self._check_is_fitted()
        norms = [np.linalg.norm(K, "fro") for K in self.kernels_]
        weighted_norms = [w * n for w, n in zip(self.weights_, norms)]
        total = sum(weighted_norms) or 1.0
        return [
            {
                "spec": spec,
                "weight": float(w),
                "kernel_frobenius_norm": float(n),
                "contribution_fraction": float(wn / total),
            }
            for spec, w, n, wn in zip(self.specs_, self.weights_, norms, weighted_norms)
        ]
