"""Shared kernel utilities and configuration objects."""

from __future__ import annotations

from typing import Callable, Literal, Optional, Union

import numpy as np
from sklearn.metrics.pairwise import (
    linear_kernel as _sk_linear,
    pairwise_distances,
    polynomial_kernel as _sk_poly,
    rbf_kernel as _sk_rbf,
)

KernelFn = Callable[[np.ndarray], np.ndarray]
KernelName = Literal["linear", "rbf", "polynomial", "precomputed"]


def center_kernel(K: np.ndarray) -> np.ndarray:
    """Center a kernel matrix in the RKHS."""
    K = np.asarray(K, dtype=float)
    row_mean = K.mean(axis=1, keepdims=True)
    col_mean = K.mean(axis=0, keepdims=True)
    grand_mean = K.mean()
    K_c = K - row_mean - col_mean + grand_mean
    return (K_c + K_c.T) / 2.0


def normalize_kernel(K: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Normalize a kernel so that K[i, i] = 1 for all i."""
    K = np.asarray(K, dtype=float)
    diag = np.sqrt(np.maximum(np.diag(K), eps))
    return np.clip(K / np.outer(diag, diag), -1.0, 1.0)


def is_valid_kernel(K: np.ndarray, tol: float = 1e-6) -> bool:
    """Return True if K is a symmetric PSD kernel matrix."""
    K = np.asarray(K, dtype=float)
    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        return False
    if not np.allclose(K, K.T, atol=tol):
        return False
    return bool(np.linalg.eigvalsh(K).min() >= -tol)


class KernelSpec:
    """Configuration for one kernel view."""

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
        self.kernel = kernel
        self.weight = weight
        self.center = center
        self.normalize = normalize
        self.kernel_params = {**(kernel_params or {}), **kwargs}

    def build(self, X: np.ndarray) -> np.ndarray:
        """Compute (and preprocess) the kernel matrix for X."""
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
                sq_dists = pairwise_distances(X, metric="sqeuclidean")
                upper = sq_dists[np.triu_indices_from(sq_dists, k=1)]
                median_sq = np.median(upper)
                gamma = 1.0 / (2.0 * median_sq) if median_sq > 0 else 1.0
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
                    f"Custom kernel must return an (n x n) matrix, got shape {K.shape}."
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
