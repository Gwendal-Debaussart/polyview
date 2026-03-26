"""
Early fusion strategies — combine views *before* any learning.

Classes
-------
ConcatFusion       horizontal concatenation of all views
WeightedFusion     per-view weighted concatenation
NormalizedFusion   z-score each view before concatenating
WeightedSumFusion  per-view weighted sum of normalized views, after a kernel transformation
WeightedProductFusion  per-view weighted product of normalized views, after a kernel transformation
"""

from __future__ import annotations
from typing import List, Optional
import numpy as np
from polyview.base import BaseMultiViewTransformer


class ConcatFusion(BaseMultiViewTransformer):
    """Fuse views by horizontal concatenation.

    Output shape is
    ``(n_samples, sum(n_features_i))``.

    Parameters
    ----------
    n_views : int or None, default=None
        Expected number of views.  ``None`` accepts any count.

    Attributes
    ----------
    n_views_in_ : int
    n_features_in_ : list of int
    n_features_out_ : int
        Total number of output features after concatenation.

    Examples
    --------
    >>> import numpy as np
    >>> from polyview.fusion.early import ConcatFusion
    >>> X1 = np.random.rand(50, 4)
    >>> X2 = np.random.rand(50, 6)
    >>> fused = ConcatFusion().fit_transform([X1, X2])
    >>> fused.shape
    (50, 10)
    """

    def fit(self, views: List, y=None) -> "ConcatFusion":
        views = self._validate_views(views, reset=True)
        self.n_features_out_ = sum(v.shape[1] for v in views)
        return self

    def transform(self, views: List) -> np.ndarray:
        views = self._validate_views(views, reset=False)
        return np.concatenate(views, axis=1)


class WeightedFusion(BaseMultiViewTransformer):
    """Fuse views by weighted concatenation.

    Each view is scaled by a scalar weight before concatenation.
    Useful when you have prior knowledge that some views are more
    informative or reliable than others.

    Parameters
    ----------
    weights : list of float or None, default=None
        One weight per view.  Weights do not need to sum to 1 — they are
        applied as-is (``view_i * weight_i``).  When ``None``, all weights
        are set to 1.0 (equivalent to ``ConcatFusion``).
    n_views : int or None, default=None

    Attributes
    ----------
    weights_ : ndarray of shape (n_views,)
        The weights actually applied (after length validation).

    Examples
    --------
    >>> fused = WeightedFusion(weights=[1.0, 0.5]).fit_transform([X1, X2])
    """

    def __init__(
        self,
        weights: Optional[List[float]] = None,
        n_views: Optional[int] = None,
    ) -> None:
        super().__init__(n_views=n_views)
        self.weights = weights

    def fit(self, views: List, y=None) -> "WeightedFusion":
        views = self._validate_views(views, reset=True)

        if self.weights is None:
            self.weights_ = np.ones(self.n_views_in_)
        else:
            w = np.asarray(self.weights, dtype=float)
            if w.ndim != 1 or len(w) != self.n_views_in_:
                raise ValueError(
                    f"weights must have one entry per view "
                    f"({self.n_views_in_}), got {len(w)}."
                )
            self.weights_ = w

        self.n_features_out_ = sum(v.shape[1] for v in views)
        return self

    def transform(self, views: List) -> np.ndarray:
        views = self._validate_views(views, reset=False)
        scaled = [v * w for v, w in zip(views, self.weights_)]
        return np.concatenate(scaled, axis=1)


class NormalizedFusion(BaseMultiViewTransformer):
    """Fuse views after per-view z-score normalization.

    Each view is standardized (zero mean, unit variance per feature)
    *independently* before concatenation.  This prevents views with
    larger numeric scales from dominating the fused representation —
    almost always a better default than plain ``ConcatFusion``.

    Parameters
    ----------
    weights : list of float or None, default=None
        Optional per-view scalar weights applied *after* normalization.
    eps : float, default=1e-8
        Small constant added to the denominator to avoid division by zero
        for constant features.
    n_views : int or None, default=None

    Attributes
    ----------
    means_ : list of ndarray of shape (n_features_i,)
        Per-feature means computed during ``fit``.
    stds_ : list of ndarray of shape (n_features_i,)
        Per-feature standard deviations (clipped to ``eps``).
    weights_ : ndarray of shape (n_views,)

    Examples
    --------
    >>> fused = NormalizedFusion().fit_transform([X1, X2])
    >>> fused.mean(axis=0)          # ≈ 0 for every feature
    >>> fused.std(axis=0)           # ≈ 1 for every feature
    """

    def __init__(
        self,
        weights: Optional[List[float]] = None,
        eps: float = 1e-8,
        n_views: Optional[int] = None,
    ) -> None:
        super().__init__(n_views=n_views)
        self.weights = weights
        self.eps = eps

    def fit(self, views: List, y=None) -> "NormalizedFusion":
        views = self._validate_views(views, reset=True)

        self.means_ = [v.mean(axis=0) for v in views]
        self.stds_  = [np.clip(v.std(axis=0), self.eps, None) for v in views]

        if self.weights is None:
            self.weights_ = np.ones(self.n_views_in_)
        else:
            w = np.asarray(self.weights, dtype=float)
            if len(w) != self.n_views_in_:
                raise ValueError(
                    f"weights must have one entry per view "
                    f"({self.n_views_in_}), got {len(w)}."
                )
            self.weights_ = w

        self.n_features_out_ = sum(v.shape[1] for v in views)
        return self

    def _normalize(self, views: List[np.ndarray]) -> List[np.ndarray]:
        return [
            (v - mu) / std
            for v, mu, std in zip(views, self.means_, self.stds_)
        ]

    def transform(self, views: List) -> np.ndarray:
        views = self._validate_views(views, reset=False)
        normed = self._normalize(views)
        scaled = [v * w for v, w in zip(normed, self.weights_)]
        return np.concatenate(scaled, axis=1)