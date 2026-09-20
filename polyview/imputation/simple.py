"""Marginal, per-view imputation of missing views."""

from __future__ import annotations

import warnings
from typing import List, Optional

import numpy as np

from polyview.imputation._base import BaseViewImputer, _nan_column_mean


__all__ = ["SimpleViewImputer"]


class SimpleViewImputer(BaseViewImputer):
    """Fill missing views with a per-view, per-feature constant.

    Each column of each view gets one statistic, computed over the samples
    for which that view is observed; every ``NaN`` in the column is then
    replaced by it.  This is the multi-view counterpart of
    :class:`sklearn.impute.SimpleImputer` and the natural baseline any other
    strategy should be compared against: it ignores the correlations between
    views, so a missing view contributes nothing beyond its own mean.

    Parameters
    ----------
    strategy : {"mean", "median", "constant"}, default="mean"
        Statistic used as fill value.  ``"constant"`` uses ``fill_value``.
    fill_value : float, default=0.0
        Value used when ``strategy="constant"``.
    n_views : int or None, default=None
        Expected number of views.  ``None`` accepts any count.

    Attributes
    ----------
    statistics_ : list of ndarray
        Fill value of each column of each view, shape ``(n_features_i,)``.
    missing_view_mask_ : ``ndarray of bool of shape (n_samples, n_views)``
        ``True`` where a whole view was missing at fit time.
    missing_rate_ : ``ndarray of shape (n_views,)``
        Fraction of fit samples for which each view was missing.

    Notes
    -----
    A column that is ``NaN`` for every fit sample has no statistic to learn;
    it is filled with 0.0 rather than raising, so that a view which is
    missing for the whole training set does not break a pipeline.

    Examples
    --------
    >>> import numpy as np
    >>> from polyview.imputation import SimpleViewImputer
    >>> X1 = np.array([[1.0, 2.0], [3.0, 4.0], [np.nan, np.nan]])
    >>> X2 = np.array([[5.0], [6.0], [7.0]])
    >>> filled = SimpleViewImputer().fit_transform([X1, X2])
    >>> filled[0]
    array([[1., 2.],
           [3., 4.],
           [2., 3.]])
    """

    def __init__(
        self,
        strategy: str = "mean",
        fill_value: float = 0.0,
        n_views: Optional[int] = None,
    ) -> None:
        super().__init__(n_views=n_views)
        self.strategy = strategy
        self.fill_value = fill_value

    def _fit_imputer(self, views: List[np.ndarray]) -> None:
        if self.strategy not in ("mean", "median", "constant"):
            raise ValueError(
                f"strategy must be 'mean', 'median' or 'constant', got {self.strategy!r}."
            )

        if self.strategy == "constant":
            self.statistics_ = [
                np.full(v.shape[1], float(self.fill_value)) for v in views
            ]
            return

        if self.strategy == "mean":
            self.statistics_ = [_nan_column_mean(v) for v in views]
            return

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            medians = [np.nanmedian(v, axis=0) for v in views]
        self.statistics_ = [np.where(np.isnan(m), 0.0, m) for m in medians]

    def _transform_views(self, views: List[np.ndarray]) -> List[np.ndarray]:
        for X, stats in zip(views, self.statistics_):
            missing = np.isnan(X)
            if missing.any():
                X[missing] = np.broadcast_to(stats, X.shape)[missing]
        return views
