"""Shared machinery for view imputers."""

from __future__ import annotations

import warnings
from abc import abstractmethod
from typing import List, Optional

import numpy as np

from polyview.base import BaseMultiViewTransformer
from polyview.dataset.multiviewdataset import MultiViewDataset
from polyview.imputation.mask import missing_view_mask


def _nan_column_mean(X: np.ndarray) -> np.ndarray:
    """Column means ignoring ``NaN``; all-``NaN`` columns become 0."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        means = np.nanmean(X, axis=0)
    return np.where(np.isnan(means), 0.0, means)


class BaseViewImputer(BaseMultiViewTransformer):
    """Base class for imputers that fill missing views.

    Missing data is encoded with ``NaN``.  Subclasses implement
    :meth:`_fit_imputer` and :meth:`_transform_views`; this base takes care of
    validation, of the diagnostics stored at fit time, and of the final
    safety net that guarantees a ``NaN``-free output.

    Parameters
    ----------
    n_views : int or None, default=None
        Expected number of views.  ``None`` accepts any count.

    Attributes
    ----------
    n_views_in_ : int
        Number of views seen during :meth:`fit`.
    n_features_in_ : list of int
        Number of features of each view.
    n_samples_ : int
        Number of samples seen during :meth:`fit`.
    missing_view_mask_ : ``ndarray of bool of shape (n_samples, n_views)``
        ``True`` where a whole view was missing at fit time.
    missing_rate_ : ``ndarray of shape (n_views,)``
        Fraction of fit samples for which each view was missing.
    column_fill_ : list of ndarray
        Per-view column means over the observed fit entries, used as a
        last-resort fill value when a view cannot be reconstructed.

    Notes
    -----
    :meth:`transform` never returns ``NaN``: whatever the strategy cannot
    reconstruct (for instance a sample with no observed view at all) falls
    back to ``column_fill_``.
    """

    def __init__(self, n_views: Optional[int] = None) -> None:
        super().__init__(n_views=n_views)

    # ------------------------------------------------------------------
    # Subclass hooks
    # ------------------------------------------------------------------
    @abstractmethod
    def _fit_imputer(self, views: List[np.ndarray]) -> None:
        """Learn whatever the strategy needs from the (incomplete) fit views."""

    @abstractmethod
    def _transform_views(self, views: List[np.ndarray]) -> List[np.ndarray]:
        """Return the views with missing entries filled in.

        The input is already validated and copied, so implementations may
        write into it.  Leftover ``NaN`` values are tolerated here: the
        caller fills them from ``column_fill_``.
        """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit(self, views: List, y=None) -> "BaseViewImputer":
        """Learn the imputation model from (possibly incomplete) views.

        Parameters
        ----------
        views : list of array-like or MultiViewDataset
            Views of shape ``(n_samples, n_features_i)``, with missing
            values encoded as ``NaN``.
        y : ignored

        Returns
        -------
        self
        """
        validated = self._validate_views(views, reset=True)
        self.missing_view_mask_ = missing_view_mask(validated, how="all")
        self.missing_rate_ = self.missing_view_mask_.mean(axis=0)
        self.column_fill_ = [_nan_column_mean(v) for v in validated]
        self._fit_imputer(validated)
        return self

    def transform(self, views: List) -> List[np.ndarray]:
        """Fill the missing entries of ``views``.

        Parameters
        ----------
        views : list of array-like or MultiViewDataset
            Views with the same number of features as seen during
            :meth:`fit`, with missing values encoded as ``NaN``.

        Returns
        -------
        list of ndarray, or MultiViewDataset
            The completed views.  A :class:`MultiViewDataset` in gives a
            :class:`MultiViewDataset` out, keeping labels and view names.
        """
        validated = self._validate_views(views, reset=False)
        filled = self._transform_views([v.copy() for v in validated])
        filled = [self._fill_remaining(v, i) for i, v in enumerate(filled)]
        return self._wrap_like(views, filled)

    def fit_transform(self, views: List, y=None) -> List[np.ndarray]:
        """Fit on ``views`` and return them completed.

        Parameters
        ----------
        views : list of array-like or MultiViewDataset
        y : ignored

        Returns
        -------
        list of ndarray, or MultiViewDataset
        """
        return self.fit(views, y).transform(views)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _fill_remaining(self, X: np.ndarray, view_idx: int) -> np.ndarray:
        """Replace any ``NaN`` the strategy left behind by the column mean."""
        still_missing = np.isnan(X)
        if still_missing.any():
            fill = np.broadcast_to(self.column_fill_[view_idx], X.shape)
            X[still_missing] = fill[still_missing]
        return X

    @staticmethod
    def _wrap_like(original, filled: List[np.ndarray]):
        """Return ``filled`` as a dataset when the input was one."""
        if isinstance(original, MultiViewDataset):
            return MultiViewDataset(
                filled,
                labels=original.labels,
                view_names=original.view_names,
            )
        return filled
