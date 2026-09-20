"""Regression-based imputation of a view from the other views."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.base import clone
from sklearn.linear_model import Ridge

from polyview.imputation._base import BaseViewImputer, _nan_column_mean


__all__ = ["CrossViewRegressionImputer"]

_EPS = 1e-12


class CrossViewRegressionImputer(BaseViewImputer):
    """Predict each missing view from the views that are available.

    One multi-output regressor is fitted per view, mapping the concatenation
    of all *other* views onto it, using the samples for which the target view
    is observed.  At transform time, samples missing that view get the
    model's prediction.  Where :class:`KNNViewImputer` copies real samples,
    this learns an explicit view-to-view mapping, which is usually the
    stronger choice when the views are close to linearly related and the
    number of complete samples is small.

    The predictor block is mean-filled and standardised before the fit, so a
    sample missing two views out of three can still be reconstructed.

    Parameters
    ----------
    estimator : sklearn regressor or None, default=None
        Multi-output regressor, cloned once per view.  ``None`` uses
        :class:`sklearn.linear_model.Ridge` with ``alpha=1.0``, whose
        regularisation matters here because neighbouring views are often
        collinear.
    estimator_params : dict or None, default=None
        Extra parameters set on each clone, e.g. ``{"alpha": 10.0}``.
    n_views : int or None, default=None
        Expected number of views.  ``None`` accepts any count.

    Attributes
    ----------
    estimators_ : list of estimator or None
        Fitted regressor per view; ``None`` for views that could not be
        modelled (see Notes).
    n_train_samples_ : ``ndarray of shape (n_views,)``
        Number of samples each view's regressor was fitted on.
    missing_view_mask_ : ``ndarray of bool of shape (n_samples, n_views)``
        ``True`` where a whole view was missing at fit time.
    missing_rate_ : ``ndarray of shape (n_views,)``
        Fraction of fit samples for which each view was missing.

    Notes
    -----
    A view's regressor is skipped, and the column mean used instead, when
    there is a single view in total or when fewer than two fit samples have
    that view observed.

    Examples
    --------
    >>> import numpy as np
    >>> from polyview.imputation import CrossViewRegressionImputer
    >>> rng = np.random.default_rng(0)
    >>> X1 = rng.normal(size=(50, 3))
    >>> X2 = X1 @ rng.normal(size=(3, 2))      # view 2 is a function of view 1
    >>> X2[:5] = np.nan                        # five samples lost view 2
    >>> filled = CrossViewRegressionImputer().fit_transform([X1, X2])
    >>> bool(np.isnan(filled[1]).any())
    False
    """

    def __init__(
        self,
        estimator: Optional[Any] = None,
        estimator_params: Optional[Dict[str, Any]] = None,
        n_views: Optional[int] = None,
    ) -> None:
        super().__init__(n_views=n_views)
        self.estimator = estimator
        self.estimator_params = estimator_params

    # ------------------------------------------------------------------
    def _make_estimator(self) -> Any:
        base = Ridge(alpha=1.0) if self.estimator is None else self.estimator
        estimator = clone(base)
        if self.estimator_params:
            estimator.set_params(**self.estimator_params)
        return estimator

    def _fit_imputer(self, views: List[np.ndarray]) -> None:
        # Predictors must be complete: mean-fill them with fit-time statistics.
        self._predictor_fill_ = [_nan_column_mean(v) for v in views]
        complete = [self._mean_filled(v, i) for i, v in enumerate(views)]

        self._center_, self._scale_ = [], []
        for X in complete:
            center = X.mean(axis=0)
            scale = X.std(axis=0)
            self._center_.append(center)
            self._scale_.append(np.where(scale < _EPS, 1.0, scale))

        self.estimators_ = []
        self.n_train_samples_ = np.zeros(self.n_views_in_, dtype=int)

        for target, X_target in enumerate(views):
            observed = np.flatnonzero(~np.isnan(X_target).any(axis=1))
            self.n_train_samples_[target] = len(observed)

            if self.n_views_in_ < 2 or len(observed) < 2:
                self.estimators_.append(None)
                continue

            predictors = self._predictor_matrix(complete, exclude=target)
            estimator = self._make_estimator()
            estimator.fit(predictors[observed], X_target[observed])
            self.estimators_.append(estimator)

    def _mean_filled(self, X: np.ndarray, view_idx: int) -> np.ndarray:
        """Copy of ``X`` with every ``NaN`` replaced by its fit column mean."""
        filled = X.copy()
        missing = np.isnan(filled)
        if missing.any():
            fill = np.broadcast_to(self._predictor_fill_[view_idx], filled.shape)
            filled[missing] = fill[missing]
        return filled

    def _predictor_matrix(self, complete: List[np.ndarray], exclude: int) -> np.ndarray:
        """Standardised concatenation of every view but ``exclude``."""
        blocks = [
            (X - self._center_[i]) / self._scale_[i]
            for i, X in enumerate(complete)
            if i != exclude
        ]
        return np.concatenate(blocks, axis=1)

    # ------------------------------------------------------------------
    def _transform_views(self, views: List[np.ndarray]) -> List[np.ndarray]:
        complete = [self._mean_filled(v, i) for i, v in enumerate(views)]

        for target, estimator in enumerate(self.estimators_):
            if estimator is None:
                continue
            rows = np.flatnonzero(np.isnan(views[target]).any(axis=1))
            if len(rows) == 0:
                continue

            predictors = self._predictor_matrix(complete, exclude=target)
            predicted = np.asarray(
                estimator.predict(predictors[rows]), dtype=float
            ).reshape(len(rows), -1)

            missing = np.isnan(views[target][rows])
            views[target][rows] = np.where(missing, predicted, views[target][rows])

        return views
