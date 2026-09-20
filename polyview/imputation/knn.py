"""Nearest-neighbour imputation of missing views."""

from __future__ import annotations

import warnings
from typing import List, Optional, Tuple

import numpy as np

from polyview.imputation._base import BaseViewImputer


__all__ = ["KNNViewImputer"]

_EPS = 1e-12


def _nan_mean_sq_dist(A: np.ndarray, B: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Pairwise squared distance between rows of ``A`` and ``B``, ignoring ``NaN``.

    The squared difference is averaged over the columns observed in *both*
    rows, so pairs that share few columns stay comparable to pairs that share
    many.  Pairs with no column in common get ``NaN``.

    Returns
    -------
    dist : ndarray of shape (len(A), len(B))
    counts : ndarray of shape (len(A), len(B))
        Number of co-observed columns behind each distance.
    """
    observed_a = (~np.isnan(A)).astype(float)
    observed_b = (~np.isnan(B)).astype(float)
    a0 = np.nan_to_num(A, nan=0.0)
    b0 = np.nan_to_num(B, nan=0.0)

    counts = observed_a @ observed_b.T
    sq_sum = (a0**2) @ observed_b.T + observed_a @ (b0**2).T - 2.0 * (a0 @ b0.T)

    with np.errstate(invalid="ignore", divide="ignore"):
        dist = np.where(counts > 0, sq_sum / np.maximum(counts, 1.0), np.nan)
    # Floating-point error can push an exact zero slightly below it.
    return np.maximum(dist, 0.0), counts


class KNNViewImputer(BaseViewImputer):
    """Reconstruct a missing view from the nearest samples that have it.

    For a sample whose view ``v`` is missing, neighbours are searched among
    the fit samples that *do* have view ``v``, using the **other** views as
    the search space; the missing block is then the (weighted) average of
    those neighbours' view ``v``.  Unlike :class:`SimpleViewImputer`, this
    exploits the fact that views of the same sample are correlated: two
    samples that look alike in the views they share usually look alike in
    the view one of them is missing.

    Distances are computed per view on standardised features and averaged
    over the views the two samples have in common, so samples that share
    only one view are still comparable to samples that share three.

    Parameters
    ----------
    n_neighbors : int, default=5
        Number of donors averaged per missing view.  Clipped to the number
        of available donors.
    weights : {"uniform", "distance"}, default="uniform"
        ``"uniform"`` averages the donors equally; ``"distance"`` weights
        them by the inverse of their distance, so closer samples count more.
    n_views : int or None, default=None
        Expected number of views.  ``None`` accepts any count.

    Attributes
    ----------
    missing_view_mask_ : ``ndarray of bool of shape (n_samples, n_views)``
        ``True`` where a whole view was missing at fit time.
    missing_rate_ : ``ndarray of shape (n_views,)``
        Fraction of fit samples for which each view was missing.
    column_fill_ : list of ndarray
        Per-view column means used when no donor can be found.

    Notes
    -----
    Donors for view ``v`` are the fit samples whose view ``v`` is fully
    observed.  When a view has no such sample, or when a sample shares no
    view with any donor, the column mean is used instead.

    With a single view there is nothing left to measure distances with, and
    every sample falls back to the column mean; use
    :class:`sklearn.impute.KNNImputer` for within-view imputation.

    Examples
    --------
    >>> import numpy as np
    >>> from polyview.imputation import KNNViewImputer
    >>> X1 = np.array([[0.0], [0.1], [5.0], [5.1]])
    >>> X2 = np.array([[1.0], [1.2], [9.0], [np.nan]])
    >>> filled = KNNViewImputer(n_neighbors=1).fit_transform([X1, X2])
    >>> filled[1].ravel()          # nearest sample in view 0 is row 2
    array([1. , 1.2, 9. , 9. ])
    """

    def __init__(
        self,
        n_neighbors: int = 5,
        weights: str = "uniform",
        n_views: Optional[int] = None,
    ) -> None:
        super().__init__(n_views=n_views)
        self.n_neighbors = n_neighbors
        self.weights = weights

    # ------------------------------------------------------------------
    def _fit_imputer(self, views: List[np.ndarray]) -> None:
        if not isinstance(self.n_neighbors, (int, np.integer)) or self.n_neighbors < 1:
            raise ValueError(
                f"n_neighbors must be a positive integer, got {self.n_neighbors!r}."
            )
        if self.weights not in ("uniform", "distance"):
            raise ValueError(
                f"weights must be 'uniform' or 'distance', got {self.weights!r}."
            )

        self._fit_views_ = [v.copy() for v in views]
        self._center_, self._scale_ = [], []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for v in views:
                center = np.nanmean(v, axis=0)
                scale = np.nanstd(v, axis=0)
                self._center_.append(np.where(np.isnan(center), 0.0, center))
                self._scale_.append(
                    np.where(np.isnan(scale) | (scale < _EPS), 1.0, scale)
                )

        self._fit_z_ = [self._standardize(v, i) for i, v in enumerate(views)]
        # Fit samples whose view is fully observed: the pool of donors.
        self._donors_ = [np.flatnonzero(~np.isnan(v).any(axis=1)) for v in views]

    def _standardize(self, X: np.ndarray, view_idx: int) -> np.ndarray:
        return (X - self._center_[view_idx]) / self._scale_[view_idx]

    # ------------------------------------------------------------------
    def _transform_views(self, views: List[np.ndarray]) -> List[np.ndarray]:
        z_query = [self._standardize(v, i) for i, v in enumerate(views)]
        per_view = [
            _nan_mean_sq_dist(z_query[i], self._fit_z_[i])
            for i in range(self.n_views_in_)
        ]

        for target in range(self.n_views_in_):
            donors = self._donors_[target]
            rows = np.flatnonzero(np.isnan(views[target]).any(axis=1))
            if len(donors) == 0 or len(rows) == 0:
                continue

            distances = self._combine_distances(per_view, exclude=target)
            block = self._impute_block(
                distances[np.ix_(rows, donors)], self._fit_views_[target][donors]
            )
            if block is None:
                continue

            missing = np.isnan(views[target][rows])
            views[target][rows] = np.where(missing, block, views[target][rows])

        return views

    def _combine_distances(
        self, per_view: List[Tuple[np.ndarray, np.ndarray]], exclude: int
    ) -> np.ndarray:
        """Average the per-view distances over the views two samples share."""
        total = np.zeros_like(per_view[exclude][0])
        shared = np.zeros_like(total)
        for i, (dist, counts) in enumerate(per_view):
            if i == exclude:
                continue
            usable = counts > 0
            total += np.where(usable, np.nan_to_num(dist, nan=0.0), 0.0)
            shared += usable
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(shared > 0, total / np.maximum(shared, 1.0), np.inf)

    def _impute_block(
        self, distances: np.ndarray, donor_values: np.ndarray
    ) -> Optional[np.ndarray]:
        """Average the ``n_neighbors`` closest donors for each row.

        Returns ``None`` when no row has a usable donor, and ``NaN`` rows for
        the individual samples that share no view with any donor; the base
        class fills those from ``column_fill_``.
        """
        n_rows, n_donors = distances.shape
        k = int(min(self.n_neighbors, n_donors))
        if k == 0:
            return None

        nearest = np.argpartition(distances, k - 1, axis=1)[:, :k]
        nearest_dist = np.take_along_axis(distances, nearest, axis=1)
        reachable = np.isfinite(nearest_dist)

        if self.weights == "distance":
            weights = np.where(reachable, 1.0 / (nearest_dist + _EPS), 0.0)
        else:
            weights = reachable.astype(float)

        totals = weights.sum(axis=1, keepdims=True)
        if not np.any(totals > 0):
            return None

        # (n_rows, k, n_features) -> weighted mean over the k donors.
        neighbour_values = donor_values[nearest]
        block = np.einsum("rk,rkf->rf", weights, neighbour_values)
        with np.errstate(invalid="ignore", divide="ignore"):
            block = np.where(totals > 0, block / np.maximum(totals, _EPS), np.nan)
        return block.reshape(n_rows, donor_values.shape[1])
