"""Complete-case handling: discard samples instead of imputing them."""

from __future__ import annotations

from typing import List, Tuple, Union

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

from polyview.dataset.multiviewdataset import MultiViewDataset
from polyview.imputation.mask import _as_view_list, missing_view_mask


__all__ = ["DropIncompleteSamples", "drop_missing_views"]


def drop_missing_views(
    views,
    max_missing_views: int = 0,
    how: str = "all",
    return_mask: bool = False,
) -> Union[List[np.ndarray], Tuple[List[np.ndarray], np.ndarray]]:
    """Keep only the samples that have enough observed views.

    Complete-case analysis is the honest baseline for missing views: it
    introduces no synthetic data, at the cost of sample size and of a bias
    whenever the views are not missing completely at random.

    Parameters
    ----------
    views : list of array-like or MultiViewDataset
    max_missing_views : int, default=0
        Maximum number of missing views a sample may have and still be kept.
        ``0`` keeps only fully observed samples.
    how : {"all", "any"}, default="all"
        Passed to :func:`~polyview.imputation.missing_view_mask`: whether a
        view counts as missing only when the whole row is ``NaN``, or as soon
        as one entry is.
    return_mask : bool, default=False
        Also return the boolean mask of the kept samples.

    Returns
    -------
    views_out : list of ndarray
        The views restricted to the kept samples.
    support : ndarray of bool of shape (n_samples,)
        Only when ``return_mask=True``.  ``True`` for kept samples.

    Examples
    --------
    >>> import numpy as np
    >>> from polyview.imputation import drop_missing_views
    >>> X1 = np.array([[1.0], [np.nan], [3.0]])
    >>> X2 = np.array([[4.0], [5.0], [6.0]])
    >>> kept, support = drop_missing_views([X1, X2], return_mask=True)
    >>> support
    array([ True, False,  True])
    >>> kept[0].ravel()
    array([1., 3.])
    """
    arrays = _as_view_list(views)
    if not isinstance(max_missing_views, (int, np.integer)) or max_missing_views < 0:
        raise ValueError(
            f"max_missing_views must be a non-negative integer, got {max_missing_views!r}."
        )
    support = missing_view_mask(arrays, how=how).sum(axis=1) <= max_missing_views
    kept = [v[support] for v in arrays]
    return (kept, support) if return_mask else kept


class DropIncompleteSamples(BaseEstimator):
    """Transformer that drops the samples with too many missing views.

    Parameters
    ----------
    max_missing_views : int, default=0
        Maximum number of missing views a sample may have and still be kept.
    how : {"all", "any"}, default="all"
        Passed to :func:`~polyview.imputation.missing_view_mask`.

    Attributes
    ----------
    support_mask_ : ndarray of bool of shape (n_samples,)
        ``True`` for the samples kept by the last call to :meth:`transform`.
    n_views_in_ : int
        Number of views seen during :meth:`fit`.
    n_features_in_ : list of int
        Number of features of each view.
    n_dropped_ : int
        Number of samples the last :meth:`transform` removed.

    Notes
    -----
    This is the one estimator of the module that changes ``n_samples``.  It
    therefore does not belong in the middle of a supervised
    :class:`~polyview.pipeline.polypipeline.PolyPipeline`, where ``y`` would
    no longer line up with the rows; apply it to the data (and to ``y``)
    beforehand, or use :meth:`support_mask_` to subset ``y`` yourself.  A
    :class:`~polyview.dataset.multiviewdataset.MultiViewDataset` in gives a
    dataset out, with its labels subset for you.

    Examples
    --------
    >>> import numpy as np
    >>> from polyview.imputation import DropIncompleteSamples
    >>> X1 = np.array([[1.0], [np.nan], [3.0]])
    >>> X2 = np.array([[4.0], [5.0], [6.0]])
    >>> dropper = DropIncompleteSamples()
    >>> kept = dropper.fit_transform([X1, X2])
    >>> kept[0].ravel()
    array([1., 3.])
    >>> dropper.n_dropped_
    1
    """

    def __init__(self, max_missing_views: int = 0, how: str = "all") -> None:
        self.max_missing_views = max_missing_views
        self.how = how

    def fit(self, views, y=None) -> "DropIncompleteSamples":
        """Record the shape of the views.  Nothing is learned from the data.

        Parameters
        ----------
        views : list of array-like or MultiViewDataset
        y : ignored

        Returns
        -------
        self
        """
        arrays = _as_view_list(views)
        self.n_views_in_ = len(arrays)
        self.n_features_in_ = [v.shape[1] for v in arrays]
        return self

    def transform(self, views):
        """Return ``views`` without the samples that miss too many views.

        Parameters
        ----------
        views : list of array-like or MultiViewDataset

        Returns
        -------
        list of ndarray, or MultiViewDataset
        """
        check_is_fitted(self, "n_views_in_")
        arrays = _as_view_list(views)
        if len(arrays) != self.n_views_in_:
            raise ValueError(
                f"Fitted on {self.n_views_in_} views but received {len(arrays)}."
            )

        kept, support = drop_missing_views(
            arrays,
            max_missing_views=self.max_missing_views,
            how=self.how,
            return_mask=True,
        )
        self.support_mask_ = support
        self.n_dropped_ = int((~support).sum())

        if isinstance(views, MultiViewDataset):
            labels = views.labels
            return MultiViewDataset(
                kept,
                labels=None if labels is None else labels[support],
                view_names=views.view_names,
            )
        return kept

    def fit_transform(self, views, y=None):
        """Fit and immediately drop.

        Parameters
        ----------
        views : list of array-like or MultiViewDataset
        y : ignored

        Returns
        -------
        list of ndarray, or MultiViewDataset
        """
        return self.fit(views, y).transform(views)
