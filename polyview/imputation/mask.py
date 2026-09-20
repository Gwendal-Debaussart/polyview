"""Helpers to describe, build and simulate missing views.

Throughout :mod:`polyview.imputation`, missing data is encoded with ``NaN``:

- a **missing entry** is a single ``NaN`` cell inside a view,
- a **missing view** is a sample whose whole row is ``NaN`` in that view,
  which is what happens when a modality (a sensor, a questionnaire, an
  imaging modality) was never acquired for that sample.

The functions below turn views into boolean masks, and back.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import numpy as np


__all__ = [
    "missing_entry_mask",
    "missing_view_mask",
    "complete_case_mask",
    "missing_rate",
    "mark_missing_views",
    "simulate_missing_views",
]


def _as_view_list(views) -> List[np.ndarray]:
    """Coerce views (or a :class:`MultiViewDataset`) to a list of 2-D arrays."""
    if hasattr(views, "views"):
        views = views.views
    if not isinstance(views, (list, tuple)):
        raise TypeError(
            f"views must be a list of array-like objects, got {type(views).__name__}. "
            "Wrap a single array in a list: [X]."
        )
    arrays = []
    for i, v in enumerate(views):
        arr = np.asarray(v, dtype=float)
        if arr.ndim != 2:
            raise ValueError(
                f"View {i} must be 2-D (n_samples x n_features), got shape {arr.shape}."
            )
        arrays.append(arr)
    if len(arrays) == 0:
        raise ValueError("At least one view is required.")
    n = arrays[0].shape[0]
    for i, arr in enumerate(arrays[1:], start=1):
        if arr.shape[0] != n:
            raise ValueError(
                f"All views must have the same number of samples. "
                f"View 0 has {n} samples; view {i} has {arr.shape[0]}."
            )
    return arrays


def missing_entry_mask(views) -> List[np.ndarray]:
    """Return the cell-level missingness mask of every view.

    Parameters
    ----------
    views : list of array-like or MultiViewDataset
        Views of shape ``(n_samples, n_features_i)``.

    Returns
    -------
    list of ndarray of bool
        One mask per view, same shape as the view, ``True`` where the entry
        is ``NaN``.

    Examples
    --------
    >>> import numpy as np
    >>> from polyview.imputation import missing_entry_mask
    >>> X = np.array([[1.0, np.nan], [3.0, 4.0]])
    >>> missing_entry_mask([X])[0]
    array([[False,  True],
           [False, False]])
    """
    return [np.isnan(v) for v in _as_view_list(views)]


def missing_view_mask(views, how: str = "all") -> np.ndarray:
    """Return the sample-level missingness mask across views.

    Parameters
    ----------
    views : list of array-like or MultiViewDataset
        Views of shape ``(n_samples, n_features_i)``.
    how : {"all", "any"}, default="all"
        ``"all"`` marks view ``v`` as missing for sample ``i`` only when the
        whole row is ``NaN`` (a genuinely unobserved view).  ``"any"`` marks
        it as soon as one entry is ``NaN`` (a partially corrupted view).

    Returns
    -------
    ``ndarray of bool of shape (n_samples, n_views)``
        ``True`` where the view is missing for that sample.

    Examples
    --------
    >>> import numpy as np
    >>> from polyview.imputation import missing_view_mask
    >>> X1 = np.array([[1.0, 2.0], [np.nan, np.nan]])
    >>> X2 = np.array([[np.nan, 5.0], [6.0, 7.0]])
    >>> missing_view_mask([X1, X2])
    array([[False, False],
           [ True, False]])
    >>> missing_view_mask([X1, X2], how="any")
    array([[False,  True],
           [ True, False]])
    """
    if how not in ("all", "any"):
        raise ValueError(f"how must be 'all' or 'any', got {how!r}.")
    arrays = _as_view_list(views)
    reduce = np.all if how == "all" else np.any
    columns = [reduce(np.isnan(v), axis=1) for v in arrays]
    return np.column_stack(columns)


def complete_case_mask(views, how: str = "all") -> np.ndarray:
    """Return the samples for which every view is observed.

    Parameters
    ----------
    views : list of array-like or MultiViewDataset
    how : {"all", "any"}, default="all"
        Passed to :func:`missing_view_mask`.

    Returns
    -------
    ndarray of bool of shape (n_samples,)
        ``True`` for samples that have no missing view.

    Examples
    --------
    >>> import numpy as np
    >>> from polyview.imputation import complete_case_mask
    >>> X1 = np.array([[1.0], [np.nan]])
    >>> X2 = np.array([[2.0], [3.0]])
    >>> complete_case_mask([X1, X2])
    array([ True, False])
    """
    return ~missing_view_mask(views, how=how).any(axis=1)


def missing_rate(views, how: str = "all") -> np.ndarray:
    """Return the fraction of samples for which each view is missing.

    Parameters
    ----------
    views : list of array-like or MultiViewDataset
    how : {"all", "any"}, default="all"
        Passed to :func:`missing_view_mask`.

    Returns
    -------
    ``ndarray of shape (n_views,)``
        Value in ``[0, 1]`` per view.

    Examples
    --------
    >>> import numpy as np
    >>> from polyview.imputation import missing_rate
    >>> X1 = np.array([[1.0], [np.nan], [3.0], [np.nan]])
    >>> X2 = np.array([[1.0], [2.0], [3.0], [4.0]])
    >>> missing_rate([X1, X2])
    array([0.5, 0. ])
    """
    return missing_view_mask(views, how=how).mean(axis=0)


def mark_missing_views(views, mask) -> List[np.ndarray]:
    """Blank out views by setting whole rows to ``NaN``.

    This is the inverse of :func:`missing_view_mask`: it turns an explicit
    availability mask into the ``NaN`` encoding the imputers expect.  Use it
    when missingness is recorded separately from the data itself.

    Parameters
    ----------
    views : list of array-like or MultiViewDataset
    mask : ``array-like of bool of shape (n_samples, n_views)``
        ``True`` marks the view as missing for that sample.

    Returns
    -------
    list of ndarray
        Copies of the views with the masked rows set to ``NaN``.

    Examples
    --------
    >>> import numpy as np
    >>> from polyview.imputation import mark_missing_views
    >>> X1 = np.ones((2, 2))
    >>> mark_missing_views([X1], np.array([[False], [True]]))[0]
    array([[ 1.,  1.],
           [nan, nan]])
    """
    arrays = _as_view_list(views)
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError(
            f"mask must be 2-D (n_samples, n_views), got shape {mask.shape}."
        )
    n_samples, n_views = arrays[0].shape[0], len(arrays)
    if mask.shape != (n_samples, n_views):
        raise ValueError(
            f"mask has shape {mask.shape} but views describe "
            f"{(n_samples, n_views)} (n_samples, n_views)."
        )
    out = []
    for v, column in zip(arrays, mask.T):
        marked = v.copy()
        marked[column] = np.nan
        out.append(marked)
    return out


def simulate_missing_views(
    views,
    missing_rate: Union[float, Sequence[float]] = 0.2,
    random_state: Optional[int] = None,
    keep_one: bool = True,
) -> Tuple[List[np.ndarray], np.ndarray]:
    """Remove views completely at random, for benchmarking imputers.

    Each ``(sample, view)`` pair is dropped independently with probability
    ``missing_rate`` (missing-completely-at-random).  The returned mask lets
    you score an imputer against the ground truth you started from.

    Parameters
    ----------
    views : list of array-like or MultiViewDataset
    missing_rate : float or sequence of float, default=0.2
        Probability that a view is missing, shared across views or given per
        view.  Must lie in ``[0, 1]``.
    random_state : int or None, default=None
        Seed for reproducibility.
    keep_one : bool, default=True
        If True, samples that lost every view get one view restored at
        random.  A sample with no observed view carries no information, so
        no imputer can do better than the population mean for it.

    Returns
    -------
    views_out : list of ndarray
        Copies of the views with the dropped rows set to ``NaN``.
    mask : ``ndarray of bool of shape (n_samples, n_views)``
        ``True`` where a view was dropped.

    Examples
    --------
    >>> import numpy as np
    >>> from polyview.imputation import simulate_missing_views
    >>> views = [np.random.rand(100, 4), np.random.rand(100, 6)]
    >>> incomplete, mask = simulate_missing_views(views, 0.3, random_state=0)
    >>> mask.shape
    (100, 2)
    >>> bool(np.all(mask.sum(axis=1) < 2))  # keep_one leaves one view
    True
    """
    arrays = _as_view_list(views)
    n_samples, n_views = arrays[0].shape[0], len(arrays)

    rates = np.broadcast_to(np.asarray(missing_rate, dtype=float), (n_views,))
    if np.any(rates < 0) or np.any(rates > 1):
        raise ValueError(f"missing_rate must lie in [0, 1], got {missing_rate!r}.")

    rng = np.random.default_rng(random_state)
    mask = rng.random((n_samples, n_views)) < rates

    if keep_one:
        empty = mask.all(axis=1)
        if np.any(empty):
            restored = rng.integers(0, n_views, size=int(empty.sum()))
            mask[np.flatnonzero(empty), restored] = False

    return mark_missing_views(arrays, mask), mask
