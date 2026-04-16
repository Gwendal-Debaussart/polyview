"""
Base classes for all multi-view estimators.
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import Iterable, List, Optional

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.utils.validation import check_is_fitted


class BaseMultiView(BaseEstimator, ABC):
    """Base class for all polyview estimators.

    Inheriting from sklearn's ``BaseEstimator`` gives every subclass:
    - ``get_params()`` / ``set_params()``  (introspection, Grid Search)
    - ``__repr__``                          (human-readable)
    - ``clone()`` compatibility             (cross-validation)

    Subclasses must implement :meth:`fit`.  They should *not* override
    ``get_params`` / ``set_params`` — those are auto-generated from the
    ``__init__`` signature by sklearn.

    Parameters
    ----------
    n_views : int or None, default=None
        Expected number of views.  When set, :meth:`_validate_views`
        raises if the data has a different count.  ``None`` means
        "accept any number of views ≥ 1".

    Notes
    -----
    Design contract for every subclass
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    * ``__init__`` stores hyperparameters only — no data, no fitting.
    * Every hyperparameter must be a keyword argument with a default.
    * Learned attributes are written as ``attr_`` (trailing underscore),
      following sklearn convention, so ``check_is_fitted`` works.
    * Views are always passed as ``list[np.ndarray]`` (or a
      ``MultiViewDataset``), *never* as a single array.
    """

    def __init__(self, n_views: Optional[int] = None) -> None:
        self.n_views = n_views

    def _validate_views(
        self,
        views: List,
        *,
        accept_sparse: bool = False,
        reset: bool = True,
    ) -> List[np.ndarray]:
        """Validate and coerce a list of views.

        Parameters
        ----------
        views : list of array-like
            Each element is one view, shape ``(n_samples, n_features_i)``.
        accept_sparse : bool, default=False
            Whether to accept scipy sparse matrices.
        reset : bool, default=True
            If True (during ``fit``), store ``n_views_`` and
            ``n_samples_``.  Set to False during ``transform`` /
            ``predict`` to check consistency instead.

        Returns
        -------
        list of ndarray
        """
        if hasattr(views, "_views"):
            views = views._views

        if not isinstance(views, (list, tuple)):
            raise TypeError(
                f"views must be a list of array-like objects, "
                f"got {type(views).__name__}. "
                "Wrap a single array in a list: [X]."
            )

        validated = []
        for i, v in enumerate(views):
            if accept_sparse:
                arr = self._check_array_sparse(v)
            else:
                arr = np.asarray(v, dtype=float)
            if arr.ndim != 2:
                raise ValueError(
                    f"View {i} must be 2-D (n_samples × n_features), got shape {arr.shape}."
                )
            validated.append(arr)

        if len(validated) == 0:
            raise ValueError("At least one view is required.")

        n = validated[0].shape[0]
        for i, arr in enumerate(validated[1:], start=1):
            if arr.shape[0] != n:
                raise ValueError(
                    f"All views must have the same number of samples. "
                    f"View 0 has {n} samples; view {i} has {arr.shape[0]}."
                )

        if self.n_views is not None and len(validated) != self.n_views:
            raise ValueError(
                f"This estimator expects {self.n_views} views, but received {len(validated)}."
            )

        if reset:
            self.n_views_in_ = len(validated)
            self.n_samples_ = n
            self.n_features_in_ = [v.shape[1] for v in validated]
        else:
            check_is_fitted(self, ["n_views_in_", "n_features_in_"])
            if len(validated) != self.n_views_in_:
                raise ValueError(
                    f"Fitted on {self.n_views_in_} views but received {len(validated)}."
                )
            for i, (arr, n_feat) in enumerate(zip(validated, self.n_features_in_)):
                if arr.shape[1] != n_feat:
                    raise ValueError(
                        f"View {i} has {arr.shape[1]} features but was fitted with {n_feat}."
                    )

        return validated

    @staticmethod
    def _check_array_sparse(v):
        """Accept scipy sparse or dense; always return 2-D array."""
        try:
            import scipy.sparse as sp

            if sp.issparse(v):
                return v  # leave sparse as-is; caller must handle
        except ImportError:
            pass
        return np.asarray(v, dtype=float)

    @abstractmethod
    def fit(self, views: List, y=None) -> "BaseMultiView":
        """Fit the model from a list of views.

        Parameters
        ----------
        views : list of array-like of shape (n_samples, n_features_i)
        y : ignored for unsupervised methods

        Returns
        -------
        self
        """

    def _more_tags(self) -> dict:
        """Extra sklearn tags (no_validation already handled by us)."""
        return {"no_validation": True, "multioutput": False}

    def _check_is_fitted(self) -> None:
        """Raise NotFittedError if the model has not been fitted yet."""
        check_is_fitted(self)


class MultiViewTransformerMixin:
    """Mixin for estimators that implement ``transform``.

    Provides a default ``fit_transform`` that calls ``fit`` then
    ``transform`` — subclasses only need to implement both methods.
    """

    @abstractmethod
    def transform(self, views: List) -> np.ndarray:
        """Apply the fitted transformation to views.

        Parameters
        ----------
        views : list of array-like

        Returns
        -------
        ndarray of shape (n_samples, n_components)
        """

    def fit_transform(self, views: List, y=None) -> np.ndarray:
        """Fit and immediately transform.

        Equivalent to ``self.fit(views, y).transform(views)`` but may be
        overridden for efficiency (e.g. CCA can reuse intermediate results).

        Parameters
        ----------
        views : list of array-like
        y : ignored

        Returns
        -------
        ndarray
        """
        return self.fit(views, y).transform(views)


class MultiViewClusterMixin:
    """Mixin for estimators that produce cluster labels.

    Provides ``fit_predict`` and enforces that ``fit`` stores
    ``labels_`` as a fitted attribute.
    """

    def fit_predict(self, views: List, y=None) -> np.ndarray:
        """Fit and return cluster labels.

        Parameters
        ----------
        views : list of array-like
        y : ignored

        Returns
        -------
        labels : ndarray of shape (n_samples,)
        """
        self.fit(views, y)
        check_is_fitted(self, "labels_")
        return self.labels_


class MultiViewEmbedderMixin(MultiViewTransformerMixin):
    """Mixin for estimators that produce a low-dimensional embedding.

    Identical contract to ``MultiViewTransformerMixin`` but signals
    that the output is an *embedding* (not a feature transformation),
    so callers can make different downstream choices (e.g. skip PCA
    post-processing).

    Subclasses should store the final embedding in ``embedding_``.
    """

    @property
    def embedding_(self) -> np.ndarray:
        check_is_fitted(self, "_embedding")
        return self._embedding

    @embedding_.setter
    def embedding_(self, value: np.ndarray) -> None:
        self._embedding = value


class BaseMultiViewTransformer(BaseMultiView, MultiViewTransformerMixin):
    """Ready-to-subclass base for multi-view transformers.

    Subclasses must implement :meth:`fit` and :meth:`transform`.

    Example
    -------
    >>> class MyConcatTransformer(BaseMultiViewTransformer):
    ...     def fit(self, views, y=None):
    ...         self._validate_views(views)
    ...         return self
    ...     def transform(self, views):
    ...         validated = self._validate_views(views, reset=False)
    ...         return np.concatenate(validated, axis=1)
    """


class BaseMultiViewClusterer(BaseMultiView, MultiViewClusterMixin):
    """Ready-to-subclass base for multi-view clustering algorithms.

    Subclasses must implement :meth:`fit`.  After fitting, ``labels_``
    must be set as an attribute.

    Example
    -------
    >>> class MyClusterer(BaseMultiViewClusterer):
    ...     def __init__(self, n_clusters=2):
    ...         super().__init__()
    ...         self.n_clusters = n_clusters
    ...     def fit(self, views, y=None):
    ...         views = self._validate_views(views)
    ...         # ... clustering logic ...
    ...         self.labels_ = np.zeros(self.n_samples_, dtype=int)
    ...         return self
    """


class BaseMultiViewEmbedder(BaseMultiView, MultiViewEmbedderMixin):
    """Ready-to-subclass base for multi-view embedding methods.

    Subclasses must implement :meth:`fit` and :meth:`transform`.
    Store the embedding in ``self.embedding_`` after fitting.
    """


class BaseLateFusion(BaseEstimator):
    """Base class for late-fusion estimators over per-view predictions.

    Late-fusion estimators consume one 1-D prediction vector per view and
    return one fused 1-D prediction vector.
    """

    def fit(self, preds_by_view: List[Iterable], y=None):
        raise NotImplementedError

    def predict(self, preds_by_view: List[Iterable]) -> np.ndarray:
        raise NotImplementedError

    def fit_predict(self, preds_by_view: List[Iterable], y=None) -> np.ndarray:
        return self.fit(preds_by_view, y).predict(preds_by_view)
