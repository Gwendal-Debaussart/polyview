from typing import Optional, Sequence

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

from polyview.dataset.multiviewdataset import MultiViewDataset


class ViewSplitter(BaseEstimator):
    """Split a single matrix into consecutive column blocks, one per view.

    This is the inverse of :class:`~polyview.fusion.early.ConcatFusion` and of
    :meth:`MultiViewDataset.to_numpy`. Placed first in a
    :class:`~polyview.pipeline.polypipeline.PolyPipeline`, it lets a
    multi-view workflow receive its views as one 2-D array, which is what
    scikit-learn's model-selection tools (e.g.
    :class:`~sklearn.model_selection.GridSearchCV` or
    :func:`~sklearn.model_selection.cross_val_score`) expect, since they split
    samples along the first axis.

    Parameters
    ----------
    n_features : sequence of int or None, default=None
        Number of columns of each view, in order. Must sum to the number of
        columns of the input.
    view_names : sequence of str, optional
        Names for each view. If None, defaults to ["view_0", "view_1", ...].

    Attributes
    ----------
    split_indices_ : ndarray
        Column indices at which the input is split.
    n_features_in_ : int
        Number of columns of the input.
    n_views_in_ : int
        Number of views produced.
    view_names_ : list of str
        Names of each view.

    Examples
    --------
    >>> import numpy as np
    >>> from polyview.augmentation.view_splitter import ViewSplitter
    >>> X = np.random.rand(100, 9)
    >>> mvd = ViewSplitter(n_features=[5, 4]).fit_transform(X)
    >>> [v.shape for v in mvd.views]
    [(100, 5), (100, 4)]
    """

    def __init__(
        self,
        n_features: Optional[Sequence[int]] = None,
        view_names: Optional[Sequence[str]] = None,
    ) -> None:
        self.n_features = n_features
        self.view_names = view_names

    @property
    def n_views(self) -> int:
        """Number of views described by ``n_features`` (0 if unset)."""
        return 0 if self.n_features is None else len(self.n_features)

    @staticmethod
    def _validate_input(X) -> np.ndarray:
        arr = np.asarray(X, dtype=float)
        if arr.ndim != 2:
            raise ValueError(
                f"X must be 2-D (n_samples, n_features), got shape {arr.shape}."
            )
        return arr

    def fit(self, X, y=None) -> "ViewSplitter":
        X = self._validate_input(X)
        if self.n_features is None:
            raise ValueError("n_features must be given: one column count per view.")
        widths = [int(w) for w in self.n_features]
        if len(widths) == 0 or any(w <= 0 for w in widths):
            raise ValueError(
                f"n_features must contain positive integers, got {list(self.n_features)}."
            )
        if sum(widths) != X.shape[1]:
            raise ValueError(
                f"n_features sums to {sum(widths)} but X has {X.shape[1]} columns."
            )
        if self.view_names is not None and len(self.view_names) != len(widths):
            raise ValueError(
                f"view_names has {len(self.view_names)} entries but there are "
                f"{len(widths)} views."
            )

        self.split_indices_ = np.cumsum(widths)[:-1]
        self.n_features_in_ = X.shape[1]
        self.n_views_in_ = len(widths)
        self.view_names_ = (
            list(self.view_names)
            if self.view_names is not None
            else [f"view_{i}" for i in range(self.n_views_in_)]
        )
        return self

    def transform(self, X) -> MultiViewDataset:
        check_is_fitted(self, "split_indices_")
        X = self._validate_input(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} columns but the splitter was fitted with "
                f"{self.n_features_in_}."
            )
        views = np.split(X, self.split_indices_, axis=1)
        return MultiViewDataset(views=views, view_names=self.view_names_)

    def fit_transform(self, X, y=None) -> MultiViewDataset:
        return self.fit(X, y).transform(X)
