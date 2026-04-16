from typing import Optional, Sequence, Union, cast
import numpy as np
from sklearn.base import BaseEstimator
from polyview.dataset.multiviewdataset import MultiViewDataset


class RandomSubspaceViews(BaseEstimator):
    """Generate multiple random-subspace views from a single matrix.

    Random subspace method creates new views by randomly selecting subsets of features (without replacement) from the original feature space. Unlike random projections which create weighted combinations, random subspaces preserve the original features but use only a subset.

    Parameters
    ----------
    n_views : int, default=2
        Number of random subspace views to generate.
    n_features_per_view : int, optional
        Number of features to select for each view. If None, defaults to
        approximately sqrt(n_features) following Ho (1998).
    random_state : int or None, default=None
        Random seed for reproducibility.
    view_names : sequence of str, optional
        Names for each view. If None, defaults to ["view_0", "view_1", ...].

    Attributes
    ----------
    feature_indices_ : list of ndarray
        Indices of selected features for each view.
    n_features_in_ : int
        Number of features in the input data.
    n_views_in_ : int
        Number of views generated (equals n_views).
    view_names_ : list of str
        Names of each view.

    References
    ----------
    - Ho, T. K. (1998). The random subspace method for constructing decision forests.
      IEEE Transactions on Pattern Analysis and Machine Intelligence, 20(8), 832-844.

    Examples
    --------
    >>> import numpy as np
    >>> from polyview.augmentation.random_subspace import RandomSubspaceViews
    >>> X = np.random.rand(100, 20)
    >>> rsv = RandomSubspaceViews(n_views=3, n_features_per_view=10, random_state=0)
    >>> mvd = rsv.fit_transform(X)
    >>> len(mvd.views)
    3
    >>> mvd.views[0].shape
    (100, 10)

    Notes
    -----
    This transformer is pipeline-friendly: it accepts a single 2-D matrix
    and outputs a :class:`~polyview.dataset.multiviewdataset.MultiViewDataset`
    containing ``n_views`` independent random subspace projections of the input.
    """

    def __init__(
        self,
        n_views: int = 2,
        n_features_per_view: Optional[int] = None,
        random_state: Optional[int] = None,
        view_names: Optional[Sequence[str]] = None,
    ) -> None:
        self.n_views = n_views
        self.n_features_per_view = n_features_per_view
        self.random_state = random_state
        self.view_names = view_names

    @staticmethod
    def _validate_input(X: Union[np.ndarray, Sequence[Sequence[float]]]) -> np.ndarray:
        arr = np.asarray(X, dtype=float)
        if arr.ndim != 2:
            raise ValueError(
                f"X must be 2-D (n_samples, n_features), got shape {arr.shape}."
            )
        return arr

    def _resolve_n_features_per_view(self, n_features: int) -> int:
        if self.n_features_per_view is not None:
            if self.n_features_per_view <= 0 or self.n_features_per_view > n_features:
                raise ValueError(
                    f"n_features_per_view must be in (0, {n_features}], "
                    f"got {self.n_features_per_view}."
                )
            return self.n_features_per_view
        # Default to sqrt(n_features) as in Ho (1998)
        return max(1, int(np.sqrt(n_features)))

    def fit(self, X, y=None) -> "RandomSubspaceViews":
        X = self._validate_input(X)
        n_samples, n_features = X.shape

        n_feat_per_view = self._resolve_n_features_per_view(n_features)

        rng = np.random.RandomState(self.random_state)
        self.feature_indices_ = []

        for view_idx in range(self.n_views):
            indices = rng.choice(n_features, size=n_feat_per_view, replace=False)
            self.feature_indices_.append(np.sort(indices))

        self.n_features_in_ = n_features
        self.n_views_in_ = self.n_views

        self.view_names_ = (
            list(self.view_names)
            if self.view_names is not None
            else [f"view_{i}" for i in range(self.n_views)]
        )
        return self

    def transform(self, X) -> MultiViewDataset:
        X = self._validate_input(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features but the transformer was fitted with {self.n_features_in_}."
            )

        views = [
            np.asarray(X[:, indices], dtype=float) for indices in self.feature_indices_
        ]
        return MultiViewDataset(views=views, view_names=self.view_names_)

    def fit_transform(self, X, y=None) -> MultiViewDataset:
        return self.fit(X, y).transform(X)


def random_subspace(
    X: Union[np.ndarray, Sequence[Sequence[float]]],
    n_views: int = 2,
    n_features_per_view: Optional[int] = None,
    random_state: Optional[int] = None,
    labels=None,
    view_names: Optional[Sequence[str]] = None,
) -> MultiViewDataset:
    """Create multiple random-subspace views from a single 2-D matrix.

    This helper is equivalent to instantiating
    :class:`RandomSubspaceViews` and calling ``fit_transform``.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input data matrix.
    n_views : int, default=2
        Number of random subspace views.
    n_features_per_view : int, optional
        Number of features per view. If None, defaults to sqrt(n_features).
    random_state : int or None, default=None
        Random seed.
    labels : array-like of shape (n_samples,), optional
        Optional ground-truth labels (attached to result).
    view_names : sequence of str, optional
        Names for each view.

    Returns
    -------
    MultiViewDataset
        Dataset containing ``n_views`` random-subspace views.
    """
    dataset = RandomSubspaceViews(
        n_views=n_views,
        n_features_per_view=n_features_per_view,
        random_state=random_state,
        view_names=view_names,
    ).fit_transform(X)
    if labels is not None:
        dataset.labels = np.asarray(labels)
    return dataset
