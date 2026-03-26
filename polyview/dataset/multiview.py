"""
Core data container for multi-view datasets.
"""

from __future__ import annotations

import numpy as np
from typing import Iterable, List, Optional, Union


class MultiViewDataset:
    """Container for multi-view datasets.

    Stores an ordered collection of views (2-D arrays sharing the same
    sample axis) together with optional sample labels and view names.

    Parameters
    ----------
    views : list of array-like of shape (n_samples, n_features_i)
        Each element is one view of the data.  All views must have the
        same number of rows (samples).
    labels : array-like of shape (n_samples,), optional
        Ground-truth or supervision labels, one per sample.
    view_names : list of str, optional
        Human-readable name for each view.  Defaults to
        ``["view_0", "view_1", ...]`` when not provided.

    Attributes
    ----------
    views : list of ndarray
        Validated, read-only copies of the input views.
    labels : ndarray or None
        Sample labels, cast to a 1-D ndarray.
    view_names : list of str
        Names assigned to each view.
    n_views : int
        Number of views.
    n_samples : int
        Number of samples (shared across views).

    Examples
    --------
    >>> import numpy as np
    >>> from polyview.datasets import MultiViewDataset
    >>> X1 = np.random.rand(100, 10)   # acoustic features
    >>> X2 = np.random.rand(100, 20)   # visual features
    >>> mvd = MultiViewDataset([X1, X2], view_names=["audio", "video"])
    >>> mvd.n_views
    2
    >>> mvd.n_samples
    100
    >>> mvd["audio"].shape
    (100, 10)
    >>> train, test = mvd.train_test_split(test_size=0.2, random_state=42)
    """

    def __init__(
        self,
        views: List,
        labels=None,
        view_names: Optional[List[str]] = None,
    ) -> None:
        self._views = self._validate_views(views)

        if labels is not None:
            labels = np.asarray(labels)
            if labels.shape[0] != self.n_samples:
                raise ValueError(
                    f"labels has {labels.shape[0]} entries but views have "
                    f"{self.n_samples} samples."
                )
        self._labels = labels

        if view_names is not None:
            if len(view_names) != self.n_views:
                raise ValueError(
                    f"view_names has {len(view_names)} entries but there are "
                    f"{self.n_views} views."
                )
            self._view_names = list(view_names)
        else:
            self._view_names = [f"view_{i}" for i in range(self.n_views)]

    @staticmethod
    def _validate_views(views) -> List[np.ndarray]:
        """Convert to numpy arrays and enforce shape consistency."""
        if not hasattr(views, "__iter__") or isinstance(views, np.ndarray):
            raise TypeError(
                "views must be a list (or iterable) of array-like objects, "
                f"got {type(views).__name__}."
            )
        validated = []
        for i, v in enumerate(views):
            arr = np.asarray(v, dtype=float)
            if arr.ndim != 2:
                raise ValueError(
                    f"View {i} must be 2-D (n_samples × n_features), "
                    f"got shape {arr.shape}."
                )
            validated.append(arr)

        if len(validated) == 0:
            raise ValueError("At least one view is required.")

        n = validated[0].shape[0]
        for i, arr in enumerate(validated[1:], start=1):
            if arr.shape[0] != n:
                raise ValueError(
                    f"All views must have the same number of samples. "
                    f"View 0 has {n} samples but view {i} has {arr.shape[0]}."
                )
        return validated

    @property
    def views(self) -> List[np.ndarray]:
        """List of view arrays (read-only copies)."""
        return [v.copy() for v in self._views]

    @property
    def labels(self) -> Optional[np.ndarray]:
        return self._labels

    @labels.setter
    def labels(self, value):
        if value is None:
            self._labels = None
            return
        arr = np.asarray(value)
        if arr.shape[0] != self.n_samples:
            raise ValueError(
                f"labels has {arr.shape[0]} entries but dataset has "
                f"{self.n_samples} samples."
            )
        self._labels = arr

    @property
    def view_names(self) -> List[str]:
        return list(self._view_names)

    @property
    def n_views(self) -> int:
        return len(self._views)

    @property
    def n_samples(self) -> int:
        return self._views[0].shape[0]

    @property
    def n_features(self) -> List[int]:
        """Number of features in each view."""
        return [v.shape[1] for v in self._views]

    def __getitem__(self, key: Union[int, str]) -> np.ndarray:
        """Return a single view by integer index or name.

        Parameters
        ----------
        key : int or str
            Integer index or view name.

        Returns
        -------
        ndarray of shape (n_samples, n_features_i)
        """
        if isinstance(key, str):
            if key not in self._view_names:
                raise KeyError(
                    f"View '{key}' not found. Available: {self._view_names}."
                )
            return self._views[self._view_names.index(key)].copy()
        if isinstance(key, int):
            return self._views[key].copy()
        raise TypeError(f"Key must be int or str, got {type(key).__name__}.")

    def __len__(self) -> int:
        return self.n_views

    def __iter__(self):
        """Iterate over view arrays."""
        return iter(self.views)

    def __repr__(self) -> str:
        parts = [f"n_views={self.n_views}", f"n_samples={self.n_samples}"]
        parts.append(f"n_features={self.n_features}")
        if self._labels is not None:
            unique = np.unique(self._labels)
            parts.append(f"n_classes={len(unique)}")
        return f"MultiViewDataset({', '.join(parts)})"

    def subset_samples(self, indices) -> "MultiViewDataset":
        """Return a new dataset with only the selected sample indices.

        Parameters
        ----------
        indices : array-like of int or bool mask
            Sample indices or boolean mask to select.

        Returns
        -------
        MultiViewDataset
        """
        idx = np.asarray(indices)
        new_views = [v[idx] for v in self._views]
        new_labels = self._labels[idx] if self._labels is not None else None
        return MultiViewDataset(
            new_views, labels=new_labels, view_names=self._view_names
        )

    def subset_views(self, keys) -> "MultiViewDataset":
        """Return a new dataset with only the selected views.

        Parameters
        ----------
        keys : list of int or str

        Returns
        -------
        MultiViewDataset
        """
        indices = []
        for k in keys:
            if isinstance(k, str):
                if k not in self._view_names:
                    raise KeyError(f"View '{k}' not found.")
                indices.append(self._view_names.index(k))
            else:
                indices.append(k)
        new_views = [self._views[i] for i in indices]
        new_names = [self._view_names[i] for i in indices]
        return MultiViewDataset(new_views, labels=self._labels, view_names=new_names)


    def train_test_split(
        self,
        test_size: float = 0.2,
        random_state: Optional[int] = None,
        stratify: bool = False,
    ) -> tuple["MultiViewDataset", "MultiViewDataset"]:
        """Split the dataset into train and test subsets.

        Parameters
        ----------
        test_size : float, default=0.2
            Fraction of samples to include in the test set.
        random_state : int, optional
            Seed for reproducibility.
        stratify : bool, default=False
            If True and labels are present, perform stratified split.

        Returns
        -------
        train : MultiViewDataset
        test : MultiViewDataset
        """
        rng = np.random.RandomState(random_state)
        n = self.n_samples
        n_test = max(1, int(np.floor(test_size * n)))

        if stratify and self._labels is not None:
            train_idx, test_idx = [], []
            for cls in np.unique(self._labels):
                cls_idx = np.where(self._labels == cls)[0]
                rng.shuffle(cls_idx)
                n_cls_test = max(1, int(np.floor(test_size * len(cls_idx))))
                test_idx.extend(cls_idx[:n_cls_test])
                train_idx.extend(cls_idx[n_cls_test:])
            train_idx = np.array(train_idx)
            test_idx = np.array(test_idx)
        else:
            perm = rng.permutation(n)
            test_idx = perm[:n_test]
            train_idx = perm[n_test:]

        return self.subset_samples(train_idx), self.subset_samples(test_idx)

    def to_numpy(self) -> np.ndarray:
        """Concatenate all views into a single array (early fusion).

        Returns
        -------
        ndarray of shape (n_samples, sum(n_features))
        """
        return np.concatenate(self._views, axis=1)

    def save(self, path: str) -> None:
        """Save the dataset to a compressed .npz file.

        Parameters
        ----------
        path : str
            File path (the .npz extension is added automatically if absent).
        """
        data = {f"view_{i}": v for i, v in enumerate(self._views)}
        data["view_names"] = np.array(self._view_names)
        if self._labels is not None:
            data["labels"] = self._labels
        np.savez_compressed(path, **data)

    @classmethod
    def load(cls, path: str) -> "MultiViewDataset":
        """Load a dataset from a .npz file saved with :meth:`save`.

        Parameters
        ----------
        path : str

        Returns
        -------
        MultiViewDataset
        """
        archive = np.load(path, allow_pickle=False)
        view_names = list(archive["view_names"])
        views = [archive[f"view_{i}"] for i in range(len(view_names))]
        labels = archive["labels"] if "labels" in archive else None
        return cls(views, labels=labels, view_names=view_names)
