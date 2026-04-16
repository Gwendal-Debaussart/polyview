"""Utilities for generating multiple random-projection views from a single matrix."""

from typing import Literal, Optional, Sequence, Union, cast

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection

from polyview.dataset.multiviewdataset import MultiViewDataset


def _resolve_components(n_features: int, n_components: Optional[int]) -> int:
    if n_components is not None:
        if n_components <= 0:
            raise ValueError("n_components must be a positive integer.")
        return n_components
    return max(1, min(n_features, max(2, n_features // 2)))


def _build_projector(
    method: Literal["gaussian", "sparse"],
    n_components: int,
    random_state: Optional[int],
):
    if method == "gaussian":
        return GaussianRandomProjection(
            n_components=n_components, random_state=random_state
        )
    if method == "sparse":
        return SparseRandomProjection(
            n_components=n_components, random_state=random_state
        )
    raise ValueError(f"Unknown projection method: {method!r}.")


class RandomProjectionViews(BaseEstimator):
    """Generate multiple projected views from a single matrix.

    This transformer is pipeline-friendly: it accepts a single 2-D matrix
    and outputs a :class:`~polyview.dataset.multiviewdataset.MultiViewDataset`
    containing ``n_views`` independent random projections of the input.
    """

    def __init__(
        self,
        n_views: int = 2,
        n_components: Optional[Union[int, Sequence[Optional[int]]]] = None,
        method: Literal["gaussian", "sparse"] = "gaussian",
        random_state: Optional[int] = None,
        view_names: Optional[Sequence[str]] = None,
    ) -> None:
        self.n_views = n_views
        self.n_components = n_components
        self.method = method
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

    def _resolve_component_list(self, n_features: int) -> list[Optional[int]]:
        if isinstance(self.n_components, Sequence) and not isinstance(
            self.n_components, (str, bytes)
        ):
            component_list = [cast(Optional[int], item) for item in self.n_components]
            if len(component_list) != self.n_views:
                raise ValueError(
                    f"n_components has {len(component_list)} entries but n_views={self.n_views}."
                )
            return component_list
        return [cast(Optional[int], self.n_components)] * self.n_views

    def fit(self, X, y=None) -> "RandomProjectionViews":
        X = self._validate_input(X)
        _, n_features = X.shape

        component_list = self._resolve_component_list(n_features)

        self.projectors_ = []
        self.n_features_in_ = n_features
        self.n_views_in_ = self.n_views

        for view_idx in range(self.n_views):
            raw_components = component_list[view_idx]
            if raw_components is not None and not isinstance(raw_components, int):
                raise TypeError("Each n_components entry must be an int or None.")
            target_components = _resolve_components(n_features, raw_components)
            method = cast(Literal["gaussian", "sparse"], self.method)
            projector = _build_projector(
                method=method,
                n_components=target_components,
                random_state=None
                if self.random_state is None
                else self.random_state + view_idx,
            )
            projector.fit(X)
            self.projectors_.append(projector)

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
            np.asarray(projector.transform(X), dtype=float)
            for projector in self.projectors_
        ]
        return MultiViewDataset(views=views, view_names=self.view_names_)

    def fit_transform(self, X, y=None) -> MultiViewDataset:
        return self.fit(X, y).transform(X)


def random_projection(
    X: Union[np.ndarray, Sequence[Sequence[float]]],
    n_views: int = 2,
    n_components: Optional[Union[int, Sequence[Optional[int]]]] = None,
    method: Literal["gaussian", "sparse"] = "gaussian",
    random_state: Optional[int] = None,
    labels=None,
    view_names: Optional[Sequence[str]] = None,
) -> MultiViewDataset:
    """Create multiple projected views from a single 2-D matrix.

    This helper is equivalent to instantiating
    :class:`RandomProjectionViews` and calling ``fit_transform``.
    """
    dataset = RandomProjectionViews(
        n_views=n_views,
        n_components=n_components,
        method=method,
        random_state=random_state,
        view_names=view_names,
    ).fit_transform(X)
    if labels is not None:
        dataset.labels = labels
    return dataset
