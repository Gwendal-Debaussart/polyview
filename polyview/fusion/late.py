"""Late fusion strategies for combining per-view predictions."""

from __future__ import annotations

from typing import Iterable, List, Literal, Optional, Sequence

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted


class BaseLateFusion(BaseEstimator):
    """Base class for late-fusion estimators over per-view predictions."""

    def fit(self, preds_by_view: List[Iterable], y=None):
        raise NotImplementedError

    def predict(self, preds_by_view: List[Iterable]) -> np.ndarray:
        raise NotImplementedError

    def fit_predict(self, preds_by_view: List[Iterable], y=None) -> np.ndarray:
        return self.fit(preds_by_view, y).predict(preds_by_view)


class MajorityVote(BaseLateFusion):
    """Fuse per-view discrete predictions with sample-wise majority vote.

    Parameters
    ----------
    weights : sequence of float or None, default=None
        Optional non-negative per-view weights. If ``None``, each view gets
        weight 1.0. Must have one value per view.
    tie_break : {"first", "random"}, default="first"
        Strategy used when multiple classes receive the same max vote.
        - "first": choose the smallest class label among tied classes.
        - "random": choose a random tied class (reproducible via random_state).
    random_state : int or None, default=None
        Seed used only when ``tie_break='random'``.
    """

    def __init__(
        self,
        weights: Optional[Sequence[float]] = None,
        tie_break: Literal["first", "random"] = "first",
        random_state: Optional[int] = None,
    ) -> None:
        self.weights = weights
        self.tie_break = tie_break
        self.random_state = random_state

    @staticmethod
    def _to_1d_int_array(x: Iterable, idx: int) -> np.ndarray:
        arr = np.asarray(x)
        if arr.ndim != 1:
            raise ValueError(
                f"Prediction vector for view {idx} must be 1-D, got shape {arr.shape}."
            )
        # Majority voting is over class ids, so force integer labels.
        try:
            arr_int = arr.astype(int)
        except ValueError as exc:
            raise TypeError(
                f"Prediction vector for view {idx} must contain integer-like labels."
            ) from exc
        return arr_int

    def _validate_predictions(self, preds_by_view: List[Iterable], *, reset: bool) -> np.ndarray:
        if not isinstance(preds_by_view, (list, tuple)) or len(preds_by_view) == 0:
            raise TypeError(
                "preds_by_view must be a non-empty list/tuple of 1-D prediction arrays."
            )

        arrays = [self._to_1d_int_array(p, i) for i, p in enumerate(preds_by_view)]
        n_samples = arrays[0].shape[0]
        for i, arr in enumerate(arrays[1:], start=1):
            if arr.shape[0] != n_samples:
                raise ValueError(
                    f"All prediction vectors must have the same length. "
                    f"View 0 has {n_samples} samples but view {i} has {arr.shape[0]}."
                )

        stacked = np.vstack(arrays)

        if reset:
            self.n_views_in_ = stacked.shape[0]
            self.n_samples_in_ = stacked.shape[1]
        else:
            check_is_fitted(self, ["n_views_in_", "n_samples_in_"])
            if stacked.shape[0] != self.n_views_in_:
                raise ValueError(
                    f"Fitted on {self.n_views_in_} views but got {stacked.shape[0]}."
                )
            if stacked.shape[1] != self.n_samples_in_:
                raise ValueError(
                    f"Fitted on {self.n_samples_in_} samples but got {stacked.shape[1]}."
                )

        return stacked

    def _resolve_weights(self, n_views: int) -> np.ndarray:
        if self.weights is None:
            return np.ones(n_views, dtype=float)
        weights = np.asarray(self.weights, dtype=float)
        if weights.ndim != 1:
            raise ValueError("weights must be a 1-D sequence.")
        if weights.shape[0] != n_views:
            raise ValueError(
                f"weights has {weights.shape[0]} values but got {n_views} views."
            )
        if np.any(weights < 0):
            raise ValueError("weights must be non-negative.")
        if not np.any(weights > 0):
            raise ValueError("At least one weight must be > 0.")
        return weights

    def _vote_single_sample(
        self,
        sample_votes: np.ndarray,
        sample_weights: np.ndarray,
        rng: np.random.RandomState,
    ) -> int:
        classes = np.unique(sample_votes)
        class_scores = np.array(
            [sample_weights[sample_votes == c].sum() for c in classes],
            dtype=float,
        )
        max_score = class_scores.max()
        tied = classes[np.isclose(class_scores, max_score)]

        if tied.shape[0] == 1 or self.tie_break == "first":
            return int(tied.min())
        if self.tie_break == "random":
            return int(rng.choice(tied))
        raise ValueError(
            f"tie_break must be 'first' or 'random', got {self.tie_break!r}."
        )

    def fit(self, preds_by_view: List[Iterable], y=None) -> "MajorityVote":
        stacked = self._validate_predictions(preds_by_view, reset=True)
        self.weights_ = self._resolve_weights(stacked.shape[0])
        return self

    def predict(self, preds_by_view: List[Iterable]) -> np.ndarray:
        stacked = self._validate_predictions(preds_by_view, reset=False)
        check_is_fitted(self, "weights_")
        if stacked.shape[0] != self.weights_.shape[0]:
            raise ValueError(
                f"weights were fitted for {self.weights_.shape[0]} views but got {stacked.shape[0]}."
            )
        rng = np.random.RandomState(self.random_state)
        fused = np.array(
            [
                self._vote_single_sample(stacked[:, j], self.weights_, rng)
                for j in range(stacked.shape[1])
            ],
            dtype=int,
        )
        return fused

    def fit_predict(self, preds_by_view: List[Iterable], y=None) -> np.ndarray:
        return super().fit_predict(preds_by_view, y)
