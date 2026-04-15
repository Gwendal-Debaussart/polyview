from __future__ import annotations

from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.utils.validation import check_is_fitted

from polyview.base import BaseMultiView


DataMode = Literal["mv", "sv"]
StepSpec = Tuple[str, Any]


class _PerViewTransformer(BaseEstimator):
    """Apply a single-view transformer independently to each view."""

    def __init__(
        self,
        estimator: Any,
        shared_params: Optional[Dict[str, Any]] = None,
        per_view_params: Optional[Sequence[Optional[Dict[str, Any]]]] = None,
    ) -> None:
        self.estimator = estimator
        self.shared_params = shared_params
        self.per_view_params = per_view_params

    @staticmethod
    def _validate_views(views: Any) -> List[np.ndarray]:
        if not isinstance(views, (list, tuple)) or len(views) == 0:
            raise TypeError(
                "Per-view transformation requires a non-empty list/tuple of views."
            )
        arrs = [np.asarray(v, dtype=float) for v in views]
        if any(a.ndim != 2 for a in arrs):
            raise ValueError("Every view must be a 2-D array.")
        return arrs

    def _build_estimator_for_view(self, view_idx: int) -> Any:
        est = clone(self.estimator)
        if self.shared_params:
            est.set_params(**self.shared_params)
        if self.per_view_params and view_idx < len(self.per_view_params):
            params = self.per_view_params[view_idx]
            if params:
                est.set_params(**params)
        return est

    def fit(self, views: Any, y: Any = None) -> "_PerViewTransformer":
        arrs = self._validate_views(views)
        self.estimators_ = []
        for i, X in enumerate(arrs):
            est = self._build_estimator_for_view(i)
            est.fit(X, y)
            self.estimators_.append(est)
        self.n_views_in_ = len(arrs)
        return self

    def transform(self, views: Any) -> List[np.ndarray]:
        check_is_fitted(self, "estimators_")
        arrs = self._validate_views(views)
        if len(arrs) != self.n_views_in_:
            raise ValueError(
                f"Per-view transformer was fitted on {self.n_views_in_} views but got {len(arrs)}."
            )
        return [est.transform(X) for est, X in zip(self.estimators_, arrs)]

    def fit_transform(self, views: Any, y: Any = None) -> List[np.ndarray]:
        return self.fit(views, y).transform(views)


class PolyPipeline(BaseEstimator):
    """
    Pipeline that supports both multiview and single-view flows.

    The pipeline infers data mode at runtime from step outputs:
    - multiview ("mv"): list/tuple of 2-D arrays
    - single-view ("sv"): one 2-D array

    Allowed transitions are:
    - mv -> mv
    - mv -> sv
    - sv -> sv

    A reverse transition (sv -> mv) is rejected.
    """

    def __init__(
        self,
        steps: Sequence[StepSpec],
        per_view_step_params: Optional[
            Dict[str, Union[Dict[str, Any], Sequence[Optional[Dict[str, Any]]]]]
        ] = None,
    ) -> None:
        self.steps = steps
        self.per_view_step_params = per_view_step_params

    @staticmethod
    def _is_mv_data(X: Any) -> bool:
        if not isinstance(X, (list, tuple)) or len(X) == 0:
            return False
        return all(np.asarray(v).ndim == 2 for v in X)

    @staticmethod
    def _is_sv_data(X: Any) -> bool:
        return isinstance(X, np.ndarray) and X.ndim == 2

    def _infer_mode(self, X: Any) -> DataMode:
        if self._is_mv_data(X):
            return "mv"
        if self._is_sv_data(X):
            return "sv"
        raise TypeError(
            "Input must be a list/tuple of 2-D arrays (multiview) or a single 2-D numpy array (single-view)."
        )

    @staticmethod
    def _as_mv(X: Union[List[Any], Tuple[Any, ...]]) -> List[np.ndarray]:
        return [np.asarray(v, dtype=float) for v in X]

    @staticmethod
    def _as_sv(X: Any) -> np.ndarray:
        arr = np.asarray(X, dtype=float)
        if arr.ndim != 2:
            raise ValueError(f"Single-view inputs must be 2-D. Got shape {arr.shape}.")
        return arr

    def _prepare_input(
        self, X: Any, mode: DataMode
    ) -> Union[List[np.ndarray], np.ndarray]:
        if mode == "mv":
            return self._as_mv(X)
        return self._as_sv(X)

    def _validate_steps(self) -> List[StepSpec]:
        if not isinstance(self.steps, Iterable):
            raise TypeError("steps must be an iterable of (name, estimator) pairs.")

        parsed = list(self.steps)
        if len(parsed) == 0:
            raise ValueError("steps must not be empty.")

        names = [name for name, _ in parsed]
        if len(names) != len(set(names)):
            raise ValueError("Step names must be unique.")

        for i, step in enumerate(parsed):
            if not isinstance(step, tuple) or len(step) != 2:
                raise TypeError(
                    f"Step {i} must be a (name, estimator) tuple, got {type(step).__name__}."
                )
            name, estimator = step
            if not isinstance(name, str):
                raise TypeError(f"Step {i} has a non-string name: {name!r}.")
            if estimator != "passthrough" and not hasattr(estimator, "fit"):
                raise TypeError(
                    f"Step '{name}' must be an estimator with fit() or 'passthrough'."
                )

        return parsed

    def _fit_transform_step(self, estimator: Any, X: Any, y: Any) -> Any:
        if hasattr(estimator, "fit_transform"):
            return estimator.fit_transform(X, y)
        estimator.fit(X, y)
        if not hasattr(estimator, "transform"):
            raise TypeError(
                f"Non-final step {type(estimator).__name__} must define transform()."
            )
        return estimator.transform(X)

    @staticmethod
    def _require_method(step_name: str, estimator: Any, method: str) -> None:
        if not hasattr(estimator, method):
            raise AttributeError(
                f"Final step '{step_name}' ({type(estimator).__name__}) does not implement {method}()."
            )

    def _apply_transforms(self, X: Any, include_final: bool = False) -> Any:
        check_is_fitted(self, "steps_")
        Xt = self._prepare_input(X, self.input_mode_)
        end = len(self.steps_) if include_final else len(self.steps_) - 1
        for name, step in self.steps_[:end]:
            if step == "passthrough":
                continue
            self._require_compatible_step_input(name, step, Xt)
            Xt = step.transform(Xt)
        return Xt

    @staticmethod
    def _mode_transition_ok(previous: DataMode, current: DataMode) -> bool:
        return previous == current or (previous == "mv" and current == "sv")

    def _require_compatible_step_input(self, step_name: str, step: Any, X: Any) -> None:
        mode = self._infer_mode(X)
        if mode == "sv" and isinstance(step, BaseMultiView):
            raise ValueError(
                f"Step '{step_name}' ({type(step).__name__}) requires multi-view input, but the pipeline is currently in single-view mode."
            )

    def _get_per_view_cfg(
        self, step_name: str
    ) -> Tuple[Optional[Dict[str, Any]], Optional[Sequence[Optional[Dict[str, Any]]]]]:
        cfg = (self.per_view_step_params or {}).get(step_name)
        if cfg is None:
            return None, None
        if isinstance(cfg, dict):
            return cfg, None
        if isinstance(cfg, Sequence) and not isinstance(cfg, (str, bytes)):
            return None, cfg
        raise TypeError(
            f"per_view_step_params entries must be dict or sequence of dict/None, got {type(cfg).__name__} for step '{step_name}'."
        )

    def _adapt_step_for_mode(
        self, step_name: str, estimator: Any, mode: DataMode
    ) -> Any:
        if mode != "mv" or isinstance(estimator, BaseMultiView):
            return estimator

        if hasattr(estimator, "transform"):
            shared_params, per_view_params = self._get_per_view_cfg(step_name)
            return _PerViewTransformer(
                estimator=estimator,
                shared_params=shared_params,
                per_view_params=per_view_params,
            )

        raise ValueError(
            f"Step '{step_name}' ({type(estimator).__name__}) expects single-view input while "
            "the pipeline is in multi-view mode. Add a fusion step first, or use a transformer that can run per-view."
        )

    def fit(self, X: Any, y: Any = None) -> "PolyPipeline":
        steps = self._validate_steps()
        current_mode = self._infer_mode(X)
        Xt = self._prepare_input(X, current_mode)

        fitted_steps: List[StepSpec] = []
        for i, (name, step) in enumerate(steps):
            is_last = i == len(steps) - 1
            if step == "passthrough":
                fitted_steps.append((name, step))
                continue

            est = clone(step)
            est = self._adapt_step_for_mode(name, est, current_mode)
            self._require_compatible_step_input(name, est, Xt)

            if is_last:
                est.fit(Xt, y)
                fitted_steps.append((name, est))
                break

            Xt = self._fit_transform_step(est, Xt, y)
            new_mode = self._infer_mode(Xt)
            if not self._mode_transition_ok(current_mode, new_mode):
                raise ValueError(
                    f"Unsupported mode transition after step '{name}': {current_mode} -> {new_mode}. Only mv->sv is allowed."
                )
            current_mode = new_mode
            Xt = self._prepare_input(Xt, current_mode)
            fitted_steps.append((name, est))

        self.steps_ = fitted_steps
        self.input_mode_ = self._infer_mode(X)
        self.mode_after_transforms_ = current_mode
        return self

    def transform(self, X: Any) -> Any:
        Xt = self._apply_transforms(X, include_final=False)
        name, final_step = self.steps_[-1]
        if final_step == "passthrough":
            return Xt
        self._require_compatible_step_input(name, final_step, Xt)
        self._require_method(name, final_step, "transform")
        return final_step.transform(Xt)

    def fit_transform(self, X: Any, y: Any = None) -> Any:
        return self.fit(X, y).transform(X)

    def predict(self, X: Any) -> Any:
        Xt = self._apply_transforms(X, include_final=False)
        name, final_step = self.steps_[-1]
        if final_step == "passthrough":
            raise AttributeError("Final step is passthrough and cannot predict().")
        self._require_compatible_step_input(name, final_step, Xt)
        self._require_method(name, final_step, "predict")
        return final_step.predict(Xt)

    def fit_predict(self, X: Any, y: Any = None) -> Any:
        self.fit(X, y)
        name, final_step = self.steps_[-1]
        if final_step == "passthrough":
            raise AttributeError("Final step is passthrough and cannot fit_predict().")
        if hasattr(final_step, "labels_"):
            return final_step.labels_
        if hasattr(final_step, "predict"):
            return self.predict(X)
        raise AttributeError(
            f"Final step '{name}' ({type(final_step).__name__}) does not support fit_predict()."
        )

    def score(self, X: Any, y: Any = None) -> Any:
        Xt = self._apply_transforms(X, include_final=False)
        name, final_step = self.steps_[-1]
        if final_step == "passthrough":
            raise AttributeError("Final step is passthrough and cannot score().")
        self._require_compatible_step_input(name, final_step, Xt)
        self._require_method(name, final_step, "score")
        return final_step.score(Xt, y)
