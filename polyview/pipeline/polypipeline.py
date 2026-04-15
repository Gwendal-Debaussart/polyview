from __future__ import annotations

from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple, Union, cast

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.utils.validation import check_is_fitted

from polyview.base import BaseMultiView

try:
    from polyview.fusion.late import BaseLateFusion
except Exception:  # pragma: no cover - optional import safety
    BaseLateFusion = None  # type: ignore[assignment]


DataMode = Literal["mv", "sv", "lf"]
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


class _PerViewEstimator(BaseEstimator):
    """Apply a single-view estimator independently to each view.

    This wrapper is intended for final pipeline steps that do not expose a
    ``transform`` method (e.g., sklearn clusterers/classifiers).
    """

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
                "Per-view estimation requires a non-empty list/tuple of views."
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

    @staticmethod
    def _fit_single(estimator: Any, X: np.ndarray, y: Any = None) -> Any:
        if y is None:
            return estimator.fit(X)
        try:
            return estimator.fit(X, y)
        except TypeError:
            # Some unsupervised sklearn estimators only accept fit(X).
            return estimator.fit(X)

    def fit(self, views: Any, y: Any = None) -> "_PerViewEstimator":
        arrs = self._validate_views(views)
        self.estimators_ = []
        for i, X in enumerate(arrs):
            est = self._build_estimator_for_view(i)
            self._fit_single(est, X, y)
            self.estimators_.append(est)
        self.n_views_in_ = len(arrs)
        self.labels_ = [getattr(est, "labels_", None) for est in self.estimators_]
        return self

    def _checked_views(self, views: Any) -> List[np.ndarray]:
        check_is_fitted(self, "estimators_")
        arrs = self._validate_views(views)
        if len(arrs) != self.n_views_in_:
            raise ValueError(
                f"Per-view estimator was fitted on {self.n_views_in_} views but got {len(arrs)}."
            )
        return arrs

    def predict(self, views: Any) -> List[np.ndarray]:
        arrs = self._checked_views(views)
        preds = []
        for est, X in zip(self.estimators_, arrs):
            if not hasattr(est, "predict"):
                raise AttributeError(
                    f"Underlying estimator {type(est).__name__} does not implement predict()."
                )
            preds.append(est.predict(X))
        return preds

    def fit_predict(self, views: Any, y: Any = None) -> List[np.ndarray]:
        self.fit(views, y)
        return self.predict(views)

    def score(self, views: Any, y: Any = None) -> np.ndarray:
        arrs = self._checked_views(views)
        scores = []
        for est, X in zip(self.estimators_, arrs):
            if not hasattr(est, "score"):
                raise AttributeError(
                    f"Underlying estimator {type(est).__name__} does not implement score()."
                )
            if y is None:
                scores.append(est.score(X))
            else:
                try:
                    scores.append(est.score(X, y))
                except TypeError:
                    scores.append(est.score(X))
        return np.asarray(scores)


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

    @staticmethod
    def _is_lf_data(X: Any) -> bool:
        if not isinstance(X, (list, tuple)) or len(X) == 0:
            return False
        arrs = [np.asarray(v) for v in X]
        return all(a.ndim == 1 for a in arrs)

    def _infer_mode(self, X: Any) -> DataMode:
        if self._is_mv_data(X):
            return "mv"
        if self._is_sv_data(X):
            return "sv"
        if self._is_lf_data(X):
            return "lf"
        raise TypeError(
            "Input must be a list/tuple of 2-D arrays (multiview), a single 2-D numpy array (single-view), or a list/tuple of 1-D prediction arrays (late-fusion)."
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

    @staticmethod
    def _as_lf(X: Union[List[Any], Tuple[Any, ...]]) -> List[np.ndarray]:
        arrs = [np.asarray(v) for v in X]
        if any(a.ndim != 1 for a in arrs):
            bad = next(a.shape for a in arrs if a.ndim != 1)
            raise ValueError(
                f"Late-fusion inputs must be 1-D prediction arrays. Got shape {bad}."
            )
        return arrs

    def _prepare_input(
        self, X: Any, mode: DataMode
    ) -> Union[List[np.ndarray], np.ndarray]:
        if mode == "mv":
            return self._as_mv(X)
        if mode == "lf":
            return self._as_lf(X)
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
        if hasattr(estimator, "transform"):
            return estimator.transform(X)
        if hasattr(estimator, "predict"):
            # Allows estimator -> late-fusion chaining (e.g. KMeans -> MajorityVote).
            return estimator.predict(X)
        raise TypeError(
            f"Non-final step {type(estimator).__name__} must define transform() or predict()."
        )

    @staticmethod
    def _require_method(step_name: str, estimator: Any, method: str) -> None:
        if not hasattr(estimator, method):
            raise AttributeError(
                f"Final step '{step_name}' ({type(estimator).__name__}) does not implement {method}()."
            )

    def _apply_transforms(self, X: Any, include_final: bool = False) -> Any:
        check_is_fitted(self, "steps_")
        Xt = self._prepare_input(X, cast(DataMode, self.input_mode_))
        end = len(self.steps_) if include_final else len(self.steps_) - 1
        for name, step in self.steps_[:end]:
            if step == "passthrough":
                continue
            self._require_compatible_step_input(name, step, Xt)
            if hasattr(step, "transform"):
                Xt = step.transform(Xt)
            elif hasattr(step, "predict"):
                Xt = step.predict(Xt)
            else:
                raise TypeError(
                    f"Step '{name}' ({type(step).__name__}) must define transform() or predict()."
                )
        return Xt

    @staticmethod
    def _mode_transition_ok(previous: DataMode, current: DataMode) -> bool:
        if previous == current:
            return True
        if previous == "mv" and current in ("sv", "lf"):
            return True
        if previous == "lf" and current == "sv":
            return True
        return False

    def _require_compatible_step_input(self, step_name: str, step: Any, X: Any) -> None:
        mode = self._infer_mode(X)
        if mode == "sv" and isinstance(step, BaseMultiView):
            raise ValueError(
                f"Step '{step_name}' ({type(step).__name__}) requires multi-view input, but the pipeline is currently in single-view mode."
            )
        if mode == "lf" and isinstance(step, BaseMultiView):
            raise ValueError(
                f"Step '{step_name}' ({type(step).__name__}) requires multi-view input, but the pipeline is currently in late-fusion mode."
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
        self,
        step_name: str,
        estimator: Any,
        mode: DataMode,
        is_last: bool,
        force_predict_path: bool = False,
    ) -> Any:
        if mode != "mv" or isinstance(estimator, BaseMultiView):
            return estimator

        shared_params, per_view_params = self._get_per_view_cfg(step_name)

        if is_last and (
            hasattr(estimator, "predict")
            or hasattr(estimator, "fit_predict")
            or hasattr(estimator, "score")
        ):
            return _PerViewEstimator(
                estimator=estimator,
                shared_params=shared_params,
                per_view_params=per_view_params,
            )

        if force_predict_path and (
            hasattr(estimator, "predict")
            or hasattr(estimator, "fit_predict")
            or hasattr(estimator, "score")
        ):
            return _PerViewEstimator(
                estimator=estimator,
                shared_params=shared_params,
                per_view_params=per_view_params,
            )

        if hasattr(estimator, "transform"):
            return _PerViewTransformer(
                estimator=estimator,
                shared_params=shared_params,
                per_view_params=per_view_params,
            )

        raise ValueError(
            f"Step '{step_name}' ({type(estimator).__name__}) expects single-view input while "
            "the pipeline is in multi-view mode. Add a fusion step first, or use a transformer/estimator that can run per-view."
        )

    def fit(self, X: Any, y: Any = None) -> "PolyPipeline":
        steps = self._validate_steps()
        current_mode = self._infer_mode(X)
        Xt = self._prepare_input(X, current_mode)

        fitted_steps: List[StepSpec] = []
        step_input_modes: List[DataMode] = []
        step_output_modes: List[DataMode] = []
        for i, (name, step) in enumerate(steps):
            is_last = i == len(steps) - 1
            step_input_modes.append(current_mode)
            if step == "passthrough":
                fitted_steps.append((name, step))
                step_output_modes.append(current_mode)
                continue

            est = clone(step)
            next_step = None if is_last else steps[i + 1][1]
            next_is_late_fusion = bool(
                BaseLateFusion is not None and isinstance(next_step, BaseLateFusion)
            )
            est = self._adapt_step_for_mode(
                name,
                est,
                current_mode,
                is_last,
                force_predict_path=next_is_late_fusion,
            )
            self._require_compatible_step_input(name, est, Xt)

            if is_last:
                est.fit(Xt, y)
                fitted_steps.append((name, est))
                step_output_modes.append(current_mode)
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
            step_output_modes.append(current_mode)

        self.steps_ = fitted_steps
        self.step_input_modes_ = step_input_modes
        self.step_output_modes_ = step_output_modes
        self.input_mode_ = cast(DataMode, self._infer_mode(X))
        self.mode_after_transforms_ = cast(DataMode, current_mode)
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
