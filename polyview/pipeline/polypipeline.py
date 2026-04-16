from __future__ import annotations

from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
)

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.utils.validation import check_is_fitted

from polyview.base import BaseLateFusion, BaseMultiView


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
    - sv -> mv
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
        if hasattr(X, "_views"):
            return True
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
    def _as_mv(X: Any) -> List[np.ndarray]:
        if hasattr(X, "_views"):
            return [np.asarray(v, dtype=float) for v in X._views]
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
        if previous == "sv" and current == "mv":
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
            next_is_late_fusion = bool(isinstance(next_step, BaseLateFusion))
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
                    f"Unsupported mode transition after step '{name}': {current_mode} -> {new_mode}."
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

    @staticmethod
    def _is_random_projection_step(step: Any) -> bool:
        cls_name = type(step).__name__.lower()
        mod_name = type(step).__module__.lower()
        return (
            "randomprojection" in cls_name
            or "random_projections" in mod_name
            or "augmentation" in mod_name
        )

    @staticmethod
    def _base_mv_step_output_mode(step: Any) -> DataMode:
        out = getattr(step, "output", None)
        if isinstance(out, str):
            if out == "list":
                return "mv"
            if out in ("concat", "mean"):
                return "sv"
        return "mv"

    def _simulate_step_mode(
        self,
        mode_in: DataMode,
        step: Any,
        is_last: bool,
        next_step: Any,
        *,
        fitted: bool,
    ) -> DataMode:
        if step == "passthrough":
            return mode_in

        # Terminal predictive/cluster steps produce label-like 1-D outputs,
        # represented here as single-view mode for diagram readability.
        if is_last:
            if isinstance(step, _PerViewEstimator):
                return "lf"
            if (
                hasattr(step, "labels_")
                or hasattr(step, "predict")
                or hasattr(step, "fit_predict")
            ):
                return "sv"

        if fitted:
            if isinstance(step, _PerViewTransformer):
                return "mv"
            if isinstance(step, _PerViewEstimator):
                return "lf"

        if mode_in == "sv":
            if self._is_random_projection_step(step):
                return "mv"
            if isinstance(step, BaseMultiView):
                return self._base_mv_step_output_mode(step)
            return "sv"

        if mode_in == "mv":
            if isinstance(step, BaseMultiView):
                return self._base_mv_step_output_mode(step)
            next_is_late_fusion = bool(isinstance(next_step, BaseLateFusion))
            if next_is_late_fusion:
                return "lf"
            if is_last:
                return "sv"
            return "mv"

        # late-fusion mode
        if mode_in == "lf":
            if isinstance(step, BaseLateFusion):
                return "sv"
            return "lf"

        return mode_in

    @staticmethod
    def _transition_behavior(mode_in: DataMode, mode_out: DataMode) -> str:
        if mode_in == "sv" and mode_out == "mv":
            return "split 1 -> 3 branches"
        if mode_in == "mv" and mode_out == "mv":
            return "3 parallel branches"
        if mode_in == "mv" and mode_out == "sv":
            return "merge 3 -> 1"
        if mode_in == "mv" and mode_out == "lf":
            return "per-view predictions (3x 1-D)"
        if mode_in == "lf" and mode_out == "sv":
            return "late-fusion (many -> 1)"
        if mode_in == mode_out:
            return "no branch change"
        return f"{mode_in} -> {mode_out}"

    @staticmethod
    def _step_label(name: str, step: Any) -> str:
        if step == "passthrough":
            return f"{name}: passthrough"
        if isinstance(step, _PerViewTransformer):
            return f"{name}: {type(step.estimator).__name__}"
        if isinstance(step, _PerViewEstimator):
            return f"{name}: {type(step.estimator).__name__}"
        return f"{name}: {type(step).__name__}"

    def draw_diagram(
        self,
        start_mode: Optional[Literal["mv", "sv"]] = None,
    ) -> str:
        """Render a readable text diagram of pipeline flow.

        Parameters
        ----------
        start_mode : {"mv", "sv"} or None, default=None
            Optional starting mode for preview. Useful before fitting.
            - ``None``: use fitted input mode if available.
            - ``"mv"``/``"sv"``: simulate flow from that start mode.

        Returns
        -------
        str
            Multi-line diagram string (also printed).
        """
        fitted = hasattr(self, "steps_")
        steps = list(self.steps_ if fitted else self._validate_steps())

        if start_mode is None:
            if fitted:
                mode: DataMode = cast(DataMode, self.input_mode_)
            else:
                lines = [
                    "PolyPipeline Diagram [unfitted]",
                    "Provide start_mode='mv' or start_mode='sv' for a full preview.",
                    "Configured steps:",
                ]
                for name, step in steps:
                    lines.append(f"  - {self._step_label(name, step)}")
                diagram = "\n".join(lines)
                print(diagram)
                return diagram
        else:
            mode = cast(DataMode, start_mode)

        lines = [
            f"PolyPipeline Diagram [{'fitted' if fitted else 'preview'}]",
            f"Start: {mode}",
            "",
        ]

        for i, (name, step) in enumerate(steps, start=1):
            is_last = i == len(steps)
            next_step = None if is_last else steps[i][1]
            mode_out = self._simulate_step_mode(
                mode_in=mode,
                step=step,
                is_last=is_last,
                next_step=next_step,
                fitted=fitted,
            )
            behavior = self._transition_behavior(mode, mode_out)
            lines.append(
                f"{i:02d}. [{mode}] -- {self._step_label(name, step)} -- [{mode_out}]"
            )
            lines.append(f"    behavior: {behavior}")
            mode = mode_out

        lines.extend(["", f"End: {mode}"])
        diagram = "\n".join(lines)
        print(diagram)
        return diagram

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
