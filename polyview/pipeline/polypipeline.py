from __future__ import annotations

import importlib
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

    Parameters
    ----------
    steps : sequence of (str, estimator) tuples
        List of (name, transform) tuples that are chained in the pipeline order.
    per_view_step_params : dict or None, default=None
        Optional per-view hyperparameters for specific steps.
    name : str or None, default=None
        Display name for the pipeline, shown in diagrams. If None, defaults to "PolyPipeline flow".
    """

    def __init__(
        self,
        steps: Sequence[StepSpec],
        per_view_step_params: Optional[
            Dict[str, Union[Dict[str, Any], Sequence[Optional[Dict[str, Any]]]]]
        ] = None,
        name: Optional[str] = None,
    ) -> None:
        self.steps = steps
        self.per_view_step_params = per_view_step_params
        self.name = name if name is not None else "PolyPipeline flow"

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

        if current_mode in ("mv", "lf"):
            self.n_views_in_ = len(self._as_mv(X))
        else:
            self.n_views_in_ = 1

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
    def _transition_behavior(
        mode_in: DataMode,
        mode_out: DataMode,
        branch_count: int = 3,
    ) -> str:
        if mode_in == "sv" and mode_out == "mv":
            return f"split 1 -> {branch_count} branches"
        if mode_in == "mv" and mode_out == "mv":
            return f"{branch_count} parallel branches"
        if mode_in == "mv" and mode_out == "sv":
            return f"merge {branch_count} -> 1"
        if mode_in == "mv" and mode_out == "lf":
            return f"per-view labels ({branch_count} -> {branch_count})"
        if mode_in == "lf" and mode_out == "sv":
            return f"late-fusion ({branch_count} -> 1)"
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

    @staticmethod
    def _is_label_producing_step(step: Any) -> bool:
        """Check if a step produces labels (clustering, classification, or late-fusion)."""
        if step == "passthrough":
            return False

        # Unwrap wrapped estimators
        actual_step = step
        if isinstance(step, (_PerViewTransformer, _PerViewEstimator)):
            actual_step = step.estimator

        # Check for BaseLateFusion (late-fusion steps)
        if isinstance(actual_step, BaseLateFusion):
            return True

        # Check for labels_ or classes_ attributes (fitted estimators)
        if hasattr(actual_step, "labels_") or hasattr(actual_step, "classes_"):
            return True

        return False

    def draw_diagram(
        self,
        start_mode: Optional[Literal["mv", "sv"]] = None,
    ) -> str:
        """Render a readable ASCII diagram of pipeline flow.

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

        Examples
        --------
        >>> pipe = PolyPipeline([
        ...     ("rp", RandomProjectionViews(n_views=3)),
        ...     ("scale", StandardScaler()),
        ...     ("cluster", MultiViewKMeans(n_clusters=3, random_state=0))
        ... ])
        >>> print(pipe.draw_diagram(start_mode='sv'))
        input
          ↓ (1 view)
        [RandomProjectionViews]
          ↓ ↓ ↓ (3 views)
        [StandardScaler]
          ↓ ↓ ↓ (3 views)
        [MultiViewKMeans]
          ↓ (1 output)
        output
        """
        fitted = hasattr(self, "steps_")
        explicit_start_mode = start_mode is not None
        steps = list(
            self._validate_steps()
            if explicit_start_mode
            else (self.steps_ if fitted else self._validate_steps())
        )
        use_fitted_inference = fitted and not explicit_start_mode

        if start_mode is None:
            if fitted:
                mode: DataMode = cast(DataMode, self.input_mode_)
            else:
                lines = [
                    "PolyPipeline diagram (unfitted)",
                    "Pass start_mode='mv' or start_mode='sv' to simulate flow.",
                ]
                diagram = "\n".join(lines)
                print(diagram)
                return diagram
        else:
            mode = cast(DataMode, start_mode)

        branch_count = int(getattr(self, "n_views_in_", 3))
        lines = [self.name, "input"]

        for i, (name, step) in enumerate(steps, start=1):
            is_last = i == len(steps)
            next_step = None if is_last else steps[i][1]
            mode_out = self._simulate_step_mode(
                mode_in=mode,
                step=step,
                is_last=is_last,
                next_step=next_step,
                fitted=use_fitted_inference,
            )

            # Compute transition behavior, accounting for label-producing steps
            produces_labels = self._is_label_producing_step(step)
            if produces_labels and mode == mode_out:
                # Label-producing step with no mode change
                if mode == "sv":
                    transition_text = "→ labels (1-D)"
                elif mode == "mv":
                    transition_text = "→ labels (per-view)"
                elif mode == "lf":
                    transition_text = "→ fused labels (1-D)"
                else:
                    transition_text = "→ labels"
            else:
                transition_text = self._transition_behavior(
                    mode,
                    mode_out,
                    branch_count=branch_count,
                )

            if mode in ("mv", "lf"):
                lines.append(f"  ↓ ↓ ↓ ({transition_text})")
            else:
                lines.append(f"  ↓ ({transition_text})")

            # Step name
            step_label = self._step_label(name, step)
            lines.append(f"[{step_label}]")

            mode = mode_out

        # Final connector
        if mode == "mv":
            lines.append("  ↓ ↓ ↓ (output - multiview)")
        elif mode == "lf":
            lines.append("  ↓ ↓ ↓ (output - late-fusion)")
        else:
            lines.append("  ↓ (output)")

        lines.append("output")
        diagram = "\n".join(lines)
        print(diagram)
        return diagram

    def draw_diagram_nx(
        self,
        start_mode: Optional[Literal["mv", "sv"]] = None,
        *,
        ax: Any = None,
        show: bool = True,
        node_size: int = 2300,
        mode_colors: Optional[Dict[str, str]] = None,
        title: Optional[str] = None,
        node_text_color: str = "white",
        node_border_color: str = "#1A1A1A",
        edge_color: str = "#333333",
        transition_text_color: str = "#222222",
        show_legend: bool = True,
        show_title: bool = True,
    ) -> Any:
        """Draw a mode-aware pipeline diagram using NetworkX.

        Parameters
        ----------
        start_mode : {"mv", "sv"} or None, default=None
            Optional starting mode for preview. Useful before fitting.
            - ``None``: use fitted input mode if available.
            - ``"mv"``/``"sv"``: simulate flow from that start mode.
        ax : matplotlib.axes.Axes or None, default=None
            Matplotlib axis to draw onto. If None, a new figure is created.
        show : bool, default=True
            Whether to call ``matplotlib.pyplot.show()`` after drawing.
        node_size : int, default=2300
            Minimum size hint for node rectangles.
        mode_colors : dict or None, default=None
            Optional color overrides for node modes and types. Supported keys are
            ``"mv"``, ``"sv"``, ``"lf"``, ``"labels"`` (clustering/classification steps),
            ``"output"``, and ``"default"``.
        title : str or None, default=None
            Plot title. If None, uses the pipeline's name attribute.
        node_text_color : str, default="white"
            Color of node labels.
        node_border_color : str, default="#1A1A1A"
            Stroke color for node borders.
        edge_color : str, default="#333333"
            Arrow color.
        transition_text_color : str, default="#222222"
            Color of edge transition labels.
        show_legend : bool, default=True
            Whether to display a legend showing color meanings (MV/SV/LF modes and label output).
        show_title : bool, default=True
            Whether to display the plot title.

        Returns
        -------
        networkx.DiGraph
            Directed graph with mode and transition metadata.
        """

        # Check for required libraries without hard dependency
        try:
            nx = importlib.import_module("networkx")
        except ImportError as exc:
            raise ImportError(
                "draw_diagram_nx requires networkx. Install with `pip install networkx`."
            ) from exc

        try:
            plt = importlib.import_module("matplotlib.pyplot")
        except ImportError as exc:
            raise ImportError(
                "draw_diagram_nx requires matplotlib. Install with `pip install matplotlib`."
            ) from exc

        fitted = hasattr(self, "steps_")
        explicit_start_mode = start_mode is not None
        steps = list(
            self._validate_steps()
            if explicit_start_mode
            else (self.steps_ if fitted else self._validate_steps())
        )
        use_fitted_inference = fitted and not explicit_start_mode

        if start_mode is None:
            if fitted:
                mode: DataMode = cast(DataMode, self.input_mode_)
            else:
                raise ValueError(
                    "Unfitted pipeline: pass start_mode='mv' or start_mode='sv' to simulate flow."
                )
        else:
            mode = cast(DataMode, start_mode)

        if title is None:
            title = self.name

        graph = nx.DiGraph(name="PolyPipeline")
        graph.add_node("input", label="input", mode=mode)
        step_behaviors: List[str] = []
        branch_count = int(getattr(self, "n_views_in_", 3))

        previous_node = "input"
        for i, (name, step) in enumerate(steps, start=1):
            is_last = i == len(steps)
            next_step = None if is_last else steps[i][1]
            mode_out = self._simulate_step_mode(
                mode_in=mode,
                step=step,
                is_last=is_last,
                next_step=next_step,
                fitted=use_fitted_inference,
            )

            node_id = f"step_{i}"
            produces_labels = self._is_label_producing_step(step)
            graph.add_node(
                node_id,
                label=self._step_label(name, step),
                mode=mode_out,
                mode_in=mode,
                produces_labels=produces_labels,
            )
            graph.add_edge(previous_node, node_id, transition="")

            # Compute transition behavior, accounting for label-producing steps
            if produces_labels and mode == mode_out:
                if mode == "sv":
                    behavior = "→ labels (1-D)"
                elif mode == "mv":
                    behavior = "→ labels (per-view)"
                elif mode == "lf":
                    behavior = "→ fused labels (1-D)"
                else:
                    behavior = "→ labels"
            else:
                behavior = self._transition_behavior(
                    mode,
                    mode_out,
                    branch_count=branch_count,
                )

            step_behaviors.append(behavior)

            previous_node = node_id
            mode = mode_out

        graph.add_node("output", label="output", mode=mode)
        graph.add_edge(previous_node, "output", transition="")

        # Show each step's mode-transition annotation on its outgoing edge.
        if len(steps) > 0:
            for i in range(1, len(steps) + 1):
                source = f"step_{i}"
                target = "output" if i == len(steps) else f"step_{i + 1}"
                graph.edges[source, target]["transition"] = step_behaviors[i - 1]


        ordered_nodes = (
            ["input"] + [f"step_{i}" for i in range(1, len(steps) + 1)] + ["output"]
        )

        # Color priority:
        # 1. Output node
        # 2. Label-producing step with 1D output (purple)
        # 3. All others -> mode color (LF for multiple per-view labels, MV for multi-view data)

        palette = {
            "mv": "#072AC8",
            "sv": "#FF8200",
            "lf": "#54A24B",
            "labels": "#7C3AED",
            "output": "#9E9E9E",
            "default": "#CCCCCC",
        }
        if mode_colors:
            palette.update(mode_colors)

        # Determine node colors based on mode and label status
        node_colors = []
        for node in ordered_nodes:
            if node == "output":
                node_colors.append(palette.get("output", "#9E9E9E"))
            else:
                node_data = graph.nodes[node]
                produces_labels = node_data.get("produces_labels", False)
                node_mode = node_data.get("mode", "sv")

                if produces_labels:
                    node_colors.append(palette.get("labels", "#7C3AED"))
                else:
                    node_colors.append(palette.get(node_mode, palette.get("default", "#CCCCCC")))

        # Draw the diagram of operations with NetworkX and Matplotlib
        try:
            from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
        except ImportError:
            FancyBboxPatch = None
            FancyArrowPatch = None

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, max(10, len(ordered_nodes) * 1.5)))
        else:
            fig = ax.get_figure()

        # Layout: vertical top-to-bottom
        n_nodes = len(ordered_nodes)
        y_spacing = 1.0
        positions = {node: (0, -i * y_spacing) for i, node in enumerate(ordered_nodes)}

        # Draw edges first (so they appear behind nodes)
        for i, source_node in enumerate(ordered_nodes[:-1]):
            target_node = ordered_nodes[i + 1]
            x1, y1 = positions[source_node]
            x2, y2 = positions[target_node]

            # Draw arrow
            if FancyArrowPatch is not None:
                arrow = FancyArrowPatch(
                    (x1, y1 - 0.25), (x2, y2 + 0.25),
                    arrowstyle='-|>', mutation_scale=18, lw=1.5,
                    color=edge_color, zorder=1
                )
                ax.add_patch(arrow)

            # Add transition label on edge (to the right of arrow)
            transition_label = graph.edges[source_node, target_node].get("transition", "")
            if transition_label:
                mid_y = (y1 + y2) / 2
                ax.text(0.15, mid_y, transition_label, ha='left', va='center',
                       fontsize=10, color=transition_text_color, zorder=2,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='none', alpha=0.8))

        # Draw nodes
        for node_idx, node in enumerate(ordered_nodes):
            x, y = positions[node]
            label = graph.nodes[node]["label"]
            color = node_colors[node_idx]

            # Wrap text if too long: split on word boundaries, max 2 lines
            max_width_chars = 20
            if len(label) > max_width_chars:
                # Try to split on spaces or underscores
                words = label.replace('_', ' ').split()
                lines = []
                current_line = []
                for word in words:
                    test_line = ' '.join(current_line + [word])
                    if len(test_line) <= max_width_chars:
                        current_line.append(word)
                    else:
                        if current_line:
                            lines.append(' '.join(current_line))
                        current_line = [word]
                if current_line:
                    lines.append(' '.join(current_line))

                # Limit to 2 lines max: if 3+ lines, try to reflow into 2
                if len(lines) > 2:
                    lines = lines[:2]

                wrapped_label = '\n'.join(lines)
            else:
                wrapped_label = label

            # Create temporary text to measure size
            temp_text = ax.text(x, y, wrapped_label, fontsize=10, ha='center', va='center',
                               visible=False, weight='bold')
            fig.canvas.draw()

            # Get text bbox in data coordinates
            bbox = temp_text.get_window_extent(renderer=fig.canvas.get_renderer())
            bbox_data = ax.transData.inverted().transform(bbox)
            text_width = bbox_data[1, 0] - bbox_data[0, 0]
            text_height = bbox_data[1, 1] - bbox_data[0, 1]
            temp_text.remove()

            # Box with padding - tight around text
            pad_x, pad_y = 0.03, 0.08
            box_width = max(text_width + 2 * pad_x, 0.5)
            box_height = max(text_height + 2 * pad_y, 0.35)
            box_x = x - box_width / 2
            box_y = y - box_height / 2

            # Draw box
            if FancyBboxPatch is not None:
                box = FancyBboxPatch(
                    (box_x, box_y), box_width, box_height,
                    boxstyle="round,pad=0.03", edgecolor=node_border_color,
                    facecolor=color, linewidth=1.5, zorder=3
                )
                ax.add_patch(box)

            # Draw label
            ax.text(x, y, wrapped_label, fontsize=10, ha='center', va='center',
                   color=node_text_color, weight='bold', zorder=4)

        # Set axis limits with minimal padding
        margin = 0.25
        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(-(n_nodes - 1) * y_spacing - margin, margin)
        ax.axis('off')

        if show_legend:
            modes_in_graph = set()
            has_labels = False

            try:
                from matplotlib.patches import Patch
            except ImportError:
                Patch = None

            for node in graph.nodes():
                node_mode = cast(str, graph.nodes[node].get("mode", "sv"))
                if node != "output":  # Don't count output node's mode for mode legend
                    modes_in_graph.add(node_mode)
                produces_labels = cast(
                    bool, graph.nodes[node].get("produces_labels", False)
                )

                if produces_labels:
                    has_labels = True

            # Build legend with only the modes present in this pipeline
            legend_elements = []

            if Patch is not None:
                if "mv" in modes_in_graph:
                    legend_elements.append(
                        Patch(
                            facecolor=palette.get("mv", "#072AC8"),
                            edgecolor=node_border_color,
                            label="Multi-view (MV)",
                        )
                    )
                if "sv" in modes_in_graph:
                    legend_elements.append(
                        Patch(
                            facecolor=palette.get("sv", "#9E9E9E"),
                            edgecolor=node_border_color,
                            label="Single-view (SV)",
                        )
                    )

                if "lf" in modes_in_graph:
                    legend_elements.append(
                        Patch(
                            facecolor=palette.get("lf", "#54A24B"),
                            edgecolor=node_border_color,
                            label="Multi-Labels",
                        )
                    )

                if has_labels:
                    legend_elements.append(
                        Patch(
                            facecolor=palette.get("labels", "#7C3AED"),
                            edgecolor=node_border_color,
                            label="Labels",
                        )
                    )

                legend_elements.append(
                    Patch(
                        facecolor=palette.get("output", "#9E9E9E"),
                        edgecolor=node_border_color,
                        label="Output node",
                    )
                )

            if legend_elements:
                ax.legend(
                    handles=legend_elements,
                    loc="lower center",
                    bbox_to_anchor=(0.5, -0.15),
                    ncol=3,
                    frameon=True,
                )

        if show_title:
            ax.set_title(title)

        if show:
            plt.show()

        return graph

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
