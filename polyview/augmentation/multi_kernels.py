from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

from polyview.dataset.multiviewdataset import MultiViewDataset
from polyview.utils.kernels import KernelName, KernelSpec

KernelSpecInput = Union[KernelSpec, KernelName]


class MultiKernel(BaseEstimator):
    """Generate multiple kernel-matrix views from one input matrix.

    Parameters
    ----------
    specs : sequence of KernelSpec or kernel-name strings, optional
        Kernel definitions to apply to the same input matrix. If omitted,
        defaults to three complementary kernels: ``linear``, ``rbf``, and
        ``polynomial``.
    view_names : sequence of str, optional
        Names to assign to the generated views.
    """

    def __init__(
        self,
        specs: Optional[Sequence[KernelSpecInput]] = None,
        view_names: Optional[Sequence[str]] = None,
    ) -> None:
        self.specs = specs
        self.view_names = view_names

    @staticmethod
    def _validate_input(X) -> np.ndarray:
        arr = np.asarray(X, dtype=float)
        if arr.ndim != 2:
            raise ValueError(
                f"X must be 2-D (n_samples, n_features), got shape {arr.shape}."
            )
        return arr

    @staticmethod
    def _to_spec(spec: KernelSpecInput) -> KernelSpec:
        if isinstance(spec, KernelSpec):
            return spec
        if isinstance(spec, str):
            return KernelSpec(kernel=spec)
        raise TypeError(
            "Each specs entry must be a KernelSpec or a kernel name string."
        )

    def _resolve_specs(self) -> list[KernelSpec]:
        if self.specs is None:
            return [
                KernelSpec("linear"),
                KernelSpec("rbf"),
                KernelSpec("polynomial"),
            ]
        specs = [self._to_spec(spec) for spec in self.specs]
        if len(specs) == 0:
            raise ValueError("specs must contain at least one kernel definition.")
        return specs

    def _resolve_view_names(self, specs: Sequence[KernelSpec]) -> list[str]:
        if self.view_names is not None:
            if len(self.view_names) != len(specs):
                raise ValueError(
                    f"view_names has {len(self.view_names)} entries but specs has {len(specs)}."
                )
            return list(self.view_names)

        names: list[str] = []
        for i, spec in enumerate(specs):
            if isinstance(spec.kernel, str):
                names.append(f"kernel_{spec.kernel}_{i}")
            else:
                names.append(f"kernel_custom_{i}")
        return names

    def fit(self, X, y=None) -> "MultiKernel":
        X = self._validate_input(X)
        self.n_features_in_ = X.shape[1]
        self.specs_ = self._resolve_specs()
        self.view_names_ = self._resolve_view_names(self.specs_)
        return self

    def transform(self, X) -> MultiViewDataset:
        check_is_fitted(self, ["specs_", "n_features_in_"])
        X = self._validate_input(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features but the transformer was fitted with {self.n_features_in_}."
            )
        kernel_views = [spec.build(X) for spec in self.specs_]
        return MultiViewDataset(views=kernel_views, view_names=self.view_names_)

    def fit_transform(self, X, y=None) -> MultiViewDataset:
        return self.fit(X, y).transform(X)


def multi_kernels(
    X,
    specs: Optional[Sequence[KernelSpecInput]] = None,
    labels=None,
    view_names: Optional[Sequence[str]] = None,
) -> MultiViewDataset:
    """Build a MultiViewDataset of kernel matrices from a single matrix."""
    dataset = MultiKernel(specs=specs, view_names=view_names).fit_transform(X)
    if labels is not None:
        dataset.labels = labels
    return dataset
