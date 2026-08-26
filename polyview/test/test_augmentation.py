import numpy as np
import pytest

from polyview.augmentation.multi_kernels import MultiKernel, multi_kernels
from polyview.augmentation.random_projections import (
    RandomProjectionViews,
    random_projection,
)
from polyview.augmentation.random_subspace import RandomSubspaceViews, random_subspace
from polyview.dataset.multiviewdataset import MultiViewDataset


def _make_X(n_samples=50, n_features=20, seed=0):
    return np.random.default_rng(seed).normal(size=(n_samples, n_features))


class TestRandomProjectionViews:
    def test_fit_transform_returns_multiview_dataset(self):
        X = _make_X()
        mvd = RandomProjectionViews(n_views=3, random_state=0).fit_transform(X)
        assert isinstance(mvd, MultiViewDataset)
        assert mvd.n_views == 3
        assert mvd.n_samples == 50

    def test_default_n_components_is_half_features(self):
        X = _make_X(n_features=20)
        mvd = RandomProjectionViews(n_views=2, random_state=0).fit_transform(X)
        for v in mvd.views:
            assert v.shape == (50, 10)

    def test_per_view_component_list(self):
        X = _make_X(n_features=20)
        mvd = RandomProjectionViews(
            n_views=2, n_components=[4, 6], random_state=0
        ).fit_transform(X)
        assert mvd.views[0].shape == (50, 4)
        assert mvd.views[1].shape == (50, 6)

    def test_component_list_length_mismatch_raises(self):
        X = _make_X()
        with pytest.raises(ValueError, match="n_components has"):
            RandomProjectionViews(n_views=3, n_components=[4, 6]).fit(X)

    def test_sparse_method_runs(self):
        X = _make_X(n_samples=100, n_features=30)
        mvd = RandomProjectionViews(
            n_views=2, method="sparse", random_state=1
        ).fit_transform(X)
        assert mvd.n_views == 2

    def test_different_random_states_give_different_projections(self):
        X = _make_X(seed=2)
        mvd1 = RandomProjectionViews(n_views=2, random_state=0).fit_transform(X)
        mvd2 = RandomProjectionViews(n_views=2, random_state=1).fit_transform(X)
        assert not np.allclose(mvd1.views[0], mvd2.views[0])

    def test_functional_wrapper_attaches_labels(self):
        X = _make_X(n_samples=30, seed=3)
        labels = np.arange(30) % 3
        mvd = random_projection(X, n_views=2, labels=labels, random_state=0)
        assert mvd.labels is not None
        assert np.array_equal(mvd.labels, labels)

    def test_transform_wrong_feature_count_raises(self):
        X = _make_X(n_features=20, seed=4)
        model = RandomProjectionViews(n_views=2, random_state=0).fit(X)
        with pytest.raises(ValueError, match="features but the transformer"):
            model.transform(_make_X(n_features=5, seed=4))


class TestRandomSubspaceViews:
    def test_fit_transform_default_sqrt_features(self):
        X = _make_X(n_features=16)
        mvd = RandomSubspaceViews(n_views=3, random_state=0).fit_transform(X)
        assert mvd.n_views == 3
        for v in mvd.views:
            assert v.shape == (50, 4)  # sqrt(16) = 4

    def test_explicit_n_features_per_view(self):
        X = _make_X(n_features=20)
        mvd = RandomSubspaceViews(
            n_views=2, n_features_per_view=5, random_state=0
        ).fit_transform(X)
        for v in mvd.views:
            assert v.shape == (50, 5)

    def test_feature_indices_are_unique_per_view(self):
        X = _make_X(n_features=20)
        model = RandomSubspaceViews(
            n_views=4, n_features_per_view=5, random_state=0
        ).fit(X)
        for indices in model.feature_indices_:
            assert len(set(indices.tolist())) == 5

    def test_invalid_n_features_per_view_raises(self):
        X = _make_X(n_features=10)
        with pytest.raises(ValueError, match="n_features_per_view must be"):
            RandomSubspaceViews(n_features_per_view=11).fit(X)

    def test_selected_columns_actually_match_original_data(self):
        X = _make_X(n_features=10, seed=5)
        model = RandomSubspaceViews(
            n_views=1, n_features_per_view=4, random_state=0
        ).fit(X)
        mvd = model.transform(X)
        expected = X[:, model.feature_indices_[0]]
        assert np.array_equal(mvd.views[0], expected)

    def test_functional_wrapper_attaches_labels(self):
        X = _make_X(n_samples=30, seed=6)
        labels = np.arange(30) % 2
        mvd = random_subspace(X, n_views=2, labels=labels, random_state=0)
        assert np.array_equal(mvd.labels, labels)


class TestMultiKernel:
    def test_default_specs_produce_three_kernel_views(self):
        X = _make_X(n_samples=25, n_features=6)
        mvd = MultiKernel().fit_transform(X)
        assert mvd.n_views == 3
        for K in mvd.views:
            assert K.shape == (25, 25)

    def test_view_names_default_to_kernel_names(self):
        X = _make_X(n_samples=20, n_features=5)
        mvd = MultiKernel().fit_transform(X)
        assert mvd.view_names == [
            "kernel_linear_0",
            "kernel_rbf_1",
            "kernel_polynomial_2",
        ]

    def test_custom_view_names(self):
        X = _make_X(n_samples=20, n_features=5)
        mvd = MultiKernel(view_names=["a", "b", "c"]).fit_transform(X)
        assert mvd.view_names == ["a", "b", "c"]

    def test_string_specs_are_accepted(self):
        X = _make_X(n_samples=20, n_features=5)
        mvd = MultiKernel(specs=["linear", "rbf"]).fit_transform(X)
        assert mvd.n_views == 2

    def test_view_names_length_mismatch_raises(self):
        X = _make_X(n_samples=20, n_features=5)
        with pytest.raises(ValueError, match="view_names has"):
            MultiKernel(specs=["linear", "rbf"], view_names=["only_one"]).fit(X)

    def test_functional_wrapper_attaches_labels(self):
        X = _make_X(n_samples=20, seed=7)
        labels = np.arange(20) % 2
        mvd = multi_kernels(X, labels=labels)
        assert np.array_equal(mvd.labels, labels)
