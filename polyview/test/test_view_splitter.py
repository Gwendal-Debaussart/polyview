import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from polyview.augmentation.view_splitter import ViewSplitter
from polyview.dataset.multiviewdataset import MultiViewDataset
from polyview.fusion.early import ConcatFusion


def _make_views(n_samples=30, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.normal(size=(n_samples, d)) for d in (3, 5, 2)]


class TestViewSplitter:
    def test_splits_columns_into_views(self):
        views = _make_views()
        X = np.hstack(views)
        mvd = ViewSplitter(n_features=[3, 5, 2]).fit_transform(X)

        assert isinstance(mvd, MultiViewDataset)
        assert mvd.view_names == ["view_0", "view_1", "view_2"]
        for got, expected in zip(mvd.views, views):
            assert np.array_equal(got, expected)

    def test_is_inverse_of_concat_fusion(self):
        views = _make_views(seed=1)
        X = ConcatFusion().fit_transform(views)
        mvd = ViewSplitter(
            n_features=[3, 5, 2], view_names=["a", "b", "c"]
        ).fit_transform(X)

        assert mvd.view_names == ["a", "b", "c"]
        assert np.array_equal(mvd.to_numpy(), X)

    def test_fitted_attributes(self):
        splitter = ViewSplitter(n_features=[3, 5, 2]).fit(np.hstack(_make_views()))
        assert splitter.n_views == 3
        assert splitter.n_views_in_ == 3
        assert splitter.n_features_in_ == 10
        assert list(splitter.split_indices_) == [3, 8]

    @pytest.mark.parametrize(
        "n_features, match",
        [
            (None, "must be given"),
            ([], "positive integers"),
            ([3, 0, 7], "positive integers"),
            ([3, 5], "sums to 8"),
        ],
    )
    def test_invalid_n_features_raise(self, n_features, match):
        with pytest.raises(ValueError, match=match):
            ViewSplitter(n_features=n_features).fit(np.zeros((4, 10)))

    def test_view_names_length_must_match(self):
        with pytest.raises(ValueError, match="view_names"):
            ViewSplitter(n_features=[4, 6], view_names=["a"]).fit(np.zeros((4, 10)))

    def test_input_must_be_2d(self):
        with pytest.raises(ValueError, match="2-D"):
            ViewSplitter(n_features=[2]).fit(np.zeros(4))

    def test_transform_checks_fit_and_width(self):
        splitter = ViewSplitter(n_features=[4, 6])
        with pytest.raises(NotFittedError):
            splitter.transform(np.zeros((4, 10)))
        splitter.fit(np.zeros((4, 10)))
        with pytest.raises(ValueError, match="columns"):
            splitter.transform(np.zeros((4, 9)))
