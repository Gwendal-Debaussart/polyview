import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score

from polyview.cluster.mv_coreg_sc import MultiViewCoRegSpectralClustering
from polyview.cluster.mv_cotrain_sc import MultiViewCoTrainSpectralClustering


def _make_separable_views(n_samples=60, k=3, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, k, size=n_samples)

    centers1 = rng.normal(scale=10.0, size=(k, 4))
    centers2 = rng.normal(scale=10.0, size=(k, 3))

    x1 = centers1[y] + rng.normal(scale=0.3, size=(n_samples, 4))
    x2 = centers2[y] + rng.normal(scale=0.3, size=(n_samples, 3))
    return [x1, x2], y


class TestMultiViewCoRegSpectralClustering:
    def test_fit_predict_recovers_clusters(self):
        views, y = _make_separable_views(seed=0)
        model = MultiViewCoRegSpectralClustering(
            n_clusters=3, max_iter=10, random_state=0
        )
        labels = model.fit_predict(views)

        assert labels.shape == (60,)
        assert adjusted_rand_score(y, labels) > 0.7

    def test_fitted_attribute_shapes(self):
        views, _ = _make_separable_views(n_samples=40, seed=1)
        model = MultiViewCoRegSpectralClustering(
            n_clusters=3, max_iter=5, random_state=1
        ).fit(views)

        assert model.embedding_.shape == (40, 3 * 2)
        assert isinstance(model.objective_, list)
        assert len(model.objective_) == 5
        assert model.labels_.shape == (40,)

    def test_v_lambda_zero_still_produces_valid_clustering(self):
        views, _ = _make_separable_views(n_samples=40, seed=2)
        model = MultiViewCoRegSpectralClustering(
            n_clusters=3, max_iter=5, v_lambda=0.0, random_state=2
        ).fit(views)
        assert model.labels_.shape == (40,)
        assert set(np.unique(model.labels_)).issubset({0, 1, 2})

    def test_fit_sets_standard_multiview_attributes(self):
        views, _ = _make_separable_views(n_samples=30, seed=5)
        model = MultiViewCoRegSpectralClustering(
            n_clusters=3, max_iter=3, random_state=5
        ).fit(views)
        assert model.n_views_in_ == 2
        assert model.n_samples_ == 30
        assert model.n_features_in_ == [4, 3]

    def test_mismatched_sample_counts_raise_clear_error(self):
        x1 = np.random.rand(20, 4)
        x2 = np.random.rand(19, 3)
        with pytest.raises(ValueError, match="same number of samples"):
            MultiViewCoRegSpectralClustering(n_clusters=2, max_iter=2).fit([x1, x2])


class TestMultiViewCoTrainSpectralClustering:
    def test_fit_predict_recovers_clusters(self):
        views, y = _make_separable_views(seed=3)
        model = MultiViewCoTrainSpectralClustering(
            n_clusters=3, max_iter=10, random_state=3
        )
        labels = model.fit_predict(views)

        assert labels.shape == (60,)
        assert adjusted_rand_score(y, labels) > 0.7

    def test_fitted_attribute_shapes(self):
        views, _ = _make_separable_views(n_samples=40, seed=4)
        model = MultiViewCoTrainSpectralClustering(
            n_clusters=3, max_iter=5, random_state=4
        ).fit(views)

        assert model.embedding_.shape == (40, 3)
        assert isinstance(model.objective_, float)
        assert model.labels_.shape == (40,)

    def test_fit_sets_standard_multiview_attributes(self):
        views, _ = _make_separable_views(n_samples=30, seed=6)
        model = MultiViewCoTrainSpectralClustering(
            n_clusters=3, max_iter=3, random_state=6
        ).fit(views)
        assert model.n_views_in_ == 2
        assert model.n_samples_ == 30
        assert model.n_features_in_ == [4, 3]

    def test_mismatched_sample_counts_raise_clear_error(self):
        x1 = np.random.rand(20, 4)
        x2 = np.random.rand(19, 3)
        with pytest.raises(ValueError, match="same number of samples"):
            MultiViewCoTrainSpectralClustering(n_clusters=2, max_iter=2).fit([x1, x2])
