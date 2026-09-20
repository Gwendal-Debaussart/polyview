import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score

from polyview.cluster.mv_kmeans import MultiViewKMeans


def _make_two_view_blobs(n_samples=150, centers=3, seed=0):
    """Two well-separated views that share the same underlying cluster labels."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, centers, size=n_samples)

    centers1 = rng.normal(scale=15.0, size=(centers, 4))
    centers2 = rng.normal(scale=15.0, size=(centers, 5))

    x1 = centers1[y] + rng.normal(scale=0.5, size=(n_samples, 4))
    x2 = centers2[y] + rng.normal(scale=0.5, size=(n_samples, 5))
    return [x1, x2], y


class TestMultiViewKMeans:
    def test_fit_predict_recovers_well_separated_clusters(self):
        views, y = _make_two_view_blobs(seed=0)
        model = MultiViewKMeans(n_clusters=3, n_init=5, random_state=0)
        labels = model.fit_predict(views)

        assert labels.shape == (150,)
        assert adjusted_rand_score(y, labels) > 0.9

    def test_fitted_attributes_have_expected_shapes(self):
        views, _ = _make_two_view_blobs(n_samples=60, seed=1)
        model = MultiViewKMeans(n_clusters=3, n_init=3, random_state=1).fit(views)

        assert model.labels_.shape == (60,)
        assert len(model.centroids_) == 2
        assert model.centroids_[0].shape == (3, 4)
        assert model.centroids_[1].shape == (3, 5)
        assert model.weights_.shape == (2,)
        assert np.isclose(model.weights_.sum(), 1.0)
        assert model.n_iter_ >= 1

    def test_learn_weights_false_gives_equal_weights(self):
        views, _ = _make_two_view_blobs(n_samples=60, seed=2)
        model = MultiViewKMeans(
            n_clusters=3, n_init=2, learn_weights=False, random_state=2
        ).fit(views)
        assert np.allclose(model.weights_, 0.5)

    def test_predict_on_new_data_matches_fit_predict_semantics(self):
        views, y = _make_two_view_blobs(n_samples=100, seed=3)
        model = MultiViewKMeans(n_clusters=3, n_init=5, random_state=3).fit(views)
        preds = model.predict(views)
        # predict() re-derives labels from centroids; should closely match labels_
        assert adjusted_rand_score(model.labels_, preds) > 0.95

    def test_predict_before_fit_raises(self):
        model = MultiViewKMeans(n_clusters=2)
        with pytest.raises(Exception):
            model.predict([np.random.rand(5, 2), np.random.rand(5, 2)])

    def test_n_init_keeps_lowest_objective_run(self):
        views, _ = _make_two_view_blobs(n_samples=60, seed=4)
        single = MultiViewKMeans(n_clusters=3, n_init=1, random_state=4).fit(views)
        multi = MultiViewKMeans(n_clusters=3, n_init=10, random_state=4).fit(views)
        assert multi.objective_ <= single.objective_ + 1e-8

    def test_gamma_one_collapses_weight_onto_single_best_view(self):
        rng = np.random.default_rng(0)
        n = 90
        y = rng.integers(0, 3, size=n)
        centers = rng.normal(scale=15.0, size=(3, 4))
        x_good = centers[y] + rng.normal(scale=0.2, size=(n, 4))  # informative
        x_bad = rng.normal(scale=15.0, size=(n, 4))  # pure noise, uninformative

        model = MultiViewKMeans(n_clusters=3, gamma=1.0, n_init=5, random_state=0).fit(
            [x_good, x_bad]
        )

        assert np.count_nonzero(model.weights_) == 1
        assert model.weights_[0] == 1.0
