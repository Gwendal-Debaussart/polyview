import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score

from polyview.cluster.mv_nmf import MultiViewNMF


def _make_nonneg_views(n_samples=90, k=3, seed=0):
    """Two non-negative views sharing the same underlying cluster labels."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, k, size=n_samples)

    centers1 = rng.uniform(1.0, 10.0, size=(k, 5))
    centers2 = rng.uniform(1.0, 10.0, size=(k, 6))

    x1 = np.clip(centers1[y] + rng.normal(scale=0.2, size=(n_samples, 5)), 0, None)
    x2 = np.clip(centers2[y] + rng.normal(scale=0.2, size=(n_samples, 6)), 0, None)
    return [x1, x2], y


class TestMultiViewNMF:
    def test_fit_predict_recovers_clusters(self):
        views, y = _make_nonneg_views(seed=0)
        model = MultiViewNMF(n_components=3, n_init=5, random_state=0)
        labels = model.fit_predict(views)

        assert labels.shape == (90,)
        assert adjusted_rand_score(y, labels) > 0.8

    def test_fitted_attribute_shapes(self):
        views, _ = _make_nonneg_views(n_samples=60, seed=1)
        model = MultiViewNMF(n_components=3, n_init=2, random_state=1).fit(views)

        assert model.H_.shape == (60, 3)
        assert np.all(model.H_ >= 0)
        assert len(model.W_) == 2
        assert model.W_[0].shape == (3, 5)
        assert model.W_[1].shape == (3, 6)
        assert model.weights_.shape == (2,)
        assert model.reconstruction_errors_.shape == (2,)
        assert model.labels_.tolist() == np.argmax(model.H_, axis=1).tolist()

    def test_negative_values_are_clipped_with_warning(self):
        views, _ = _make_nonneg_views(n_samples=40, seed=2)
        views[0] = views[0] - 5.0  # introduce negatives
        with pytest.warns(UserWarning, match="clipped to zero"):
            MultiViewNMF(n_components=2, n_init=1, random_state=2).fit(views)

    def test_fit_transform_returns_H(self):
        views, _ = _make_nonneg_views(n_samples=40, seed=3)
        model = MultiViewNMF(n_components=3, n_init=2, random_state=3)
        H = model.fit_transform(views)
        assert np.array_equal(H, model.H_)

    def test_transform_projects_new_samples_into_same_space(self):
        views, _ = _make_nonneg_views(n_samples=80, seed=4)
        train_views = [v[:60] for v in views]
        test_views = [v[60:] for v in views]

        model = MultiViewNMF(n_components=3, n_init=3, random_state=4).fit(train_views)
        H_new = model.transform(test_views)

        assert H_new.shape == (20, 3)
        assert np.all(H_new >= 0)

    def test_equal_weights_by_default(self):
        views, _ = _make_nonneg_views(n_samples=40, seed=5)
        model = MultiViewNMF(n_components=2, n_init=1, random_state=5).fit(views)
        assert np.allclose(model.weights_, 0.5)

    def test_gamma_one_collapses_weight_onto_single_best_view(self):
        rng = np.random.default_rng(1)
        n = 60
        y = rng.integers(0, 3, size=n)
        centers = rng.uniform(1.0, 10.0, size=(3, 5))
        x_good = np.clip(centers[y] + rng.normal(scale=0.1, size=(n, 5)), 0, None)
        x_bad = np.clip(rng.uniform(1.0, 10.0, size=(n, 5)), 0, None)  # uninformative

        model = MultiViewNMF(
            n_components=3, gamma=1.0, learn_weights=True, n_init=5, random_state=1
        ).fit([x_good, x_bad])

        assert np.count_nonzero(model.weights_) == 1
        assert model.weights_[0] == 1.0
