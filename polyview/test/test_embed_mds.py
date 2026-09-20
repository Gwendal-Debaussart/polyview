import numpy as np
import pytest
from sklearn.metrics import pairwise_distances

from polyview.embed.multiviewmds import MultiViewMDS


def _make_views(n_samples=40, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n_samples, 2))
    x1 = z @ rng.normal(size=(2, 5)) + 0.05 * rng.normal(size=(n_samples, 5))
    x2 = z @ rng.normal(size=(2, 4)) + 0.05 * rng.normal(size=(n_samples, 4))
    return [x1, x2]


class TestMultiViewMDS:
    def test_fit_produces_valid_embedding(self):
        views = _make_views(seed=0)
        model = MultiViewMDS(n_components=2, max_iter=50, random_state=0)
        model.fit(views)

        assert model.embedding_.shape == (40, 2)
        assert np.all(np.isfinite(model.embedding_))
        assert model.view_weights_.shape == (2,)
        assert np.isclose(model.view_weights_.sum(), 1.0)
        assert np.isfinite(model.stress_)
        assert model.view_stress_.shape == (2,)
        assert model.n_iter_ >= 1

    def test_fit_transform_returns_embedding(self):
        views = _make_views(seed=1)
        model = MultiViewMDS(n_components=2, max_iter=50, random_state=1)
        Z = model.fit_transform(views)
        assert np.array_equal(Z, model.embedding_)

    def test_transform_returns_fitted_embedding(self):
        views = _make_views(seed=2)
        model = MultiViewMDS(n_components=2, max_iter=50, random_state=2).fit(views)
        Z = model.transform(views)
        assert np.array_equal(Z, model.embedding_)

    def test_precomputed_dissimilarity_mode(self):
        views = _make_views(seed=3)
        dissim = [pairwise_distances(v) for v in views]
        model = MultiViewMDS(
            n_components=2, dissimilarity="precomputed", max_iter=50, random_state=3
        )
        model.fit(dissim)
        assert model.embedding_.shape == (40, 2)

    def test_precomputed_requires_symmetric_matrix(self):
        n = 20
        bad = np.random.default_rng(4).random((n, n))  # not symmetric
        good = pairwise_distances(np.random.default_rng(5).normal(size=(n, 3)))
        model = MultiViewMDS(n_components=2, dissimilarity="precomputed")
        with pytest.raises(ValueError, match="symmetric"):
            model.fit([bad, good])

    def test_gamma_less_than_one_raises_during_fit(self):
        views = _make_views(n_samples=20, seed=6)
        model = MultiViewMDS(n_components=2, gamma=0.5, max_iter=5, random_state=6)
        with pytest.raises(ValueError, match="gamma must be > 1.0"):
            model.fit(views)

    def test_non_positive_n_components_raises(self):
        views = _make_views(n_samples=20, seed=7)
        with pytest.raises(ValueError, match="n_components must be a positive"):
            MultiViewMDS(n_components=0).fit(views)

    def test_non_positive_max_iter_raises(self):
        views = _make_views(n_samples=20, seed=8)
        with pytest.raises(ValueError, match="max_iter must be a positive"):
            MultiViewMDS(n_components=2, max_iter=0).fit(views)

    def test_custom_init_used_when_provided(self):
        views = _make_views(n_samples=20, seed=9)
        init = np.random.default_rng(9).normal(size=(20, 2))
        model = MultiViewMDS(n_components=2, max_iter=5, init=init, random_state=9)
        model.fit(views)
        assert model.embedding_.shape == (20, 2)

    def test_init_with_wrong_shape_raises(self):
        views = _make_views(n_samples=20, seed=10)
        bad_init = np.random.default_rng(10).normal(size=(20, 3))
        model = MultiViewMDS(n_components=2, init=bad_init)
        with pytest.raises(ValueError, match="init must have shape"):
            model.fit(views)

    def test_guttman_transform_matches_pseudo_inverse(self):
        views = _make_views(n_samples=30, seed=11)
        model = MultiViewMDS(n_components=2)
        deltas = model._compute_view_dissimilarities(views)
        n = deltas[0].shape[0]
        weight_mats = [np.ones((n, n)) - np.eye(n) for _ in deltas]
        alpha = np.array([0.7, 0.3])
        Z = np.random.default_rng(11).normal(size=(n, 2))
        Z -= Z.mean(axis=0)
        B = model._compute_b_matrix(Z, deltas, weight_mats, alpha)
        V = model._compute_v_matrix(weight_mats, alpha)
        # Pseudo-inverse of the singular V, discarding its null direction
        w, U = np.linalg.eigh(V)
        keep = w > 1e-8 * w.max()
        V_pinv = (U[:, keep] / w[keep]) @ U[:, keep].T
        expected = V_pinv @ B @ Z
        expected -= expected.mean(axis=0)
        np.testing.assert_allclose(
            model._guttman_transform(B, Z, alpha), expected, rtol=1e-8, atol=1e-12
        )

    def test_objective_is_non_increasing(self):
        views = _make_views(n_samples=40, seed=12)
        model = MultiViewMDS(n_components=2, max_iter=100, tol=0.0, random_state=12)
        history = model.fit(views).objective_history_
        assert np.all(np.diff(history) <= 1e-10 * np.abs(history[:-1]))
