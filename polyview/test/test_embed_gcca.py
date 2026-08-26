import numpy as np
import pytest

from polyview.embed.gcca import GCCA


def _make_correlated_views(n_samples=100, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n_samples, 2))
    x1 = z @ rng.normal(size=(2, 6)) + 0.01 * rng.normal(size=(n_samples, 6))
    x2 = z @ rng.normal(size=(2, 5)) + 0.01 * rng.normal(size=(n_samples, 5))
    x3 = z @ rng.normal(size=(2, 4)) + 0.01 * rng.normal(size=(n_samples, 4))
    return [x1, x2, x3]


class TestGCCA:
    def test_shared_output_shape(self):
        views = _make_correlated_views()
        gcca = GCCA(n_components=2, output="shared")
        Z = gcca.fit_transform(views)
        assert Z.shape == (100, 2)

    def test_concat_output_shape(self):
        views = _make_correlated_views(seed=1)
        gcca = GCCA(n_components=2, output="concat")
        Z = gcca.fit_transform(views)
        assert Z.shape == (100, 2 * 3)

    def test_mean_output_shape(self):
        views = _make_correlated_views(seed=2)
        gcca = GCCA(n_components=2, output="mean")
        Z = gcca.fit_transform(views)
        assert Z.shape == (100, 2)

    def test_list_output_returns_per_view_projections(self):
        views = _make_correlated_views(seed=3)
        gcca = GCCA(n_components=2, output="list")
        Zs = gcca.fit_transform(views)
        assert isinstance(Zs, list)
        assert len(Zs) == 3
        for Z in Zs:
            assert Z.shape == (100, 2)

    def test_recovers_strong_shared_structure(self):
        views = _make_correlated_views(seed=4)
        gcca = GCCA(n_components=2, output="shared")
        gcca.fit(views)
        corrs = gcca.canonical_correlations()
        # With near-noiseless shared latent structure, the leading canonical
        # correlations between any pair of views should be close to 1.
        assert corrs.shape == (3, 3, 2)
        assert corrs[0, 1, 0] > 0.95
        assert corrs[0, 2, 0] > 0.95

    def test_transform_on_new_data_uses_fitted_weights(self):
        # output="shared" always returns the training G_ (non-parametric),
        # so out-of-sample projection needs "concat" or "mean" instead.
        views = _make_correlated_views(n_samples=120, seed=5)
        train = [v[:100] for v in views]
        test = [v[100:] for v in views]

        gcca = GCCA(n_components=2, output="mean").fit(train)
        Z_test = gcca.transform(test)
        assert Z_test.shape == (20, 2)

    def test_shared_output_ignores_new_data_and_returns_training_embedding(self):
        views = _make_correlated_views(n_samples=120, seed=5)
        train = [v[:100] for v in views]
        test = [v[100:] for v in views]

        gcca = GCCA(n_components=2, output="shared").fit(train)
        Z_test = gcca.transform(test)
        assert Z_test.shape == (100, 2)
        assert np.array_equal(Z_test, gcca.G_)

    def test_regularisation_list_length_mismatch_raises(self):
        views = _make_correlated_views(seed=6)
        gcca = GCCA(n_components=2, regularisation=[1e-4, 1e-4])
        with pytest.raises(ValueError, match="regularisation list has"):
            gcca.fit(views)

    def test_invalid_output_mode_raises(self):
        views = _make_correlated_views(seed=7)
        gcca = GCCA(n_components=2, output="bogus")
        with pytest.raises(ValueError, match="output must be"):
            gcca.fit_transform(views)

    def test_transform_before_fit_raises(self):
        gcca = GCCA(n_components=2)
        with pytest.raises(Exception):
            gcca.transform(_make_correlated_views(seed=8))
