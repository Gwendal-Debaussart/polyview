import numpy as np
import pytest

from polyview.embed.mcca import MCCA


def _make_correlated_views(n_samples=100, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n_samples, 2))
    x1 = z @ rng.normal(size=(2, 6)) + 0.01 * rng.normal(size=(n_samples, 6))
    x2 = z @ rng.normal(size=(2, 5)) + 0.01 * rng.normal(size=(n_samples, 5))
    return [x1, x2]


class TestMCCA:
    @pytest.mark.parametrize("objective", ["sumcor", "maxvar"])
    def test_concat_output_shape(self, objective):
        views = _make_correlated_views(seed=0)
        mcca = MCCA(n_components=2, objective=objective, output="concat")
        Z = mcca.fit_transform(views)
        assert Z.shape == (100, 4)

    def test_mean_output_shape(self):
        views = _make_correlated_views(seed=1)
        mcca = MCCA(n_components=2, output="mean")
        Z = mcca.fit_transform(views)
        assert Z.shape == (100, 2)

    def test_list_output_shape(self):
        views = _make_correlated_views(seed=2)
        mcca = MCCA(n_components=2, output="list")
        Zs = mcca.fit_transform(views)
        assert isinstance(Zs, list)
        assert len(Zs) == 2
        assert all(Z.shape == (100, 2) for Z in Zs)

    def test_default_n_components_uses_smallest_view_dim(self):
        views = _make_correlated_views(seed=3)  # dims 6, 5
        mcca = MCCA(output="mean").fit(views)
        assert mcca.n_components_ == 5

    def test_n_components_exceeding_smallest_dim_raises(self):
        views = _make_correlated_views(seed=4)  # smallest dim is 5
        with pytest.raises(ValueError, match="exceeds smallest view dimension"):
            MCCA(n_components=10).fit(views)

    def test_recovers_strong_shared_structure(self):
        views = _make_correlated_views(seed=5)
        mcca = MCCA(n_components=2, objective="maxvar").fit(views)
        corrs = mcca.canonical_correlations()
        assert corrs.shape == (2, 2, 2)
        assert corrs[0, 1, 0] > 0.9

    def test_invalid_objective_raises(self):
        views = _make_correlated_views(seed=6)
        with pytest.raises(ValueError, match="objective must be"):
            MCCA(n_components=2, objective="bogus").fit(views)

    def test_transform_on_new_data(self):
        views = _make_correlated_views(n_samples=120, seed=7)
        train = [v[:100] for v in views]
        test = [v[100:] for v in views]

        mcca = MCCA(n_components=2, output="mean").fit(train)
        Z_test = mcca.transform(test)
        assert Z_test.shape == (20, 2)

    def test_transform_before_fit_raises(self):
        mcca = MCCA(n_components=2)
        with pytest.raises(Exception):
            mcca.transform(_make_correlated_views(seed=8))
