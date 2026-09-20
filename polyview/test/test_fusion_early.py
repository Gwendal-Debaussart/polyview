import numpy as np
import pytest

from polyview.fusion.early import ConcatFusion, NormalizedFusion, WeightedFusion


def _make_views(n_samples=30, seed=0):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(loc=10.0, scale=5.0, size=(n_samples, 3))
    x2 = rng.normal(loc=-3.0, scale=1.0, size=(n_samples, 4))
    return [x1, x2]


class TestConcatFusion:
    def test_transform_shape_and_content(self):
        x1, x2 = _make_views()
        fused = ConcatFusion().fit_transform([x1, x2])
        assert fused.shape == (30, 7)
        assert np.allclose(fused, np.concatenate([x1, x2], axis=1))

    def test_n_features_out_recorded(self):
        x1, x2 = _make_views()
        fusion = ConcatFusion().fit([x1, x2])
        assert fusion.n_features_out_ == 7
        assert fusion.n_views_in_ == 2

    def test_transform_before_fit_raises(self):
        with pytest.raises(Exception):
            ConcatFusion().transform(_make_views())

    def test_rejects_mismatched_sample_counts(self):
        x1 = np.random.rand(10, 3)
        x2 = np.random.rand(11, 3)
        with pytest.raises(ValueError, match="same number of samples"):
            ConcatFusion().fit([x1, x2])


class TestWeightedFusion:
    def test_default_weights_equivalent_to_concat(self):
        x1, x2 = _make_views(seed=1)
        fused = WeightedFusion().fit_transform([x1, x2])
        expected = np.concatenate([x1, x2], axis=1)
        assert np.allclose(fused, expected)

    def test_explicit_weights_scale_each_view(self):
        x1, x2 = _make_views(seed=2)
        fused = WeightedFusion(weights=[2.0, 0.5]).fit_transform([x1, x2])
        expected = np.concatenate([x1 * 2.0, x2 * 0.5], axis=1)
        assert np.allclose(fused, expected)

    def test_wrong_number_of_weights_raises(self):
        x1, x2 = _make_views(seed=3)
        with pytest.raises(ValueError, match="one entry per view"):
            WeightedFusion(weights=[1.0, 2.0, 3.0]).fit([x1, x2])


class TestNormalizedFusion:
    def test_output_is_zero_mean_unit_variance_per_view(self):
        x1, x2 = _make_views(seed=4)
        fusion = NormalizedFusion()
        fused = fusion.fit_transform([x1, x2])

        assert fused.shape == (30, 7)
        assert np.allclose(fused.mean(axis=0), 0.0, atol=1e-10)
        assert np.allclose(fused.std(axis=0), 1.0, atol=1e-8)

    def test_weights_applied_after_normalization(self):
        x1, x2 = _make_views(seed=5)
        fusion = NormalizedFusion(weights=[3.0, 1.0])
        fused = fusion.fit_transform([x1, x2])

        normed1 = (x1 - x1.mean(axis=0)) / np.clip(x1.std(axis=0), 1e-8, None)
        normed2 = (x2 - x2.mean(axis=0)) / np.clip(x2.std(axis=0), 1e-8, None)
        expected = np.concatenate([normed1 * 3.0, normed2 * 1.0], axis=1)
        assert np.allclose(fused, expected)

    def test_constant_feature_does_not_divide_by_zero(self):
        rng = np.random.default_rng(6)
        x1 = np.concatenate([np.full((20, 1), 5.0), rng.normal(size=(20, 2))], axis=1)
        x2 = rng.normal(size=(20, 3))
        fused = NormalizedFusion().fit_transform([x1, x2])
        assert np.all(np.isfinite(fused))
        # Constant column becomes 0 after (x - mean) / eps-clamped std.
        assert np.allclose(fused[:, 0], 0.0)
