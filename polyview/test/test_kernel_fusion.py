import numpy as np
import pytest

from polyview.fusion.kernel_fusion import KernelFusion, KernelSpec


def test_kernel_fusion_product_mode_matches_weighted_geometric_product():
    k1 = np.array([[1.0, 0.5], [0.5, 1.0]])
    k2 = np.array([[1.0, 0.25], [0.25, 1.0]])

    specs = [
        KernelSpec("precomputed", weight=2.0, center=False, normalize=False),
        KernelSpec("precomputed", weight=1.0, center=False, normalize=False),
    ]
    fusion = KernelFusion(specs=specs, fusion_mode="product")

    k_fused = fusion.fit_transform([k1, k2])
    expected = (k1**2.0) * (k2**1.0)

    assert np.allclose(k_fused, expected)
    assert np.allclose(k_fused, k_fused.T)


def test_kernel_fusion_product_mode_rejects_negative_entries():
    k1 = np.array([[1.0, -0.2], [-0.2, 1.0]])
    k2 = np.array([[1.0, 0.3], [0.3, 1.0]])

    specs = [
        KernelSpec("precomputed", center=False, normalize=False),
        KernelSpec("precomputed", center=False, normalize=False),
    ]
    fusion = KernelFusion(specs=specs, fusion_mode="product")

    with pytest.raises(ValueError, match="non-negative kernel entries"):
        fusion.fit([k1, k2])


def test_kernel_fusion_product_mode_transform_uses_fitted_weights():
    k1_train = np.array([[1.0, 0.6], [0.6, 1.0]])
    k2_train = np.array([[1.0, 0.4], [0.4, 1.0]])

    specs = [
        KernelSpec("precomputed", weight=3.0, center=False, normalize=False),
        KernelSpec("precomputed", weight=1.0, center=False, normalize=False),
    ]
    fusion = KernelFusion(specs=specs, normalize_weights=True, fusion_mode="product")
    fusion.fit([k1_train, k2_train])

    k1_test = np.array([[1.0, 0.2], [0.2, 1.0]])
    k2_test = np.array([[1.0, 0.8], [0.8, 1.0]])
    transformed = fusion.transform([k1_test, k2_test])

    expected = (k1_test**0.75) * (k2_test**0.25)
    assert np.allclose(transformed, expected)


def _views(n_samples=12, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.normal(size=(n_samples, 3)), rng.normal(size=(n_samples, 4))]


def test_kernel_fusion_default_and_broadcast_specs():
    views = _views()
    assert KernelFusion().fit_transform(views).shape == (12, 12)

    fusion = KernelFusion(specs=KernelSpec("linear", weight=2.0)).fit(views)
    assert len(fusion.specs_) == 2
    assert all(spec.kernel == "linear" for spec in fusion.specs_)
    assert np.allclose(fusion.weights_, [2.0, 2.0])


def test_kernel_fusion_sum_mode_with_normalized_weights():
    views = _views(seed=1)
    specs = [KernelSpec("linear", weight=3.0), KernelSpec("linear", weight=1.0)]
    fusion = KernelFusion(specs=specs, normalize_weights=True).fit(views)

    k1, k2 = fusion.kernels_
    assert np.allclose(fusion.K_fused_, 0.75 * k1 + 0.25 * k2)
    assert np.allclose(fusion.kernel_matrix(), fusion.K_fused_)
    fractions = [c["contribution_fraction"] for c in fusion.view_contributions()]
    assert np.isclose(sum(fractions), 1.0)


def test_kernel_fusion_product_mode_skips_zero_weight_views():
    k1 = np.array([[1.0, 0.5], [0.5, 1.0]])
    k2 = np.array([[1.0, -0.9], [-0.9, 1.0]])  # would be rejected if used
    specs = [
        KernelSpec("precomputed", weight=1.0, center=False, normalize=False),
        KernelSpec("precomputed", weight=0.0, center=False, normalize=False),
    ]
    fused = KernelFusion(specs=specs, fusion_mode="product").fit_transform([k1, k2])
    assert np.allclose(fused, k1)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(specs=[KernelSpec("linear")]), "Expected 2 KernelSpecs"),
        (
            dict(specs=[KernelSpec("linear", weight=0.0)] * 2, normalize_weights=True),
            "All weights are zero",
        ),
        (dict(fusion_mode="product", product_eps=0.0), "product_eps"),
        (dict(fusion_mode="bogus"), "fusion_mode"),
    ],
)
def test_kernel_fusion_invalid_configurations_raise(kwargs, match):
    with pytest.raises(ValueError, match=match):
        KernelFusion(**kwargs).fit(_views(seed=2))


def test_kernel_fusion_transform_validates_views():
    views = _views(seed=3)
    fusion = KernelFusion(specs=KernelSpec("linear")).fit(views)

    assert fusion.transform(views).shape == (12, 12)
    with pytest.raises(ValueError, match="Fitted on 2 views"):
        fusion.transform(views[:1])
    with pytest.raises(ValueError, match="features, expected"):
        fusion.transform([views[0][:, :2], views[1]])
