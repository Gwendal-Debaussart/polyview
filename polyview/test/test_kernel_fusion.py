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

    expected = (k1_test ** 0.75) * (k2_test ** 0.25)
    assert np.allclose(transformed, expected)
