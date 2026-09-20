import numpy as np
import pytest

from polyview.utils.kernels import (
    KernelSpec,
    center_kernel,
    is_valid_kernel,
    normalize_kernel,
)


def _make_X(n_samples=20, n_features=5, seed=0):
    return np.random.default_rng(seed).normal(size=(n_samples, n_features))


class TestCenterKernel:
    def test_centered_kernel_has_zero_row_and_col_means(self):
        X = _make_X()
        K = X @ X.T
        Kc = center_kernel(K)
        assert np.allclose(Kc.mean(axis=0), 0.0, atol=1e-10)
        assert np.allclose(Kc.mean(axis=1), 0.0, atol=1e-10)

    def test_center_kernel_is_symmetric(self):
        rng = np.random.default_rng(1)
        K = rng.normal(size=(10, 10))
        Kc = center_kernel(K)
        assert np.allclose(Kc, Kc.T)


class TestNormalizeKernel:
    def test_diagonal_becomes_one(self):
        X = _make_X(seed=2)
        K = X @ X.T
        Kn = normalize_kernel(K)
        assert np.allclose(np.diag(Kn), 1.0)

    def test_output_is_clipped_to_valid_range(self):
        K = np.array([[1.0, 100.0], [100.0, 1.0]])
        Kn = normalize_kernel(K)
        assert np.all(Kn <= 1.0)
        assert np.all(Kn >= -1.0)


class TestIsValidKernel:
    def test_psd_kernel_is_valid(self):
        X = _make_X(seed=3)
        K = X @ X.T
        assert is_valid_kernel(K) is True

    def test_non_symmetric_is_invalid(self):
        K = np.array([[1.0, 2.0], [0.5, 1.0]])
        assert is_valid_kernel(K) is False

    def test_non_square_is_invalid(self):
        K = np.zeros((3, 4))
        assert is_valid_kernel(K) is False

    def test_indefinite_matrix_is_invalid(self):
        K = np.array([[0.0, 2.0], [2.0, 0.0]])
        assert is_valid_kernel(K) is False


class TestKernelSpec:
    def test_negative_weight_raises(self):
        with pytest.raises(ValueError, match="weight must be >= 0"):
            KernelSpec(weight=-1.0)

    def test_linear_kernel_matches_gram_matrix(self):
        X = _make_X(seed=4)
        K = KernelSpec("linear", center=False, normalize=False).build(X)
        assert np.allclose(K, X @ X.T)

    def test_rbf_kernel_default_gamma_produces_valid_kernel(self):
        X = _make_X(seed=5)
        K = KernelSpec("rbf", center=False, normalize=False).build(X)
        assert is_valid_kernel(K)
        assert np.allclose(np.diag(K), 1.0)  # RBF(x, x) = 1

    def test_polynomial_kernel_runs(self):
        X = _make_X(seed=6)
        K = KernelSpec("polynomial", center=False, normalize=False, degree=2).build(X)
        assert K.shape == (20, 20)

    def test_precomputed_requires_square_matrix(self):
        X = _make_X(n_samples=10, n_features=5, seed=7)
        with pytest.raises(ValueError, match="square"):
            KernelSpec("precomputed").build(X)

    def test_precomputed_passes_through(self):
        rng = np.random.default_rng(8)
        A = rng.normal(size=(6, 6))
        K = KernelSpec("precomputed", center=False, normalize=False).build(A)
        assert np.allclose(K, A)

    def test_custom_callable_kernel(self):
        X = _make_X(n_samples=8, seed=9)

        def my_kernel(X):
            return X @ X.T

        K = KernelSpec(my_kernel, center=False, normalize=False).build(X)
        assert np.allclose(K, X @ X.T)

    def test_custom_callable_wrong_shape_raises(self):
        X = _make_X(n_samples=8, seed=10)

        def bad_kernel(X):
            return np.zeros((3, 3))

        with pytest.raises(ValueError, match="Custom kernel must return"):
            KernelSpec(bad_kernel).build(X)

    def test_unknown_kernel_name_raises(self):
        with pytest.raises(ValueError, match="Unknown kernel"):
            KernelSpec("bogus").build(_make_X(seed=11))

    def test_center_and_normalize_applied_by_default(self):
        X = _make_X(seed=12)
        K = KernelSpec("linear").build(X)
        assert np.allclose(np.diag(K), 1.0, atol=1e-8)
