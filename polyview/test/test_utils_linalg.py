import numpy as np
import pytest
from scipy.sparse.linalg import aslinearoperator

from polyview.utils.linalg import (
    resolve_smoother_solver,
    smoother_sum_operator,
    truncated_eigh,
)


def _make_symmetric(n=60, seed=0):
    A = np.random.default_rng(seed).normal(size=(n, n))
    return A + A.T


class TestTruncatedEigh:
    @pytest.mark.parametrize("solver", ["dense", "arpack"])
    @pytest.mark.parametrize("largest", [True, False])
    def test_matches_full_eigendecomposition(self, solver, largest):
        A = _make_symmetric()
        k = 4
        full_vals, full_vecs = np.linalg.eigh(A)
        if largest:
            full_vals, full_vecs = full_vals[::-1], full_vecs[:, ::-1]

        vals, vecs = truncated_eigh(
            A, k, largest=largest, eigen_solver=solver, random_state=0
        )

        assert vals.shape == (k,)
        assert vecs.shape == (60, k)
        assert np.allclose(vals, full_vals[:k])
        # Eigenvectors agree up to sign
        assert np.allclose(np.abs(np.sum(vecs * full_vecs[:, :k], axis=0)), 1.0)

    @pytest.mark.parametrize("solver", ["dense", "arpack"])
    def test_accepts_linear_operator(self, solver):
        A = _make_symmetric(seed=1)
        vals_dense, _ = truncated_eigh(A, 3, eigen_solver="dense")
        vals_op, _ = truncated_eigh(
            aslinearoperator(A), 3, eigen_solver=solver, random_state=0
        )
        assert np.allclose(vals_op, vals_dense)

    def test_solvers_return_identical_signs(self):
        A = _make_symmetric(seed=2)
        _, vecs_dense = truncated_eigh(A, 5, eigen_solver="dense")
        _, vecs_arpack = truncated_eigh(A, 5, eigen_solver="arpack", random_state=0)
        assert np.allclose(vecs_dense, vecs_arpack, atol=1e-8)

    def test_invalid_solver_raises(self):
        with pytest.raises(ValueError, match="eigen_solver must be"):
            truncated_eigh(_make_symmetric(), 2, eigen_solver="bogus")

    def test_arpack_requires_k_smaller_than_n(self):
        with pytest.raises(ValueError, match="requires k < n"):
            truncated_eigh(_make_symmetric(n=10), 10, eigen_solver="arpack")

    def test_k_out_of_range_raises(self):
        with pytest.raises(ValueError, match="k must be in"):
            truncated_eigh(_make_symmetric(n=10), 0)


class TestResolveSmootherSolver:
    @pytest.mark.parametrize(
        "n, total_features, expected",
        [
            (1000, 100, "arpack"),  # low-rank operator: Lanczos
            (1000, 3000, "dense"),  # smoothers close to identity: LAPACK
            (1000, 1000, "dense"),
            (150, 10, "dense"),  # small problem
        ],
    )
    def test_auto(self, n, total_features, expected):
        assert resolve_smoother_solver("auto", n, total_features, 5) == expected

    @pytest.mark.parametrize("solver", ["dense", "arpack"])
    def test_explicit_solver_is_kept(self, solver):
        assert resolve_smoother_solver(solver, 1000, 3000, 5) == solver

    def test_invalid_solver_raises(self):
        with pytest.raises(ValueError, match="eigen_solver must be"):
            resolve_smoother_solver("bogus", 1000, 100, 5)


class TestSmootherSumOperator:
    def test_matches_explicit_smoother_sum(self):
        rng = np.random.default_rng(3)
        views = [rng.normal(size=(30, 4)), rng.normal(size=(30, 7))]
        regs = [1e-2, 1e-3]
        scale = 30.0

        expected = sum(
            X @ np.linalg.solve(X.T @ X / scale + r * np.eye(X.shape[1]), X.T / scale)
            for X, r in zip(views, regs)
        )
        op, factors = smoother_sum_operator(views, regs, scale=scale)

        assert len(factors) == 2
        assert np.allclose(op @ np.eye(30), expected)
        u = rng.normal(size=30)
        assert np.allclose(op @ u, expected @ u)
