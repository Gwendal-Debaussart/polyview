import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score, pairwise_kernels

from polyview.cluster._spectral import normalized_similarity, top_eigenvectors
from polyview.cluster.mv_coreg_sc import MultiViewCoRegSpectralClustering
from polyview.cluster.mv_cotrain_sc import MultiViewCoTrainSpectralClustering


def _make_separable_views(n_samples=60, k=3, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, k, size=n_samples)

    centers1 = rng.normal(scale=10.0, size=(k, 4))
    centers2 = rng.normal(scale=10.0, size=(k, 3))

    x1 = centers1[y] + rng.normal(scale=0.3, size=(n_samples, 4))
    x2 = centers2[y] + rng.normal(scale=0.3, size=(n_samples, 3))
    return [x1, x2], y


def _make_overlapping_views(n_samples=150, k=3, spread=3.0, seed=1):
    """Overlapping Gaussian blobs whose rbf affinity graphs are connected."""
    rng = np.random.default_rng(seed)
    y = np.repeat(np.arange(k), n_samples // k)
    views = []
    for n_features in (4, 3):
        centers = rng.normal(scale=spread, size=(k, n_features))
        views.append(centers[y] + rng.normal(size=(len(y), n_features)))
    return views, y


def _n_connected_components(X):
    """Number of zero eigenvalues of the unnormalised Laplacian of the rbf graph."""
    A = pairwise_kernels(X, metric="rbf")
    eigvals = np.linalg.eigvalsh(np.diag(A.sum(axis=1)) - A)
    return int(np.sum(eigvals < 1e-8 * eigvals.max()))


def _relative_row_spread(U):
    """||U - mean row|| / ||U||: close to 0 when all samples share one embedding."""
    return np.linalg.norm(U - U.mean(axis=0)) / np.linalg.norm(U)


def test_overlapping_views_have_connected_graphs():
    views, _ = _make_overlapping_views()
    assert [_n_connected_components(X) for X in views] == [1, 1]


class TestMultiViewCoRegSpectralClustering:
    def test_fit_predict_recovers_clusters(self):
        views, y = _make_separable_views(seed=0)
        model = MultiViewCoRegSpectralClustering(
            n_clusters=3, max_iter=10, random_state=0
        )
        labels = model.fit_predict(views)

        assert labels.shape == (60,)
        assert adjusted_rand_score(y, labels) > 0.7

    def test_fitted_attribute_shapes(self):
        views, _ = _make_separable_views(n_samples=40, seed=1)
        model = MultiViewCoRegSpectralClustering(
            n_clusters=3, max_iter=5, random_state=1
        ).fit(views)

        assert model.embedding_.shape == (40, 3 * 2)
        assert isinstance(model.objective_, list)
        assert len(model.objective_) == model.n_iter_ <= 5
        assert model.labels_.shape == (40,)

    def test_v_lambda_zero_still_produces_valid_clustering(self):
        views, _ = _make_separable_views(n_samples=40, seed=2)
        model = MultiViewCoRegSpectralClustering(
            n_clusters=3, max_iter=5, v_lambda=0.0, random_state=2
        ).fit(views)
        assert model.labels_.shape == (40,)
        assert set(np.unique(model.labels_)).issubset({0, 1, 2})

    def test_fit_sets_standard_multiview_attributes(self):
        views, _ = _make_separable_views(n_samples=30, seed=5)
        model = MultiViewCoRegSpectralClustering(
            n_clusters=3, max_iter=3, random_state=5
        ).fit(views)
        assert model.n_views_in_ == 2
        assert model.n_samples_ == 30
        assert model.n_features_in_ == [4, 3]

    def test_mismatched_sample_counts_raise_clear_error(self):
        x1 = np.random.rand(20, 4)
        x2 = np.random.rand(19, 3)
        with pytest.raises(ValueError, match="same number of samples"):
            MultiViewCoRegSpectralClustering(n_clusters=2, max_iter=2).fit([x1, x2])

    def test_arpack_and_dense_solvers_agree(self):
        views, _ = _make_separable_views(n_samples=80, seed=7)
        params = dict(n_clusters=3, max_iter=5, random_state=7, tol=0.0)
        dense = MultiViewCoRegSpectralClustering(eigen_solver="dense", **params)
        arpack = MultiViewCoRegSpectralClustering(eigen_solver="arpack", **params)
        dense.fit(views)
        arpack.fit(views)
        assert np.allclose(dense.objective_, arpack.objective_)
        assert adjusted_rand_score(dense.labels_, arpack.labels_) == 1.0

    def test_objective_matches_explicit_form(self):
        model = MultiViewCoRegSpectralClustering(n_clusters=3, max_iter=1, v_lambda=0.5)
        rng = np.random.default_rng(0)
        laplacians = [np.eye(30), 2 * np.eye(30), 3 * np.eye(30)]
        embeddings = [np.linalg.qr(rng.normal(size=(30, 3)))[0] for _ in range(3)]
        expected = sum(np.trace(U.T @ L @ U) for U, L in zip(embeddings, laplacians))
        for v in range(3):
            for w in range(v + 1, 3):
                Uv, Uw = embeddings[v], embeddings[w]
                expected -= 0.5 * np.trace(Uv @ Uv.T @ Uw @ Uw.T)
        assert np.isclose(model._objective(laplacians, embeddings), expected)

    def test_objective_is_non_increasing(self):
        rng = np.random.default_rng(11)
        views = [
            rng.normal(size=(80, 4)),
            rng.normal(size=(80, 3)),
            rng.normal(size=(80, 5)),
        ]
        model = MultiViewCoRegSpectralClustering(
            n_clusters=3, max_iter=10, tol=0.0, random_state=11
        ).fit(views)
        obj = np.asarray(model.objective_)
        assert np.all(np.diff(obj) <= 1e-8 * abs(obj[0]))

    def test_update_rewards_agreement_with_other_views(self):
        rng = np.random.default_rng(12)
        A = rng.random((40, 40))
        A = (A + A.T) / 2
        laplacian = np.diag(A.sum(axis=1)) - A
        others = [np.linalg.qr(rng.normal(size=(40, 3)))[0]]

        def agreement(v_lambda):
            model = MultiViewCoRegSpectralClustering(n_clusters=3, v_lambda=v_lambda)
            U = model._update_single_embedding(laplacian, others, others[0], "dense")
            return np.linalg.norm(U.T @ others[0]) ** 2

        assert agreement(20.0) > agreement(0.0)

    def test_stops_when_objective_converges(self):
        views, _ = _make_separable_views(seed=9)
        converged = MultiViewCoRegSpectralClustering(
            n_clusters=3, max_iter=50, random_state=9
        ).fit(views)
        assert converged.n_iter_ < 50
        assert len(converged.objective_) == converged.n_iter_

        full = MultiViewCoRegSpectralClustering(
            n_clusters=3, max_iter=6, tol=0.0, random_state=9
        ).fit(views)
        assert full.n_iter_ == 6

    @pytest.mark.parametrize(
        "eigen_solver, n_samples, expected",
        [("auto", 300, "arpack"), ("auto", 100, "dense"), ("dense", 300, "dense")],
    )
    def test_eigen_solver_resolution(self, eigen_solver, n_samples, expected):
        model = MultiViewCoRegSpectralClustering(
            n_clusters=10, eigen_solver=eigen_solver
        )
        assert model._resolve_eigen_solver(n_samples) == expected

    def test_regularized_laplacian_operator_matches_dense_matrix(self):
        rng = np.random.default_rng(10)
        A = rng.random((25, 25))
        A = (A + A.T) / 2
        laplacian = np.diag(A.sum(axis=1)) - A
        others = [np.linalg.qr(rng.normal(size=(25, 3)))[0] for _ in range(2)]
        model = MultiViewCoRegSpectralClustering(n_clusters=3, v_lambda=0.5)

        op = model._regularized_laplacian(laplacian, others)
        expected = laplacian - 0.5 * sum(U @ U.T for U in others)
        assert np.allclose(op @ np.eye(25), expected)
        u = rng.normal(size=25)
        assert np.allclose(op @ u, expected @ u)

    def test_normalized_and_unnormalized_laplacians(self):
        rng = np.random.default_rng(13)
        A = rng.random((20, 20))
        A = (A + A.T) / 2

        normalized = MultiViewCoRegSpectralClustering(
            laplacian="normalized"
        )._graph_laplacian(A)
        B = A.copy()
        np.fill_diagonal(B, 0.0)
        d = B.sum(axis=1)
        assert np.allclose(normalized, np.eye(20) - B / np.sqrt(np.outer(d, d)))
        vals = np.linalg.eigvalsh(normalized)
        assert vals.min() > -1e-10 and vals.max() < 2 + 1e-10

        unnormalized = MultiViewCoRegSpectralClustering(
            laplacian="unnormalized"
        )._graph_laplacian(A)
        assert np.allclose(unnormalized, np.diag(A.sum(axis=1)) - A)

    def test_invalid_laplacian_raises(self):
        views, _ = _make_separable_views(n_samples=20, seed=14)
        with pytest.raises(ValueError, match="laplacian must be"):
            MultiViewCoRegSpectralClustering(n_clusters=2, laplacian="bogus").fit(views)

    @pytest.mark.parametrize("laplacian", ["normalized", "unnormalized"])
    def test_both_laplacians_recover_clusters(self, laplacian):
        views, y = _make_separable_views(seed=15)
        labels = MultiViewCoRegSpectralClustering(
            n_clusters=3, laplacian=laplacian, random_state=15
        ).fit_predict(views)
        assert adjusted_rand_score(y, labels) > 0.7

    def test_recovers_clusters_on_connected_graphs(self):
        views, y = _make_overlapping_views()
        model = MultiViewCoRegSpectralClustering(n_clusters=3, random_state=0).fit(
            views
        )
        assert adjusted_rand_score(y, model.labels_) > 0.8

    def test_negative_v_lambda_raises(self):
        views, _ = _make_separable_views(n_samples=30)
        with pytest.raises(ValueError, match="v_lambda"):
            MultiViewCoRegSpectralClustering(n_clusters=3, v_lambda=-1.0).fit(views)

    def test_view_embeddings_are_orthonormal(self):
        views, _ = _make_overlapping_views(n_samples=60)
        model = MultiViewCoRegSpectralClustering(n_clusters=3, random_state=0).fit(
            views
        )
        assert len(model.view_embeddings_) == 2
        for U in model.view_embeddings_:
            assert U.shape == (60, 3)
            np.testing.assert_allclose(U.T @ U, np.eye(3), atol=1e-8)

    def test_median_gamma(self):
        views, y = _make_overlapping_views()
        model = MultiViewCoRegSpectralClustering(
            n_clusters=3, gamma="median", random_state=0
        ).fit(views)
        assert adjusted_rand_score(y, model.labels_) > 0.8


class TestMultiViewCoTrainSpectralClustering:
    def test_fit_predict_recovers_clusters(self):
        views, y = _make_separable_views(seed=3)
        model = MultiViewCoTrainSpectralClustering(
            n_clusters=3, max_iter=10, random_state=3
        )
        labels = model.fit_predict(views)

        assert labels.shape == (60,)
        assert adjusted_rand_score(y, labels) > 0.7

    def test_fitted_attribute_shapes(self):
        views, _ = _make_separable_views(n_samples=40, seed=4)
        model = MultiViewCoTrainSpectralClustering(
            n_clusters=3, max_iter=5, random_state=4
        ).fit(views)

        assert model.embedding_.shape == (40, 3 * 2)
        assert isinstance(model.objective_, list)
        assert len(model.objective_) == 5
        assert all(0.0 <= a <= 1.0 + 1e-9 for a in model.objective_)
        assert model.labels_.shape == (40,)

    def test_fit_sets_standard_multiview_attributes(self):
        views, _ = _make_separable_views(n_samples=30, seed=6)
        model = MultiViewCoTrainSpectralClustering(
            n_clusters=3, max_iter=3, random_state=6
        ).fit(views)
        assert model.n_views_in_ == 2
        assert model.n_samples_ == 30
        assert model.n_features_in_ == [4, 3]

    def test_mismatched_sample_counts_raise_clear_error(self):
        x1 = np.random.rand(20, 4)
        x2 = np.random.rand(19, 3)
        with pytest.raises(ValueError, match="same number of samples"):
            MultiViewCoTrainSpectralClustering(n_clusters=2, max_iter=2).fit([x1, x2])

    def test_recovers_clusters_on_connected_graphs(self):
        views, y = _make_overlapping_views()
        model = MultiViewCoTrainSpectralClustering(
            n_clusters=3, max_iter=50, random_state=0
        ).fit(views)
        assert adjusted_rand_score(y, model.labels_) > 0.8

    @pytest.mark.parametrize("max_iter", [1, 10, 50])
    def test_embedding_does_not_collapse_with_iterations(self, max_iter):
        # Regression: a graph low-pass filter iterated on a connected graph drives
        # every row of the embedding to the same vector.
        views, _ = _make_overlapping_views()
        model = MultiViewCoTrainSpectralClustering(
            n_clusters=3, max_iter=max_iter, random_state=0
        ).fit(views)
        assert _relative_row_spread(model.embedding_) > 0.5

    def test_update_matches_dense_reference(self):
        views, _ = _make_overlapping_views(n_samples=60, spread=2.0)
        K = pairwise_kernels(views[0], metric="rbf")
        U_other = top_eigenvectors(
            normalized_similarity(pairwise_kernels(views[1], metric="rbf")),
            3,
            np.random.RandomState(0),
        )
        U = MultiViewCoTrainSpectralClustering._cotrain_embedding(
            K, U_other, 3, np.random.RandomState(0)
        )

        S = U_other @ U_other.T @ K
        expected = np.linalg.eigh(normalized_similarity((S + S.T) / 2))[1][:, -3:]
        np.testing.assert_allclose(U @ U.T, expected @ expected.T, atol=1e-8)

    def test_agreement_is_reported_per_iteration(self):
        views, _ = _make_overlapping_views()
        model = MultiViewCoTrainSpectralClustering(
            n_clusters=3, max_iter=4, random_state=0
        ).fit(views)
        assert len(model.objective_) == 4
        assert all(0.0 <= a <= 1.0 + 1e-9 for a in model.objective_)

    def test_median_gamma(self):
        views, y = _make_overlapping_views()
        model = MultiViewCoTrainSpectralClustering(
            n_clusters=3, max_iter=10, gamma="median", random_state=0
        ).fit(views)
        assert adjusted_rand_score(y, model.labels_) > 0.8

    def test_lambda_reg_is_deprecated(self):
        views, _ = _make_separable_views(n_samples=30)
        with pytest.warns(FutureWarning, match="lambda_reg"):
            MultiViewCoTrainSpectralClustering(
                n_clusters=3, max_iter=2, lambda_reg=1.0
            ).fit(views)

    def test_single_view_raises(self):
        views, _ = _make_separable_views(n_samples=30)
        with pytest.raises(ValueError, match="at least 2 views"):
            MultiViewCoTrainSpectralClustering(n_clusters=3).fit(views[:1])
