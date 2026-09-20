import numpy as np
import pytest
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression

import polyview as pv
from polyview.dataset.multiviewdataset import MultiViewDataset
from polyview.imputation import (
    CrossViewRegressionImputer,
    DropIncompleteSamples,
    KNNViewImputer,
    SimpleViewImputer,
    complete_case_mask,
    drop_missing_views,
    mark_missing_views,
    missing_entry_mask,
    missing_rate,
    missing_view_mask,
    simulate_missing_views,
)

IMPUTERS = [SimpleViewImputer, KNNViewImputer, CrossViewRegressionImputer]


def _correlated_views(n_samples=120, seed=0):
    """Three views generated from one shared latent factor, plus noise."""
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=(n_samples, 4))
    return [
        latent @ rng.normal(size=(4, d)) + 0.1 * rng.normal(size=(n_samples, d))
        for d in (5, 3, 6)
    ]


class TestMasks:
    def test_missing_entry_mask_is_cell_level(self):
        X = np.array([[1.0, np.nan], [3.0, 4.0]])
        mask = missing_entry_mask([X])[0]
        assert mask.tolist() == [[False, True], [False, False]]

    def test_missing_view_mask_all_requires_whole_row(self):
        X1 = np.array([[1.0, np.nan], [np.nan, np.nan]])
        X2 = np.array([[3.0, 4.0], [5.0, 6.0]])
        assert missing_view_mask([X1, X2]).tolist() == [[False, False], [True, False]]

    def test_missing_view_mask_any_flags_partial_rows(self):
        X1 = np.array([[1.0, np.nan], [np.nan, np.nan]])
        X2 = np.array([[3.0, 4.0], [5.0, 6.0]])
        assert missing_view_mask([X1, X2], how="any").tolist() == [
            [True, False],
            [True, False],
        ]

    def test_complete_case_and_missing_rate(self):
        X1 = np.array([[1.0], [np.nan], [3.0], [np.nan]])
        X2 = np.array([[1.0], [2.0], [3.0], [4.0]])
        assert complete_case_mask([X1, X2]).tolist() == [True, False, True, False]
        assert missing_rate([X1, X2]).tolist() == [0.5, 0.0]

    def test_mark_missing_views_is_inverse_of_missing_view_mask(self):
        views = _correlated_views(20)
        rng = np.random.default_rng(3)
        mask = rng.random((20, 3)) < 0.3
        marked = mark_missing_views(views, mask)
        assert np.array_equal(missing_view_mask(marked), mask)

    def test_mark_missing_views_does_not_mutate_input(self):
        views = _correlated_views(10)
        before = [v.copy() for v in views]
        mark_missing_views(views, np.ones((10, 3), dtype=bool))
        for original, untouched in zip(views, before):
            assert np.array_equal(original, untouched)

    def test_mark_missing_views_rejects_wrong_shape(self):
        with pytest.raises(ValueError, match="mask has shape"):
            mark_missing_views(_correlated_views(10), np.zeros((10, 2), dtype=bool))

    def test_accepts_multiview_dataset(self):
        mvd = MultiViewDataset(_correlated_views(15))
        assert missing_view_mask(mvd).shape == (15, 3)

    @pytest.mark.parametrize("bad", [np.zeros((4, 3)), "views", 5])
    def test_rejects_non_list_input(self, bad):
        with pytest.raises(TypeError, match="list of array-like"):
            missing_view_mask(bad)

    def test_rejects_unknown_how(self):
        with pytest.raises(ValueError, match="how must be"):
            missing_view_mask(_correlated_views(5), how="some")


class TestSimulateMissingViews:
    def test_mask_matches_the_returned_views(self):
        views = _correlated_views(200)
        incomplete, mask = simulate_missing_views(views, 0.3, random_state=0)
        assert np.array_equal(missing_view_mask(incomplete), mask)

    def test_rate_is_approximately_respected(self):
        views = _correlated_views(500)
        _, mask = simulate_missing_views(views, 0.4, random_state=0, keep_one=False)
        assert abs(mask.mean() - 0.4) < 0.05

    def test_per_view_rates(self):
        views = _correlated_views(500)
        _, mask = simulate_missing_views(
            views, [0.0, 0.5, 0.0], random_state=0, keep_one=False
        )
        assert mask[:, 0].sum() == 0
        assert mask[:, 2].sum() == 0
        assert abs(mask[:, 1].mean() - 0.5) < 0.06

    def test_keep_one_leaves_every_sample_with_a_view(self):
        views = _correlated_views(200)
        _, mask = simulate_missing_views(views, 0.9, random_state=0, keep_one=True)
        assert np.all(mask.sum(axis=1) < mask.shape[1])

    def test_keep_one_false_can_empty_a_sample(self):
        views = _correlated_views(200)
        _, mask = simulate_missing_views(views, 0.9, random_state=0, keep_one=False)
        assert np.any(mask.all(axis=1))

    def test_is_reproducible(self):
        views = _correlated_views(50)
        _, first = simulate_missing_views(views, 0.3, random_state=7)
        _, second = simulate_missing_views(views, 0.3, random_state=7)
        assert np.array_equal(first, second)

    @pytest.mark.parametrize("rate", [-0.1, 1.5])
    def test_rejects_rate_outside_unit_interval(self, rate):
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            simulate_missing_views(_correlated_views(10), rate)


class TestImputerContract:
    """Behaviour every imputer must share."""

    @pytest.mark.parametrize("imputer_cls", IMPUTERS)
    def test_output_has_no_nan(self, imputer_cls):
        incomplete, _ = simulate_missing_views(_correlated_views(), 0.3, random_state=0)
        filled = imputer_cls().fit_transform(incomplete)
        assert not any(np.isnan(v).any() for v in filled)

    @pytest.mark.parametrize("imputer_cls", IMPUTERS)
    def test_shapes_are_preserved(self, imputer_cls):
        views = _correlated_views()
        incomplete, _ = simulate_missing_views(views, 0.3, random_state=0)
        filled = imputer_cls().fit_transform(incomplete)
        assert [v.shape for v in filled] == [v.shape for v in views]

    @pytest.mark.parametrize("imputer_cls", IMPUTERS)
    def test_observed_entries_are_untouched(self, imputer_cls):
        views = _correlated_views()
        incomplete, mask = simulate_missing_views(views, 0.3, random_state=0)
        filled = imputer_cls().fit_transform(incomplete)
        for original, out, column in zip(views, filled, mask.T):
            assert np.allclose(out[~column], original[~column])

    @pytest.mark.parametrize("imputer_cls", IMPUTERS)
    def test_input_is_not_mutated(self, imputer_cls):
        incomplete, _ = simulate_missing_views(_correlated_views(), 0.3, random_state=0)
        before = [v.copy() for v in incomplete]
        imputer_cls().fit_transform(incomplete)
        for original, untouched in zip(incomplete, before):
            assert np.array_equal(
                np.isnan(original), np.isnan(untouched)
            ) and np.allclose(original, untouched, equal_nan=True)

    @pytest.mark.parametrize("imputer_cls", IMPUTERS)
    def test_complete_input_is_returned_unchanged(self, imputer_cls):
        views = _correlated_views()
        filled = imputer_cls().fit_transform(views)
        for original, out in zip(views, filled):
            assert np.allclose(out, original)

    @pytest.mark.parametrize("imputer_cls", IMPUTERS)
    def test_transform_before_fit_raises(self, imputer_cls):
        with pytest.raises(NotFittedError):
            imputer_cls().transform(_correlated_views(10))

    @pytest.mark.parametrize("imputer_cls", IMPUTERS)
    def test_transform_applies_to_unseen_samples(self, imputer_cls):
        train, _ = simulate_missing_views(_correlated_views(), 0.3, random_state=0)
        test, _ = simulate_missing_views(
            _correlated_views(30, seed=9), 0.3, random_state=1
        )
        filled = imputer_cls().fit(train).transform(test)
        assert not any(np.isnan(v).any() for v in filled)
        assert [v.shape[0] for v in filled] == [30, 30, 30]

    @pytest.mark.parametrize("imputer_cls", IMPUTERS)
    def test_transform_rejects_a_different_number_of_features(self, imputer_cls):
        imputer = imputer_cls().fit(_correlated_views(20))
        with pytest.raises(ValueError, match="features"):
            imputer.transform([np.zeros((20, 99)) for _ in range(3)])

    @pytest.mark.parametrize("imputer_cls", IMPUTERS)
    def test_n_views_is_enforced(self, imputer_cls):
        with pytest.raises(ValueError, match="expects 2 views"):
            imputer_cls(n_views=2).fit(_correlated_views(10))

    @pytest.mark.parametrize("imputer_cls", IMPUTERS)
    def test_fit_records_missingness_diagnostics(self, imputer_cls):
        incomplete, mask = simulate_missing_views(
            _correlated_views(), 0.3, random_state=0
        )
        imputer = imputer_cls().fit(incomplete)
        assert np.array_equal(imputer.missing_view_mask_, mask)
        assert np.allclose(imputer.missing_rate_, mask.mean(axis=0))
        assert imputer.n_views_in_ == 3
        assert imputer.n_features_in_ == [5, 3, 6]

    @pytest.mark.parametrize("imputer_cls", IMPUTERS)
    def test_is_sklearn_clonable(self, imputer_cls):
        assert clone(imputer_cls()).get_params() == imputer_cls().get_params()

    @pytest.mark.parametrize("imputer_cls", IMPUTERS)
    def test_multiview_dataset_in_dataset_out(self, imputer_cls):
        views = _correlated_views(60)
        incomplete, _ = simulate_missing_views(views, 0.3, random_state=0)
        mvd = MultiViewDataset(
            incomplete, labels=np.arange(60), view_names=["a", "b", "c"]
        )
        filled = imputer_cls().fit_transform(mvd)
        assert isinstance(filled, MultiViewDataset)
        assert filled.view_names == ["a", "b", "c"]
        assert np.array_equal(filled.labels, np.arange(60))

    @pytest.mark.parametrize("imputer_cls", IMPUTERS)
    def test_sample_with_no_observed_view_falls_back_to_the_mean(self, imputer_cls):
        views = _correlated_views(60)
        incomplete = mark_missing_views(
            views, np.array([[True, True, True]] + [[False] * 3] * 59)
        )
        filled = imputer_cls().fit_transform(incomplete)
        for out, view in zip(filled, views):
            assert not np.isnan(out[0]).any()
            assert np.allclose(out[0], view[1:].mean(axis=0))

    @pytest.mark.parametrize("imputer_cls", IMPUTERS)
    def test_view_missing_everywhere_does_not_raise(self, imputer_cls):
        views = _correlated_views(40)
        views[1][:] = np.nan
        filled = imputer_cls().fit_transform(views)
        assert np.allclose(filled[1], 0.0)

    @pytest.mark.parametrize("imputer_cls", IMPUTERS)
    def test_single_view_still_produces_a_complete_output(self, imputer_cls):
        X = _correlated_views(40)[0]
        X[::4] = np.nan
        filled = imputer_cls().fit_transform([X])
        assert not np.isnan(filled[0]).any()

    @pytest.mark.parametrize("imputer_cls", IMPUTERS)
    def test_partially_missing_rows_are_completed(self, imputer_cls):
        views = _correlated_views(60)
        views[0][3, 0] = np.nan  # a single cell, not a whole view
        filled = imputer_cls().fit_transform(views)
        assert not np.isnan(filled[0]).any()


class TestSimpleViewImputer:
    def test_mean_strategy_uses_observed_column_means(self):
        X1 = np.array([[1.0, 2.0], [3.0, 4.0], [np.nan, np.nan]])
        X2 = np.array([[5.0], [6.0], [7.0]])
        filled = SimpleViewImputer().fit_transform([X1, X2])
        assert filled[0][2].tolist() == [2.0, 3.0]

    def test_median_strategy_resists_outliers(self):
        X1 = np.array([[1.0], [2.0], [1000.0], [np.nan]])
        X2 = np.ones((4, 1))
        filled = SimpleViewImputer(strategy="median").fit_transform([X1, X2])
        assert filled[0][3, 0] == 2.0

    def test_constant_strategy_uses_fill_value(self):
        X1 = np.array([[1.0], [np.nan]])
        X2 = np.ones((2, 1))
        filled = SimpleViewImputer(strategy="constant", fill_value=-7.0).fit_transform(
            [X1, X2]
        )
        assert filled[0][1, 0] == -7.0

    def test_statistics_come_from_fit_not_transform(self):
        train = [np.array([[0.0], [2.0]]), np.ones((2, 1))]
        test = [np.array([[100.0], [np.nan]]), np.ones((2, 1))]
        filled = SimpleViewImputer().fit(train).transform(test)
        assert filled[0][1, 0] == 1.0  # the training mean, not the test mean

    def test_rejects_unknown_strategy(self):
        with pytest.raises(ValueError, match="strategy must be"):
            SimpleViewImputer(strategy="mode").fit(_correlated_views(10))


class TestKNNViewImputer:
    def test_copies_the_nearest_donor(self):
        X1 = np.array([[0.0], [0.1], [5.0], [5.1]])
        X2 = np.array([[1.0], [1.2], [9.0], [np.nan]])
        filled = KNNViewImputer(n_neighbors=1).fit_transform([X1, X2])
        assert filled[1][3, 0] == 9.0

    def test_averages_several_donors(self):
        X1 = np.array([[0.0], [0.0], [0.0]])
        X2 = np.array([[2.0], [4.0], [np.nan]])
        filled = KNNViewImputer(n_neighbors=2).fit_transform([X1, X2])
        assert filled[1][2, 0] == pytest.approx(3.0)

    def test_distance_weighting_favours_the_closer_donor(self):
        X1 = np.array([[0.0], [10.0], [0.5]])
        X2 = np.array([[0.0], [100.0], [np.nan]])
        uniform = KNNViewImputer(n_neighbors=2).fit_transform([X1, X2])[1][2, 0]
        weighted = KNNViewImputer(n_neighbors=2, weights="distance").fit_transform(
            [X1, X2]
        )[1][2, 0]
        assert weighted < uniform

    def test_neighbours_are_capped_at_the_number_of_donors(self):
        X1 = np.array([[0.0], [1.0], [2.0]])
        X2 = np.array([[10.0], [20.0], [np.nan]])
        filled = KNNViewImputer(n_neighbors=50).fit_transform([X1, X2])
        assert filled[1][2, 0] == pytest.approx(15.0)

    def test_beats_the_marginal_baseline_on_correlated_views(self):
        views = _correlated_views(200)
        incomplete, mask = simulate_missing_views(views, 0.25, random_state=0)

        def error(imputer):
            filled = imputer.fit_transform(incomplete)
            return np.mean(
                [
                    np.mean((out[column] - truth[column]) ** 2)
                    for out, truth, column in zip(filled, views, mask.T)
                ]
            )

        assert error(KNNViewImputer(n_neighbors=5)) < error(SimpleViewImputer())

    @pytest.mark.parametrize("n_neighbors", [0, -3, 2.5])
    def test_rejects_invalid_n_neighbors(self, n_neighbors):
        with pytest.raises(ValueError, match="n_neighbors"):
            KNNViewImputer(n_neighbors=n_neighbors).fit(_correlated_views(10))

    def test_rejects_unknown_weights(self):
        with pytest.raises(ValueError, match="weights must be"):
            KNNViewImputer(weights="gaussian").fit(_correlated_views(10))


class TestCrossViewRegressionImputer:
    def test_recovers_a_linear_relation_between_views(self):
        rng = np.random.default_rng(0)
        X1 = rng.normal(size=(200, 3))
        mapping = rng.normal(size=(3, 2))
        X2 = X1 @ mapping
        incomplete = mark_missing_views(
            [X1, X2], np.array([[False, True]] * 10 + [[False, False]] * 190)
        )
        filled = CrossViewRegressionImputer(estimator=LinearRegression()).fit_transform(
            incomplete
        )
        assert np.allclose(filled[1][:10], X2[:10], atol=1e-6)

    def test_accepts_a_custom_estimator(self):
        incomplete, _ = simulate_missing_views(_correlated_views(), 0.3, random_state=0)
        imputer = CrossViewRegressionImputer(estimator=LinearRegression())
        filled = imputer.fit_transform(incomplete)
        assert all(isinstance(e, LinearRegression) for e in imputer.estimators_)
        assert not any(np.isnan(v).any() for v in filled)

    def test_estimator_params_are_applied(self):
        incomplete, _ = simulate_missing_views(_correlated_views(), 0.3, random_state=0)
        imputer = CrossViewRegressionImputer(estimator_params={"alpha": 42.0}).fit(
            incomplete
        )
        assert all(e.alpha == 42.0 for e in imputer.estimators_)

    def test_records_the_training_size_of_each_view(self):
        incomplete, mask = simulate_missing_views(
            _correlated_views(), 0.3, random_state=0
        )
        imputer = CrossViewRegressionImputer().fit(incomplete)
        assert imputer.n_train_samples_.tolist() == (~mask).sum(axis=0).tolist()

    def test_skips_a_view_with_too_few_observed_samples(self):
        views = _correlated_views(40)
        views[1][1:] = np.nan  # a single observed sample left
        imputer = CrossViewRegressionImputer().fit(views)
        assert imputer.estimators_[1] is None
        assert imputer.estimators_[0] is not None

    def test_beats_the_marginal_baseline_on_correlated_views(self):
        views = _correlated_views(200)
        incomplete, mask = simulate_missing_views(views, 0.25, random_state=0)

        def error(imputer):
            filled = imputer.fit_transform(incomplete)
            return np.mean(
                [
                    np.mean((out[column] - truth[column]) ** 2)
                    for out, truth, column in zip(filled, views, mask.T)
                ]
            )

        assert error(CrossViewRegressionImputer()) < error(SimpleViewImputer())


class TestDropIncompleteSamples:
    def test_keeps_only_complete_samples(self):
        X1 = np.array([[1.0], [np.nan], [3.0]])
        X2 = np.array([[4.0], [5.0], [6.0]])
        dropper = DropIncompleteSamples()
        kept = dropper.fit_transform([X1, X2])
        assert kept[0].ravel().tolist() == [1.0, 3.0]
        assert kept[1].ravel().tolist() == [4.0, 6.0]
        assert dropper.support_mask_.tolist() == [True, False, True]
        assert dropper.n_dropped_ == 1

    def test_max_missing_views_tolerates_partial_samples(self):
        views = _correlated_views(50)
        incomplete, mask = simulate_missing_views(views, 0.4, random_state=0)
        kept = DropIncompleteSamples(max_missing_views=1).fit_transform(incomplete)
        assert kept[0].shape[0] == int((mask.sum(axis=1) <= 1).sum())

    def test_how_any_also_drops_partially_missing_rows(self):
        X1 = np.array([[1.0, np.nan], [3.0, 4.0]])
        X2 = np.ones((2, 1))
        assert DropIncompleteSamples().fit_transform([X1, X2])[0].shape[0] == 2
        assert DropIncompleteSamples(how="any").fit_transform([X1, X2])[0].shape[0] == 1

    def test_subsets_dataset_labels(self):
        views = _correlated_views(50)
        incomplete, mask = simulate_missing_views(views, 0.3, random_state=0)
        mvd = MultiViewDataset(incomplete, labels=np.arange(50))
        kept = DropIncompleteSamples().fit_transform(mvd)
        assert isinstance(kept, MultiViewDataset)
        assert kept.labels.tolist() == np.flatnonzero(~mask.any(axis=1)).tolist()

    def test_function_and_transformer_agree(self):
        incomplete, _ = simulate_missing_views(
            _correlated_views(50), 0.3, random_state=0
        )
        by_function = drop_missing_views(incomplete)
        by_transformer = DropIncompleteSamples().fit_transform(incomplete)
        for a, b in zip(by_function, by_transformer):
            assert np.array_equal(a, b)

    def test_transform_before_fit_raises(self):
        with pytest.raises(NotFittedError):
            DropIncompleteSamples().transform(_correlated_views(10))

    def test_rejects_a_different_number_of_views(self):
        dropper = DropIncompleteSamples().fit(_correlated_views(10))
        with pytest.raises(ValueError, match="Fitted on 3 views"):
            dropper.transform(_correlated_views(10)[:2])

    def test_rejects_negative_max_missing_views(self):
        with pytest.raises(ValueError, match="non-negative integer"):
            drop_missing_views(_correlated_views(10), max_missing_views=-1)


class TestIntegration:
    @pytest.mark.parametrize("imputer_cls", IMPUTERS)
    def test_runs_as_the_first_step_of_a_polypipeline(self, imputer_cls):
        mvd = pv.make_multiview_gaussian(
            n_samples=90, n_features=6, n_views=3, centers=3, random_state=0
        )
        incomplete, _ = simulate_missing_views(mvd, 0.2, random_state=0)
        pipe = pv.PolyPipeline(
            steps=[
                ("impute", imputer_cls()),
                ("fuse", pv.NormalizedFusion()),
                ("kmeans", KMeans(n_clusters=3, n_init=10, random_state=0)),
            ]
        )
        labels = pipe.fit_predict(incomplete)
        assert labels.shape == (90,)
        assert len(np.unique(labels)) == 3

    @pytest.mark.parametrize("imputer_cls", IMPUTERS)
    def test_feeds_a_native_multiview_clusterer(self, imputer_cls):
        mvd = pv.make_multiview_gaussian(
            n_samples=90, n_features=6, n_views=3, centers=3, random_state=0
        )
        incomplete, _ = simulate_missing_views(mvd, 0.2, random_state=0)
        filled = imputer_cls().fit_transform(incomplete)
        labels = pv.MultiViewKMeans(n_clusters=3, random_state=0).fit_predict(filled)
        assert labels.shape == (90,)

    def test_exposed_at_the_package_root(self):
        assert pv.SimpleViewImputer is SimpleViewImputer
        assert pv.imputation.KNNViewImputer is KNNViewImputer
