import numpy as np
import pytest
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

from polyview.augmentation.view_splitter import ViewSplitter
from polyview.dataset.multiviewdataset import MultiViewDataset
from polyview.embed.gcca import GCCA
from polyview.fusion.early import ConcatFusion
from polyview.fusion.late import MajorityVote
from polyview.pipeline.polypipeline import PolyPipeline


class _InPlaceZeroingStep(BaseEstimator, TransformerMixin):
    """Sklearn-style per-view transformer that mutates its input in place.

    Used to prove that PolyPipeline reads views through the public,
    copy-returning ``.views`` property rather than the private ``_views``
    attribute, so a mutating step can't corrupt the caller's dataset.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X[:] = 0.0
        return X


def _make_views(n_samples=120, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n_samples, 3))
    x1 = z @ rng.normal(size=(3, 5)) + 1.5 + 0.05 * rng.normal(size=(n_samples, 5))
    x2 = z @ rng.normal(size=(3, 4)) - 2.0 + 0.05 * rng.normal(size=(n_samples, 4))
    return [x1, x2]


def test_mv_to_sv_pipeline_with_sklearn_tail_fit_predict():
    views = _make_views()
    pipe = PolyPipeline(
        steps=[
            ("fuse", ConcatFusion()),
            ("scale", StandardScaler()),
            ("cluster", KMeans(n_clusters=3, random_state=0, n_init=5)),
        ]
    )

    labels = pipe.fit_predict(views)

    assert labels.shape == (120,)
    assert set(np.unique(labels)).issubset({0, 1, 2})


def test_pipeline_rejects_mv_step_after_switch_to_sv():
    views = _make_views(n_samples=40, seed=12)
    pipe = PolyPipeline(
        steps=[
            ("fuse", ConcatFusion()),
            ("gcca", GCCA(n_components=2)),
        ]
    )

    with pytest.raises(ValueError, match="requires multi-view input"):
        pipe.fit(views)


def test_direct_sklearn_usage_from_single_view_input():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(80, 10))

    pipe = PolyPipeline(
        steps=[
            ("scale", StandardScaler()),
            ("pca", PCA(n_components=4, random_state=0)),
        ]
    )

    Z = pipe.fit_transform(X)
    assert Z.shape == (80, 4)


def test_single_view_transformer_is_applied_per_view_in_mv_mode():
    views = _make_views(n_samples=60, seed=5)
    pipe = PolyPipeline(steps=[("scale", StandardScaler())])

    scaled_views = pipe.fit_transform(views)

    assert isinstance(scaled_views, list)
    assert len(scaled_views) == 2
    for view in scaled_views:
        assert np.allclose(view.mean(axis=0), 0.0, atol=1e-10)
        assert np.allclose(view.std(axis=0), 1.0, atol=1e-8)


def test_per_view_step_params_override_by_step_name():
    views = _make_views(n_samples=50, seed=7)
    pipe = PolyPipeline(
        steps=[("scale", StandardScaler())],
        per_view_step_params={
            "scale": [
                {"with_std": False},
                {"with_mean": False},
            ]
        },
    )

    out = pipe.fit_transform(views)

    # View 0: centered but not scaled.
    assert np.allclose(out[0].mean(axis=0), 0.0, atol=1e-10)
    assert not np.allclose(out[0].std(axis=0), 1.0, atol=1e-3)

    # View 1: scaled but not centered.
    assert np.allclose(out[1].std(axis=0), 1.0, atol=1e-8)
    assert not np.allclose(out[1].mean(axis=0), 0.0, atol=1e-3)


def test_draw_requires_start_mode_when_unfitted():
    pytest.importorskip("networkx")
    pipe = PolyPipeline(steps=[("scale", StandardScaler())])
    with pytest.raises(ValueError, match="Unfitted pipeline"):
        pipe.draw()


def test_draw_returns_graph_when_dependencies_available():
    nx = pytest.importorskip("networkx")
    pytest.importorskip("matplotlib")

    pipe = PolyPipeline(steps=[("scale", StandardScaler())])

    graph = pipe.draw(start_mode="mv", show=False)

    assert isinstance(graph, nx.DiGraph)
    assert "input" in graph.nodes
    assert "output" in graph.nodes
    assert graph.has_edge("input", "step_1")
    assert graph.has_edge("step_1", "output")


def test_draw_accepts_custom_style_arguments():
    nx = pytest.importorskip("networkx")
    plt = pytest.importorskip("matplotlib.pyplot")

    pipe = PolyPipeline(steps=[("scale", StandardScaler())])
    fig, ax = plt.subplots(figsize=(4, 6))
    graph = pipe.draw(
        start_mode="mv",
        ax=ax,
        show=False,
        mode_colors={
            "mv": "#1f77b4",
            "sv": "#ff7f0e",
            "lf": "#2ca02c",
            "default": "#cccccc",
        },
        title="Custom Pipeline",
        node_text_color="#ffffff",
        node_border_color="#000000",
        edge_color="#111111",
        transition_text_color="#222222",
    )

    assert isinstance(graph, nx.DiGraph)
    assert ax.get_title() == "Custom Pipeline"


def test_draw_explicit_start_mode_ignores_fitted_wrappers():
    nx = pytest.importorskip("networkx")
    pytest.importorskip("matplotlib")

    views = _make_views(n_samples=40, seed=19)
    pipe = PolyPipeline(
        steps=[
            ("scale", StandardScaler()),
            ("kmeans", KMeans(n_clusters=3, random_state=0, n_init=5)),
        ]
    )

    pipe.fit(views)
    graph = pipe.draw(start_mode="sv", show=False)

    assert isinstance(graph, nx.DiGraph)
    assert graph.nodes["step_1"]["mode"] == "sv"
    assert graph.nodes["step_2"]["mode"] == "sv"


def test_multiview_dataset_input_is_recognized_as_mv():
    x1, x2 = _make_views(n_samples=30, seed=20)
    mvd = MultiViewDataset([x1, x2])
    pipe = PolyPipeline(steps=[("fuse", ConcatFusion())])

    fused = pipe.fit_transform(mvd)
    assert fused.shape == (30, x1.shape[1] + x2.shape[1])


def test_mutating_step_does_not_corrupt_source_multiview_dataset():
    # Regression test: PolyPipeline must read views through the public
    # MultiViewDataset.views property (which returns copies), not the
    # private _views attribute, or an in-place-mutating step would corrupt
    # the caller's original dataset.
    x1, x2 = _make_views(n_samples=10, seed=21)
    mvd = MultiViewDataset([x1, x2])
    pipe = PolyPipeline(steps=[("zero", _InPlaceZeroingStep())])

    pipe.fit_transform(mvd)

    assert not np.allclose(mvd[0], 0.0)
    assert not np.allclose(mvd[1], 0.0)
    assert np.allclose(mvd[0], x1)
    assert np.allclose(mvd[1], x2)


def _make_labelled_views(n_samples=90, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.permutation(np.repeat(np.arange(3), n_samples // 3))
    x1 = rng.normal(scale=4.0, size=(3, 5))[y] + rng.normal(size=(n_samples, 5))
    x2 = rng.normal(scale=4.0, size=(3, 4))[y] + rng.normal(size=(n_samples, 4))
    return [x1, x2], y


def test_get_and_set_nested_step_parameters():
    pipe = PolyPipeline(
        steps=[("scale", StandardScaler()), ("km", KMeans(n_clusters=3, n_init=2))]
    )
    params = pipe.get_params()
    assert params["km__n_clusters"] == 3
    assert "scale__with_mean" in params

    pipe.set_params(km__n_clusters=4)
    assert pipe.named_steps["km"].n_clusters == 4


def test_set_params_can_replace_a_step():
    pipe = PolyPipeline(
        steps=[("scale", StandardScaler()), ("km", KMeans(n_clusters=3, n_init=2))]
    )
    pipe.set_params(scale="passthrough")
    assert pipe.named_steps["scale"] == "passthrough"


def test_clone_copies_steps_and_parameters():
    pipe = PolyPipeline(steps=[("km", KMeans(n_clusters=5, n_init=2))], name="demo")
    cloned = clone(pipe)
    assert cloned.named_steps["km"] is not pipe.named_steps["km"]
    assert cloned.get_params()["km__n_clusters"] == 5
    assert cloned.name == "demo"


def test_grid_search_over_multiview_pipeline_with_view_splitter():
    views, y = _make_labelled_views()
    X = np.hstack(views)
    pipe = PolyPipeline(
        steps=[
            ("split", ViewSplitter(n_features=[5, 4])),
            ("gcca", GCCA(n_components=2)),
            ("km", KMeans(n_clusters=3, n_init=5, random_state=0)),
        ]
    )
    search = GridSearchCV(
        pipe,
        {"gcca__n_components": [1, 2]},
        scoring=lambda est, X_, y_: adjusted_rand_score(y_, est.predict(X_)),
        cv=3,
    )
    search.fit(X, y)

    assert search.best_params_["gcca__n_components"] in (1, 2)
    assert search.best_estimator_.predict(X).shape == (90,)
    assert search.best_score_ > 0.8


def test_predict_and_score_with_single_view_classifier():
    views, y = _make_labelled_views(seed=1)
    X = np.hstack(views)
    pipe = PolyPipeline(
        steps=[("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=500))]
    ).fit(X, y)

    assert pipe.predict(X).shape == (90,)
    assert 0.9 <= pipe.score(X, y) <= 1.0


def test_per_view_classifiers_predict_and_score_in_multiview_mode():
    views, y = _make_labelled_views(seed=2)
    pipe = PolyPipeline(
        steps=[("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=500))]
    ).fit(views, y)

    preds = pipe.predict(views)
    assert isinstance(preds, list) and len(preds) == 2
    assert all(p.shape == (90,) for p in preds)
    assert pipe.score(views, y).shape == (2,)


def test_late_fusion_of_per_view_classifiers():
    views, y = _make_labelled_views(seed=3)
    pipe = PolyPipeline(
        steps=[("clf", LogisticRegression(max_iter=500)), ("vote", MajorityVote())]
    )
    labels = pipe.fit_predict(views, y)
    assert np.mean(labels == y) > 0.9


def test_late_fusion_of_per_view_clusterings_with_label_alignment():
    views, y = _make_labelled_views(seed=4)
    pipe = PolyPipeline(
        steps=[
            ("km", KMeans(n_clusters=3, n_init=5, random_state=0)),
            ("vote", MajorityVote(align_labels=True)),
        ]
    )
    labels = pipe.fit_predict(views)
    assert adjusted_rand_score(y, labels) > 0.9


@pytest.mark.parametrize(
    "steps, error, match",
    [
        ([], ValueError, "must not be empty"),
        ([("a", StandardScaler()), ("a", StandardScaler())], ValueError, "unique"),
        ([["a", StandardScaler()]], TypeError, "tuple"),
        ([(1, StandardScaler())], TypeError, "non-string name"),
        ([("a", object())], TypeError, "fit"),
    ],
)
def test_invalid_steps_raise(steps, error, match):
    with pytest.raises(error, match=match):
        PolyPipeline(steps=steps).fit(np.zeros((5, 2)))


def test_invalid_input_type_raises():
    with pytest.raises(TypeError, match="Input must be"):
        PolyPipeline(steps=[("scale", StandardScaler())]).fit("not data")


def test_passthrough_final_step():
    X = np.random.default_rng(4).normal(size=(20, 3))
    pipe = PolyPipeline(steps=[("scale", StandardScaler()), ("last", "passthrough")])
    pipe.fit(X)

    assert pipe.transform(X).shape == (20, 3)
    for method in ("predict", "score"):
        with pytest.raises(AttributeError, match="passthrough"):
            getattr(pipe, method)(X)


def test_final_step_without_requested_method_raises():
    X = np.random.default_rng(5).normal(size=(20, 3))
    pipe = PolyPipeline(steps=[("scale", StandardScaler())]).fit(X)
    with pytest.raises(AttributeError, match="does not implement predict"):
        pipe.predict(X)


def test_print_diagrams_before_and_after_fit():
    views, _ = _make_labelled_views(seed=6)
    pipe = PolyPipeline(
        steps=[
            ("scale", StandardScaler()),
            ("km", KMeans(n_clusters=3, n_init=2, random_state=0)),
        ]
    )
    assert "unfitted" in pipe.print()

    preview = pipe.print(start_mode="mv")
    assert "[scale: StandardScaler]" in preview
    assert "[km: KMeans]" in preview

    pipe.fit(views)
    assert "per-view labels" in pipe.print()


def test_print_diagram_of_view_splitting():
    pipe = PolyPipeline(
        steps=[
            ("split", ViewSplitter(n_features=[5, 4])),
            ("gcca", GCCA(n_components=2)),
            ("km", KMeans(n_clusters=3, n_init=2)),
        ]
    )
    assert "split 1 -> 2 branches" in pipe.print(start_mode="sv")


def test_draw_fitted_late_fusion_pipeline():
    nx = pytest.importorskip("networkx")
    pytest.importorskip("matplotlib")

    views, y = _make_labelled_views(seed=7)
    pipe = PolyPipeline(
        steps=[("clf", LogisticRegression(max_iter=500)), ("vote", MajorityVote())]
    ).fit(views, y)
    graph = pipe.draw(show=False)

    assert isinstance(graph, nx.DiGraph)
    assert {"input", "step_1", "step_2", "output"} <= set(graph.nodes)
