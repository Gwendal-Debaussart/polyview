import numpy as np
import pytest
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from polyview.dataset.multiviewdataset import MultiViewDataset
from polyview.embed.gcca import GCCA
from polyview.fusion.early import ConcatFusion
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
    pipe = PolyPipeline(steps=[("scale", StandardScaler())])
    with pytest.raises(ValueError, match="Unfitted pipeline"):
        pipe.draw()


def test_draw_returns_graph_when_dependencies_available():
    nx = pytest.importorskip("networkx")
    pytest.importorskip("matplotlib")

    views = _make_views(n_samples=40, seed=11)
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
