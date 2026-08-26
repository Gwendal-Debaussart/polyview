import numpy as np
import pytest

from polyview.dataset.make_multiview_gaussian import make_multiview_gaussian
from polyview.dataset.multiviewdataset import MultiViewDataset


def _make_dataset(n_samples=40, seed=0):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=(n_samples, 5))
    x2 = rng.normal(size=(n_samples, 3))
    labels = rng.integers(0, 3, size=n_samples)
    return MultiViewDataset([x1, x2], labels=labels, view_names=["audio", "video"])


class TestMultiViewDataset:
    def test_basic_properties(self):
        mvd = _make_dataset()
        assert mvd.n_views == 2
        assert mvd.n_samples == 40
        assert mvd.n_features == [5, 3]
        assert len(mvd) == 2

    def test_getitem_by_index_and_name(self):
        mvd = _make_dataset()
        assert np.array_equal(mvd[0], mvd["audio"])
        assert mvd[0].shape == (40, 5)
        assert mvd[1].shape == (40, 3)

    def test_getitem_unknown_name_raises(self):
        mvd = _make_dataset()
        with pytest.raises(KeyError):
            mvd["nonexistent"]

    def test_getitem_invalid_key_type_raises(self):
        mvd = _make_dataset()
        with pytest.raises(TypeError):
            mvd[1.5]

    def test_views_are_copies(self):
        mvd = _make_dataset()
        v = mvd[0]
        v[:] = 0.0
        assert not np.allclose(mvd[0], 0.0)

    def test_mismatched_sample_counts_raise(self):
        x1 = np.random.rand(10, 3)
        x2 = np.random.rand(11, 3)
        with pytest.raises(ValueError, match="same number of samples"):
            MultiViewDataset([x1, x2])

    def test_mismatched_labels_length_raises(self):
        x1 = np.random.rand(10, 3)
        with pytest.raises(ValueError, match="labels has"):
            MultiViewDataset([x1], labels=np.zeros(5))

    def test_default_view_names(self):
        x1 = np.random.rand(10, 3)
        x2 = np.random.rand(10, 4)
        mvd = MultiViewDataset([x1, x2])
        assert mvd.view_names == ["view_0", "view_1"]

    def test_subset_samples(self):
        mvd = _make_dataset(n_samples=20, seed=1)
        subset = mvd.subset_samples([0, 1, 2])
        assert subset.n_samples == 3
        assert np.array_equal(subset[0], mvd[0][:3])
        assert np.array_equal(subset.labels, mvd.labels[:3])

    def test_subset_views(self):
        mvd = _make_dataset(seed=2)
        subset = mvd.subset_views(["video"])
        assert subset.n_views == 1
        assert subset.view_names == ["video"]
        assert np.array_equal(subset[0], mvd["video"])

    def test_train_test_split_sizes(self):
        mvd = _make_dataset(n_samples=50, seed=3)
        train, test = mvd.train_test_split(test_size=0.2, random_state=0)
        assert train.n_samples + test.n_samples == 50
        assert test.n_samples == 10

    def test_train_test_split_stratified_preserves_class_ratio_roughly(self):
        rng = np.random.default_rng(4)
        x1 = rng.normal(size=(90, 3))
        labels = np.repeat([0, 1, 2], 30)
        mvd = MultiViewDataset([x1], labels=labels)
        train, test = mvd.train_test_split(
            test_size=0.2, random_state=0, stratify=True
        )
        for cls in [0, 1, 2]:
            assert np.sum(test.labels == cls) == 6

    def test_to_numpy_concatenates_views(self):
        mvd = _make_dataset(seed=5)
        flat = mvd.to_numpy()
        assert flat.shape == (40, 8)
        assert np.array_equal(flat, np.concatenate([mvd[0], mvd[1]], axis=1))

    def test_save_and_load_round_trip(self, tmp_path):
        mvd = _make_dataset(seed=6)
        path = tmp_path / "dataset.npz"
        mvd.save(str(path))
        loaded = MultiViewDataset.load(str(path))

        assert loaded.n_views == mvd.n_views
        assert loaded.n_samples == mvd.n_samples
        assert loaded.view_names == mvd.view_names
        assert np.array_equal(loaded.labels, mvd.labels)
        for a, b in zip(loaded.views, mvd.views):
            assert np.allclose(a, b)

    def test_repr_contains_summary_info(self):
        mvd = _make_dataset(n_samples=15, seed=7)
        text = repr(mvd)
        assert "n_views=2" in text
        assert "n_samples=15" in text


class TestMakeMultiviewGaussian:
    def test_default_shapes(self):
        mvd = make_multiview_gaussian(
            n_samples=100, n_features=8, n_views=3, centers=4, random_state=0
        )
        assert mvd.n_views == 3
        assert mvd.n_samples == 100
        assert mvd.n_features == [8, 8, 8]
        assert mvd.labels.shape == (100,)
        assert set(np.unique(mvd.labels)).issubset(set(range(4)))

    def test_reproducible_with_same_seed(self):
        mvd1 = make_multiview_gaussian(n_samples=50, random_state=42)
        mvd2 = make_multiview_gaussian(n_samples=50, random_state=42)
        assert np.array_equal(mvd1.labels, mvd2.labels)
        for a, b in zip(mvd1.views, mvd2.views):
            assert np.allclose(a, b)

    def test_low_noise_views_share_pairwise_geometry(self):
        mvd = make_multiview_gaussian(
            n_samples=150,
            n_features=10,
            latent_dim=3,
            centers=3,
            n_views=2,
            noise_std=0.01,
            random_state=0,
        )
        # Both views are noisy projections of the same shared latent factor,
        # so their pairwise-distance structures should be strongly correlated.
        from sklearn.metrics import pairwise_distances

        v1, v2 = mvd.views
        d1 = pairwise_distances(v1)
        d2 = pairwise_distances(v2)
        iu = np.triu_indices_from(d1, k=1)
        corr = np.corrcoef(d1[iu], d2[iu])[0, 1]
        assert corr > 0.8
