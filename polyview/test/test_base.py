import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.exceptions import NotFittedError

from polyview.base import BaseMultiView
from polyview.dataset.multiviewdataset import MultiViewDataset


class _DummyMultiView(BaseMultiView):
    def fit(self, views, y=None):
        self._validate_views(views)
        return self


def _make_views(n_samples=10, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.normal(size=(n_samples, 3)), rng.normal(size=(n_samples, 2))]


class TestValidateViews:
    def test_accepts_multiview_dataset(self):
        model = _DummyMultiView().fit(MultiViewDataset(_make_views()))
        assert model.n_views_in_ == 2
        assert model.n_samples_ == 10
        assert model.n_features_in_ == [3, 2]

    @pytest.mark.parametrize(
        "views, error, match",
        [
            (np.zeros((10, 3)), TypeError, "list of array-like"),
            ([np.zeros(10)], ValueError, "must be 2-D"),
            ([], ValueError, "At least one view"),
            (
                [np.zeros((10, 3)), np.zeros((9, 3))],
                ValueError,
                "same number of samples",
            ),
        ],
    )
    def test_invalid_views_raise(self, views, error, match):
        with pytest.raises(error, match=match):
            _DummyMultiView().fit(views)

    def test_expected_number_of_views(self):
        with pytest.raises(ValueError, match="expects 3 views"):
            _DummyMultiView(n_views=3).fit(_make_views())

    def test_consistency_checks_after_fit(self):
        model = _DummyMultiView().fit(_make_views())
        assert len(model._validate_views(_make_views(seed=1), reset=False)) == 2
        with pytest.raises(ValueError, match="Fitted on 2 views"):
            model._validate_views(_make_views()[:1], reset=False)
        with pytest.raises(ValueError, match="fitted with 3"):
            model._validate_views([np.zeros((10, 4)), np.zeros((10, 2))], reset=False)

    def test_consistency_checks_require_fit(self):
        with pytest.raises(NotFittedError):
            _DummyMultiView()._validate_views(_make_views(), reset=False)

    def test_sparse_views_are_kept_when_accepted(self):
        views = [
            sp.random(10, 3, density=0.5, format="csr", random_state=0),
            np.zeros((10, 2)),
        ]
        out = _DummyMultiView()._validate_views(views, accept_sparse=True)
        assert sp.issparse(out[0])
        assert isinstance(out[1], np.ndarray)

    def test_more_tags(self):
        assert _DummyMultiView()._more_tags()["no_validation"] is True
