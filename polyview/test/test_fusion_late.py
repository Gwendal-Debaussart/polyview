import numpy as np
import pytest

from polyview.fusion.late import MajorityVote


class TestMajorityVote:
    def test_simple_majority(self):
        preds = [
            np.array([0, 1, 1, 0]),
            np.array([0, 1, 0, 0]),
            np.array([1, 1, 1, 1]),
        ]
        fused = MajorityVote().fit_predict(preds)
        assert fused.tolist() == [0, 1, 1, 0]

    def test_weighted_vote_favors_higher_weight_view(self):
        preds = [
            np.array([0, 0]),
            np.array([1, 1]),
        ]
        fused = MajorityVote(weights=[0.1, 5.0]).fit_predict(preds)
        assert fused.tolist() == [1, 1]

    def test_tie_break_first_picks_smallest_label(self):
        preds = [
            np.array([0]),
            np.array([1]),
        ]
        fused = MajorityVote(tie_break="first").fit_predict(preds)
        assert fused.tolist() == [0]

    def test_tie_break_random_is_reproducible(self):
        preds = [
            np.array([0, 2]),
            np.array([1, 0]),
        ]
        fused1 = MajorityVote(tie_break="random", random_state=42).fit_predict(preds)
        fused2 = MajorityVote(tie_break="random", random_state=42).fit_predict(preds)
        assert fused1.tolist() == fused2.tolist()

    def test_mismatched_lengths_raise(self):
        preds = [np.array([0, 1, 0]), np.array([1, 1])]
        with pytest.raises(ValueError, match="same length"):
            MajorityVote().fit_predict(preds)

    def test_non_integer_like_labels_raise(self):
        preds = [np.array(["a", "b"]), np.array(["a", "b"])]
        with pytest.raises(TypeError, match="integer-like labels"):
            MajorityVote().fit_predict(preds)

    def test_negative_weights_raise(self):
        preds = [np.array([0, 1]), np.array([1, 0])]
        with pytest.raises(ValueError, match="non-negative"):
            MajorityVote(weights=[-1.0, 1.0]).fit_predict(preds)

    def test_predict_requires_same_sample_count_as_fit(self):
        train_preds = [np.array([0, 1]), np.array([0, 1])]
        test_preds = [np.array([1, 0, 1]), np.array([1, 1, 1])]
        model = MajorityVote().fit(train_preds)
        with pytest.raises(ValueError, match="Fitted on 2 samples"):
            model.predict(test_preds)

    def test_predict_after_fit_with_matching_sample_count(self):
        train_preds = [np.array([0, 1, 0]), np.array([0, 1, 1])]
        test_preds = [np.array([1, 0, 1]), np.array([1, 1, 1])]
        model = MajorityVote().fit(train_preds)
        fused = model.predict(test_preds)
        # sample 1 is a tie (0 vs 1); tie_break="first" picks the smaller label.
        assert fused.tolist() == [1, 0, 1]
