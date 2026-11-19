"""Unit tests for history windows, masking in the models, losses and dynamics features."""

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fall_detection.learning.features import dynamics_features
from fall_detection.learning.losses import FocalLoss, class_weights, make_loss
from fall_detection.learning.models import Classifier
from fall_detection.learning.prepare import window_arrays


def record():
    """Build a synthetic prepared record for window tests."""
    n = 120
    rng = np.random.default_rng(1)
    return dict(
        times=np.arange(n) / 15,
        valid=np.ones(n, bool),
        ids=np.zeros(n, int),
        pose=rng.normal(size=(n, 63)).astype(np.float32),
        keypoints=rng.normal(size=(n, 51)).astype(np.float32),
        motion=np.zeros((n, 12), np.float32),
        scene=np.zeros((n, 8), np.float32),
        rgb=np.zeros((n, 576), np.float32),
        labels=np.arange(n) % 3,
        ends=np.arange(29, n, 3),
        meta=dict(window=30),
    )


class TemporalTests(unittest.TestCase):
    """Causality and masking properties of the temporal models and features."""
    def test_long_history_preserves_endpoints_and_warmup(self):
        r = record()
        short = window_arrays([r], "pose_lstm", history=30, return_lengths=True)
        long = window_arrays([r], "pose_lstm", history=90, return_lengths=True)
        self.assertEqual(short[2], long[2])
        np.testing.assert_array_equal(short[1], long[1])
        self.assertEqual(long[3][0], 30)
        self.assertEqual(long[3][-1], 90)
        np.testing.assert_array_equal(long[0][0, 30:], 0)

    def test_long_history_never_crosses_identity(self):
        r = record()
        r["ids"][:60] = 4
        r["ids"][60:] = 5
        r["ends"] = np.array([89])
        x, y, refs, lengths = window_arrays([r], "pose_lstm", history=90, return_lengths=True)
        self.assertEqual(lengths[0], 30)
        np.testing.assert_allclose(x[0, :30, :-1], r["pose"][60:90])

    def test_future_does_not_change_current_input(self):
        r = record()
        r["ends"] = np.array([59])
        a = window_arrays([r], "pose_transformer", feature_set="dynamics", history=90, time_features=True)[0]
        r["pose"][60:] = 999
        r["motion"][60:] = 999
        b = window_arrays([r], "pose_transformer", feature_set="dynamics", history=90, time_features=True)[0]
        np.testing.assert_array_equal(a, b)

    def test_padding_does_not_change_outputs(self):
        torch.manual_seed(17)
        x = torch.randn(2, 90, 10)
        x[:, :, -1] = 1
        lengths = torch.tensor([30, 60])
        z = x.clone()
        z[0, 30:] = 999
        z[1, 60:] = -999
        for kind in ("pose_lstm", "pose_transformer", "cnn2d"):
            m = Classifier(kind, 10).eval()
            with torch.no_grad():
                torch.testing.assert_close(m(x, lengths), m(z, lengths), atol=1e-6, rtol=1e-5)

    def test_masked_transformer_tokens_cannot_influence_output(self):
        x = torch.randn(1, 30, 10)
        x[:, :, -1] = 1
        x[:, 5, -1] = 0
        z = x.clone()
        z[:, 5, :-1] = 999
        m = Classifier("pose_transformer", 10).eval()
        with torch.no_grad():
            torch.testing.assert_close(m(x), m(z), atol=1e-6, rtol=1e-5)

    def test_focal_gamma_zero_equals_weighted_ce(self):
        logits = torch.tensor([[2.0, 1.0, 0.0], [0.3, 0.8, 1.0]], requires_grad=True)
        y = torch.tensor([0, 2])
        alpha = torch.tensor([0.5, 1.0, 2.0])
        torch.testing.assert_close(
            FocalLoss(alpha, 0)(logits, y), torch.nn.functional.cross_entropy(logits, y, weight=alpha)
        )

    def test_focal_loss_has_finite_gradient(self):
        z = torch.tensor([[100.0, -100.0, 0.0], [-100.0, 100.0, 0.0]], requires_grad=True)
        loss = FocalLoss([1, 2, 3])(z, torch.tensor([0, 2]))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(z.grad).all())

    def test_minority_weight_is_larger(self):
        w = class_weights([0] * 30 + [1] * 3 + [2] * 6)
        self.assertGreater(w[1], w[0])
        self.assertGreater(w[2], w[0])

    def test_dynamics_reset_on_missing_and_identity_change(self):
        n = 8
        m = np.zeros((n, 12), np.float32)
        m[:, 8] = np.arange(n)
        t = np.arange(n) / 15
        v = np.ones(n, bool)
        v[3] = False
        ids = np.zeros(n, int)
        ids[6:] = 1
        a = dynamics_features(m, t, v, ids)
        self.assertGreater(a[2, 0], 0)
        np.testing.assert_array_equal(a[3:5], 0)
        np.testing.assert_array_equal(a[6], 0)

    def test_large_timestamp_gap_resets_derivative(self):
        m = np.zeros((4, 12), np.float32)
        m[:, 8] = np.arange(4)
        a = dynamics_features(m, np.array([0, 0.1, 0.6, 0.7]), np.ones(4, bool), np.zeros(4, int))
        np.testing.assert_array_equal(a[2], 0)

    def test_all_three_losses_supported(self):
        for name in ("ce", "weighted_ce", "focal"):
            loss, _ = make_loss(name, [0, 0, 1, 2])
            self.assertTrue(torch.isfinite(loss(torch.zeros(4, 3), torch.tensor([0, 0, 1, 2]))))


if __name__ == "__main__":
    unittest.main()
