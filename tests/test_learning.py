"""Unit tests for features, windows, alarm confirmation and clip/event evaluation."""

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fall_detection.learning.features import body_features, scene_features
from fall_detection.learning.metrics import alarm_onsets, episodes, evaluate
from fall_detection.learning.models import Classifier
from fall_detection.learning.prepare import crop_tensor
from fall_detection.learning.training import scale_windows


class LearningTests(unittest.TestCase):
    """Tests of the learning pipeline on small synthetic inputs."""
    def test_dataset_reports_preserve_counts_threshold_and_missing_clips(self):
        from fall_detection.learning.metrics import evaluate_by_dataset

        records = []
        for i, name in enumerate(("A", "B", "A")):
            records.append(
                dict(
                    times=np.arange(10) / 10,
                    labels=np.ones(10, int),
                    valid=np.ones(10, bool),
                    meta=dict(
                        dataset=name,
                        window=1,
                        video_id=str(i),
                        intervals=[dict(start=0, end=1, mapped_label=1)],
                    ),
                )
            )
        refs = [(j, e) for j in (0, 1) for e in range(10)]
        probs = np.tile([0.01, 0.98, 0.01], (20, 1))
        cfg = dict(stride=1, alarm_confirm_seconds=0.2, alarm_release_seconds=1.0, alarm_off_ratio=0.7)
        full = evaluate(records, probs, refs, 0.73, cfg)
        groups = evaluate_by_dataset(records, probs, refs, 0.73, cfg)
        self.assertEqual(groups["A"]["clip"]["count"], 2)
        self.assertEqual(groups["A"]["event"]["fn"], 1)
        for group in groups.values():
            self.assertEqual(group["threshold"], 0.73)
        np.testing.assert_array_equal(
            np.sum([g["clip"]["confusion_matrix"] for g in groups.values()], axis=0),
            full["clip"]["confusion_matrix"],
        )
        for key in ("tp", "fp", "fn"):
            self.assertEqual(sum(g["event"][key] for g in groups.values()), full["event"][key])

    def test_mask_does_not_create_background_rectangle_edges(self):
        image = np.zeros((120, 160, 3), np.uint8)
        image[30:90, 50:100] = 255
        f, edges, lines, _ = scene_features(image, [48, 28, 102, 92])
        self.assertEqual(np.count_nonzero(edges), 0)
        self.assertEqual(len(lines), 0)
        self.assertEqual(len(f), 8)

    def test_real_line_output_shape_is_supported(self):
        import cv2

        image = np.zeros((120, 160, 3), np.uint8)
        cv2.line(image, (5, 20), (155, 20), (255, 255, 255), 3)
        f, edges, lines, _ = scene_features(image, [60, 60, 90, 110])
        self.assertEqual(lines.shape[1], 4)
        self.assertGreater(len(lines), 0)
        self.assertTrue(np.isfinite(f).all())

    def test_no_visible_background_has_missing_scene_flag(self):
        f, _, _, _ = scene_features(np.zeros((100, 100, 3), np.uint8), [0, 0, 100, 100])
        np.testing.assert_array_equal(f, np.zeros(8))

    def test_pose_normalization_translation_invariant(self):
        k = np.zeros((2, 17, 3), np.float32)
        k[:, :, :2] = np.arange(34).reshape(17, 2) + 30
        k[:, :, 2] = 1
        b = np.array([[10, 10, 90, 110]] * 2, np.float32)
        t = np.array([0, 0.1])
        v = np.ones(2, bool)
        ids = np.zeros(2, int)
        a, _, _ = body_features(k, b, t, v, ids, 200, 200)
        k[:, :, :2] += 20
        b += 20
        z, _, _ = body_features(k, b, t, v, ids, 200, 200)
        np.testing.assert_allclose(a, z)

    def test_crop_is_bounded_fixed_shape(self):
        image = np.full((100, 160, 3), 90, np.uint8)
        x = crop_tensor(image, [-10, -10, 60, 80], "gamma_bright")
        self.assertEqual(x.shape, (3, 224, 224))
        self.assertTrue(np.isfinite(x).all())

    def test_missing_cancels_alarm_confirmation(self):
        onsets, _ = alarm_onsets([0, 0.1, 0.2], [0.99, np.nan, 0.99], 0.7, confirm=0.2)
        self.assertEqual(onsets, [])

    def test_contiguous_falling_and_fallen_form_one_event(self):
        self.assertEqual(
            episodes([dict(start=1, end=2, mapped_label=1), dict(start=2, end=4, mapped_label=2)]),
            [[1.0, 4.0]],
        )

    def test_event_delay_and_clip_decision(self):
        t = np.arange(10) / 10
        r = dict(
            times=t,
            labels=np.where(t < 0.2, 0, 1),
            valid=np.ones(10, bool),
            meta=dict(
                window=1,
                video_id="test",
                intervals=[dict(start=0, end=0.2, mapped_label=0), dict(start=0.2, end=1, mapped_label=1)],
            ),
        )
        p = np.tile([0.01, 0.98, 0.01], (10, 1))
        p[:2] = [0.98, 0.01, 0.01]
        c = dict(stride=1, alarm_confirm_seconds=0.2, alarm_release_seconds=1.0, alarm_off_ratio=0.7)
        report = evaluate([r], p, [(0, i) for i in range(10)], 0.7, c)
        self.assertEqual(report["event"]["tp"], 1)
        self.assertAlmostEqual(report["event"]["mean_delay_seconds"], 0.2)
        self.assertEqual(report["clip"]["confusion_matrix"], [[0, 0], [0, 1]])

    def test_tiny_unannotated_gap_merges_but_adl_does_not(self):
        z = [dict(start=1, end=2, mapped_label=1), dict(start=2.01, end=4, mapped_label=2)]
        self.assertEqual(len(episodes(z, 0.067)), 1)
        z.append(dict(start=2, end=2.01, mapped_label=0))
        self.assertEqual(len(episodes(z, 0.067)), 2)

    def test_alarm_in_overlapping_labels_is_unknown_event(self):
        t = np.arange(10) / 10
        r = dict(
            times=t,
            labels=np.ones(10, int),
            valid=np.ones(10, bool),
            meta=dict(
                window=1,
                video_id="test",
                intervals=[dict(start=0, end=1, mapped_label=1), dict(start=0, end=1, mapped_label=0)],
            ),
        )
        cfg = dict(stride=1, alarm_confirm_seconds=0.2, alarm_release_seconds=1.0, alarm_off_ratio=0.7)
        q = evaluate([r], np.tile([0.01, 0.98, 0.01], (10, 1)), [(0, i) for i in range(10)], 0.7, cfg)
        self.assertEqual(q["event"]["unknown_alarms"], 1)
        self.assertEqual(q["event"]["tp"], 0)
        self.assertEqual(q["clip"]["recall"], 1.0)

    def test_lying_subgroup_reports_false_alarm_and_coverage(self):
        t = np.arange(10) / 10
        r = dict(
            times=t,
            labels=np.zeros(10, int),
            valid=np.ones(10, bool),
            meta=dict(
                window=1,
                video_id="lying",
                intervals=[dict(start=0, end=1, mapped_label=0)],
                source_intervals=[dict(start=0, end=1, label=6, mapped_label=0)],
            ),
        )
        cfg = dict(stride=1, alarm_confirm_seconds=0.2, alarm_release_seconds=1.0, alarm_off_ratio=0.7)
        q = evaluate([r], np.tile([0.01, 0.98, 0.01], (10, 1)), [(0, i) for i in range(10)], 0.7, cfg)
        self.assertEqual(q["hard_negative_lying"]["false_positive_clips"], 1)
        self.assertEqual(q["hard_negative_lying"]["mean_coverage"], 1.0)

    def test_no_prediction_clip_counts_as_missed_fall(self):
        r = dict(
            times=np.arange(10) / 10,
            labels=np.ones(10, int),
            valid=np.zeros(10, bool),
            meta=dict(window=1, video_id="test", intervals=[dict(start=0, end=1, mapped_label=1)]),
        )
        c = dict(stride=1, alarm_confirm_seconds=0.2, alarm_release_seconds=1.0, alarm_off_ratio=0.7)
        q = evaluate([r], np.empty((0, 3)), [], 0.7, c)
        self.assertEqual(q["event"]["fn"], 1)
        self.assertEqual(q["clip"]["confusion_matrix"], [[0, 0], [1, 0]])
        self.assertEqual(q["prediction_coverage"], 0)

    def test_scaler_excludes_missing_and_preserves_mask(self):
        x = np.array([[[2, 4, 1], [0, 0, 0], [4, 8, 1]]], np.float32)
        z, s = scale_windows(x)
        np.testing.assert_allclose(s.mean_, [3, 6])
        np.testing.assert_array_equal(z[0, 1], [0, 0, 0])
        np.testing.assert_array_equal(z[:, :, -1], x[:, :, -1])

    def test_neural_head_shapes(self):
        for kind in ("cnn2d", "cnn_lstm", "pose_lstm"):
            m = Classifier(kind, 9)
            self.assertEqual(tuple(m(torch.zeros(2, 30, 9)).shape), (2, 3))


if __name__ == "__main__":
    unittest.main()
