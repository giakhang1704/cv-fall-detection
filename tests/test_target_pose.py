"""Behavioral checks for identity safety and missing-data handling."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fall_detection.target_pose import Candidate, TargetSelector


def candidate(box=(10, 10, 40, 100), confidence=0.9, embedding=(1.0, 0.0)):
    """Build a synthetic Candidate for selector tests."""
    keys = np.zeros((17, 3), np.float32)
    keys[:, :2] = [20, 40]
    keys[:, 2] = 0.9
    return Candidate(np.array(box, dtype=np.float32), keys, confidence, np.array(embedding, dtype=np.float32))


class TargetTests(unittest.TestCase):
    """Behaviour of the single-target selector on synthetic detections."""
    def ready(self):
        """Return a selector that has already confirmed its target."""
        s = TargetSelector()
        self.assertIsNone(s.update([candidate()], 0.0))
        self.assertIsNone(s.update([candidate()], 0.1))
        self.assertIsNotNone(s.update([candidate()], 0.2))
        return s

    def test_requires_temporal_confirmation(self):
        self.ready()

    def test_weak_observation_continues_but_cannot_initialize(self):
        s = TargetSelector()
        self.assertIsNone(s.update([candidate(confidence=0.2)], 0.0))
        s = self.ready()
        self.assertIsNotNone(s.update([candidate(confidence=0.2)], 0.3))

    def test_wrong_appearance_and_far_object_rejected(self):
        s = self.ready()
        self.assertIsNone(s.update([candidate(box=(200, 200, 230, 290), embedding=(0.0, 1.0))], 0.3))
        self.assertEqual(s.last_seen, 0.2)

    def test_short_occlusion_keeps_identity_without_fabricating_pose(self):
        s = self.ready()
        self.assertIsNone(s.update([], 0.3))
        self.assertIsNotNone(s.update([candidate()], 0.8))
        self.assertEqual(s.last_seen, 0.8)

    def test_long_occlusion_requires_review_not_silent_reassignment(self):
        s = self.ready()
        self.assertIsNone(s.update([candidate()], 6.0))
        self.assertEqual(s.reason, "expired_requires_review")

    def test_ambiguous_matches_abstain(self):
        s = self.ready()
        self.assertIsNone(s.update([candidate(), candidate()], 0.3))
        self.assertEqual(s.reason, "ambiguous")

    def test_roi_recovery_and_rate_limit(self):
        s = self.ready()
        calls = []

        def rescue(box):
            """Fake ROI recovery that counts how often it is called."""
            calls.append(box)
            return []

        self.assertIsNone(s.update([], 0.3, rescue))
        self.assertIsNone(s.update([], 0.4, rescue))
        self.assertEqual(len(calls), 1)
        c = candidate()
        c.source = "roi"
        self.assertIs(s.update([], 0.6, lambda box: [c]), c)

    def test_invalid_pose_rejected(self):
        s = self.ready()
        c = candidate()
        c.keypoints[5:, 2] = 0
        self.assertIsNone(s.update([c], 0.3))

    def test_disappearance_preserves_identity_but_returns_no_pose(self):
        s = self.ready()
        self.assertIsNone(s.update([], 0.3))
        self.assertEqual(s.last_seen, 0.2)
        self.assertIsNotNone(s.update([candidate()], 0.6))

    def test_time_cannot_go_backwards(self):
        s = self.ready()
        with self.assertRaises(ValueError):
            s.update([], 0.1)


if __name__ == "__main__":
    unittest.main()
