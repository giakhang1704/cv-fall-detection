"""Causal single-target pose extraction, with appearance association and ROI recovery.

One target per video is an explicit dataset assumption, never a multi-person tracker.
Missing observations remain missing. Labels are never read by the selector.
"""

import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULTS = dict(
    init_conf=0.45,
    init_hits=3,
    pending_gap_seconds=0.3,
    keypoint_conf=0.3,
    min_keypoints=6,
    min_core=2,
    retention_seconds=5.0,
    appearance_min=0.35,
    reacquire_min=0.6,
    ambiguity_margin=0.06,
    roi_scale=2.2,
    roi_interval_seconds=0.2,
)


@dataclass
class Candidate:
    """One person detection: box (x1, y1, x2, y2), 17 keypoints (x, y, conf), score and ReID embedding."""
    box: np.ndarray
    keypoints: np.ndarray
    confidence: float
    embedding: np.ndarray
    source: str = "full"


def unit(x):
    """Return ``x`` scaled to unit L2 norm."""
    x = np.asarray(x, dtype=np.float32)
    return x / max(float(np.linalg.norm(x)), 1e-8)


def iou(a, b):
    """Intersection over union of two boxes in (x1, y1, x2, y2) format."""
    intersection = np.maximum(0, np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2])).prod()
    return float(
        intersection
        / max(np.maximum(0, a[2:] - a[:2]).prod() + np.maximum(0, b[2:] - b[:2]).prod() - intersection, 1e-8)
    )


class TargetSelector:
    """Conservative identity association; no generated/interpolated keypoints."""

    def __init__(self, settings=None, seed=None):
        self.cfg = dict(DEFAULTS, **(settings or {}))
        unknown = set(self.cfg) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown target settings: {unknown}")
        if self.cfg["init_hits"] < 1 or self.cfg["retention_seconds"] <= 0:
            raise ValueError("Invalid target lifetime/confirmation")
        self.seed = seed
        self.anchor = self.appearance = self.box = self.last_seen = None
        self.velocity = np.zeros(2, dtype=np.float32)
        self.gallery = deque(maxlen=12)
        self.pending = None
        self.pending_hits = 0
        self.previous_time = None
        self.last_roi = -math.inf
        self.reason = "initializing"

    def pose_ok(self, candidate):
        """True if the detection is finite and has at least ``min_keypoints`` visible keypoints,
        including ``min_core`` of the shoulders/hips.
        """
        k = candidate.keypoints
        if (
            k.shape != (17, 3)
            or not np.isfinite(k).all()
            or not np.isfinite(candidate.box).all()
            or np.any(candidate.box[2:] <= candidate.box[:2])
            or not np.isfinite(candidate.confidence)
            or not np.isfinite(candidate.embedding).all()
            or np.linalg.norm(candidate.embedding) < 1e-6
        ):
            return False
        visible = k[:, 2] >= self.cfg["keypoint_conf"]
        return bool(
            visible.sum() >= self.cfg["min_keypoints"]
            and visible[[5, 6, 11, 12]].sum() >= self.cfg["min_core"]
        )

    def prediction(self, t):
        """Predict the target box at time ``t`` from its last box and smoothed velocity."""
        dt = min(max(0, t - self.last_seen), 0.5)
        delta = self.velocity * dt
        scale = max(float(np.linalg.norm(self.box[2:] - self.box[:2])), 1.0)
        delta *= min(1.0, 0.5 * scale / max(float(np.linalg.norm(delta)), 1e-8))
        return self.box + np.tile(delta, 2)

    def choose(self, candidates, t):
        """Pick the candidate matching the current target.

        Score = 0.4 * appearance + 0.4 * IoU + 0.2 * exp(-centre distance). Returns None if no
        candidate is reliable, if the best two are too close (ambiguous) or if the target expired.
        """
        if t - self.last_seen > self.cfg["retention_seconds"]:
            self.reason = "expired_requires_review"
            return None
        predicted = self.prediction(t)
        scale = max(float(np.linalg.norm(self.box[2:] - self.box[:2])), 1.0)
        dt = t - self.last_seen
        ranked = []
        for c in candidates:
            if not self.pose_ok(c):
                continue
            appearance = max(
                float(c.embedding @ self.anchor),
                float(c.embedding @ self.appearance),
                *[float(c.embedding @ g) for g in self.gallery],
            )
            distance = (
                float(np.linalg.norm((c.box[:2] + c.box[2:] - predicted[:2] - predicted[2:]) / 2)) / scale
            )
            overlap = max(iou(c.box, predicted), iou(c.box, self.box))
            reacquisition = dt > self.cfg["pending_gap_seconds"]
            minimum = self.cfg["reacquire_min"] if reacquisition else self.cfg["appearance_min"]
            if not reacquisition and overlap >= 0.5:
                minimum = min(minimum, 0.2)
            if appearance < minimum or (overlap < 0.03 and distance > min(2.0, 0.6 + dt)):
                continue
            score = 0.4 * appearance + 0.4 * overlap + 0.2 * math.exp(-distance)
            ranked.append((score, c))
        ranked.sort(key=lambda pair: pair[0], reverse=True)
        if not ranked:
            self.reason = "no_reliable_candidate"
            return None
        if len(ranked) > 1 and ranked[0][0] - ranked[1][0] < self.cfg["ambiguity_margin"]:
            self.reason = "ambiguous"
            return None
        return ranked[0][1]

    def bootstrap(self, candidates, t):
        """Initialise the target: a confident candidate must appear in ``init_hits`` consecutive
        observations before it is accepted.
        """
        if self.seed and t < self.seed.get("time", 0):
            self.reason = "before_seed"
            return None
        eligible = [c for c in candidates if self.pose_ok(c) and c.confidence >= self.cfg["init_conf"]]
        if self.seed:
            eligible = [c for c in eligible if iou(c.box, np.asarray(self.seed["box"])) >= 0.3]
        # Visible area is only a bootstrap ranking cue, never a rule rejecting lying bodies.
        eligible.sort(
            key=lambda c: c.confidence
            * (0.5 + 0.5 * float((c.keypoints[:, 2] >= self.cfg["keypoint_conf"]).mean()))
            * math.sqrt(float(np.prod(c.box[2:] - c.box[:2]))),
            reverse=True,
        )
        if not eligible:
            self.pending = None
            self.pending_hits = 0
            self.reason = "initializing"
            return None
        c = eligible[0]
        continuous = (
            self.pending is not None
            and t - self.pending[1] <= self.cfg["pending_gap_seconds"]
            and iou(c.box, self.pending[0].box) >= 0.1
            and c.embedding @ self.pending[0].embedding >= self.cfg["appearance_min"]
        )
        self.pending_hits = self.pending_hits + 1 if continuous else 1
        self.pending = (c, t)
        if self.pending_hits < self.cfg["init_hits"]:
            self.reason = "confirming_target"
            return None
        self.anchor = self.appearance = c.embedding.copy()
        self.gallery.append(c.embedding.copy())
        self.reason = "initialized"
        return c

    def update(self, candidates, t, rescue=None):
        """Process the detections of one observation and return the target candidate or None.

        If nothing matches, ``rescue`` (ROI recovery) is called on the predicted box. Missing
        observations stay missing; no pose is copied or interpolated.
        """
        if not np.isfinite(t) or (self.previous_time is not None and t <= self.previous_time):
            raise ValueError("Timestamps must strictly increase")
        self.previous_time = t
        c = self.bootstrap(candidates, t) if self.box is None else self.choose(candidates, t)
        if (
            c is None
            and self.box is not None
            and rescue is not None
            and t - self.last_seen <= self.cfg["retention_seconds"]
            and t - self.last_roi >= self.cfg["roi_interval_seconds"]
        ):
            self.last_roi = t
            c = self.choose(rescue(self.prediction(t)), t)
        if c is None:
            return None
        if self.box is not None:
            dt = t - self.last_seen
            v = ((c.box[:2] + c.box[2:]) - (self.box[:2] + self.box[2:])) / (2 * dt)
            scale = max(float(np.linalg.norm(self.box[2:] - self.box[:2])), 1.0)
            v *= min(1.0, 3 * scale / max(float(np.linalg.norm(v)), 1e-8))
            self.velocity = 0.5 * self.velocity + 0.5 * v
            # Update cautiously; retain immutable anchor to reduce appearance drift.
            if c.confidence >= self.cfg["init_conf"] and iou(c.box, self.box) >= 0.1:
                self.appearance = unit(0.8 * self.appearance + 0.2 * c.embedding)
                self.gallery.append(c.embedding.copy())
            self.reason = "observed_" + c.source
        self.box = c.box.copy()
        self.last_seen = t
        return c


class TargetPoseEngine:
    """YOLO26n-pose + ReID + TargetSelector: returns the target person's pose frame by frame."""
    def __init__(self, root, cfg):
        import torch
        from ultralytics.trackers.utils.reid import ReID

        from ultralytics import YOLO

        root = Path(root)
        self.cfg = cfg
        self.root = root
        torch.set_num_threads(cfg["threads"])
        self.model = YOLO(str(root / cfg["pose_weights"]))
        self.model.fuse(verbose=False)
        self.model.add_callback("on_predict_start", lambda p: torch.set_num_threads(cfg["threads"]))
        self.model.predict(
            np.zeros((480, 640, 3), np.uint8), imgsz=cfg["pose_imgsz"], conf=0.1, device="cpu", verbose=False
        )
        self.encoder = ReID(str(root / cfg["reid_weights"]), device="cpu")
        self.settings = dict(DEFAULTS, **cfg.get("target", {}))
        self.selector = None

    def reset(self, seed=None):
        """Start a new video with a fresh selector (optionally seeded with a known box)."""
        self.selector = TargetSelector(self.settings, seed)

    def detect(self, frame, source="full", offset=(0, 0)):
        """Run YOLO-pose on an image (or ROI) and return valid candidates with ReID embeddings."""
        r = self.model.predict(frame, imgsz=self.cfg["pose_imgsz"], conf=0.1, device="cpu", verbose=False)[0]
        if r.keypoints is None:
            return []
        boxes = r.boxes.xyxy.cpu().numpy()
        keys = r.keypoints.data.cpu().numpy()
        scores = r.boxes.conf.cpu().numpy()
        selected = []
        for i, (b, k, s) in enumerate(zip(boxes, keys, scores)):
            # Pose validity does not depend on embeddings. Delay the encoder until after this gate.
            dummy = Candidate(b, k, float(s), np.ones(1, np.float32))
            if self.selector.pose_ok(dummy):
                selected.append(i)
        if not selected:
            return []
        xywh = r.boxes.xywh.cpu().numpy()[selected]
        features = self.encoder(frame, xywh)
        result = []
        for i, feature in zip(selected, features):
            if feature is None:
                continue
            b = boxes[i].copy() + np.tile(offset, 2)
            k = keys[i].copy()
            k[:, :2] += offset
            result.append(Candidate(b, k, float(scores[i]), unit(feature), source))
        return result

    def update(self, frame, t):
        """Preprocess the frame, detect people, select the target (with ROI recovery) and return
        (candidate or None, diagnostics).
        """
        if self.selector is None:
            self.reset()
        from .preprocessing import apply, image_metrics

        frame = apply(frame, self.cfg["preprocessing"])
        metrics = image_metrics(frame)
        roi_calls = 0

        def rescue(box):
            """ROI recovery: re-run detection on an enlarged square crop around the predicted box."""
            nonlocal roi_calls
            h, w = frame.shape[:2]
            center = (box[:2] + box[2:]) / 2
            side = max(box[2:] - box[:2]) * self.settings["roi_scale"]
            a = np.maximum([0, 0], np.floor(center - side / 2)).astype(int)
            b = np.minimum([w, h], np.ceil(center + side / 2)).astype(int)
            if min(b - a) < 24 or np.prod(b - a) > 0.9 * w * h:
                return []
            roi_calls += 1
            return self.detect(frame[a[1] : b[1], a[0] : b[0]], "roi", a)

        candidate = self.selector.update(self.detect(frame), t, rescue)
        return candidate, dict(
            reason=self.selector.reason,
            roi_calls=roi_calls,
            image_metrics=metrics,
            logical_id=0 if candidate is not None else -1,
            since_seen=None if self.selector.last_seen is None else t - self.selector.last_seen,
        )
