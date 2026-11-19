"""Live single-target demo using the exact checkpoint feature/scaler/alarm path."""

import argparse
import csv
import json
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fall_detection.common import rel
from fall_detection.learning.features import body_features, scene_features
from fall_detection.learning.metrics import alarm_onsets
from fall_detection.learning.models import Classifier
from fall_detection.learning.prepare import window_arrays
from fall_detection.learning.training import probabilities, scale_windows
from fall_detection.target_pose import TargetPoseEngine

LINKS = [
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
]


def main():
    """Run the full pipeline on a webcam or video file, display the result and save
    demo.mp4 and predictions.csv under outputs/demos/.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/pose_lstm_scene.pt")
    parser.add_argument("--check-camera", action="store_true")
    parser.add_argument(
        "--headless", action="store_true", help="Process a video file without opening a window"
    )
    args = parser.parse_args()
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera/video source")
    if args.check_camera:
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise RuntimeError("Camera opened but returned no image")
        print("Camera OK:", frame.shape)
        return
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    info = state["info"]
    cfg = dict(info["config"])
    if info["model"] not in ("pose_lstm", "pose_transformer"):
        raise ValueError("This demo supports pose checkpoints")
    model = Classifier(
        info["model"],
        state["input_dim"],
        cfg["hidden_size"],
        cfg["dropout"],
        info.get("transformer_dim", 32),
        info.get("heads", 2),
        info.get("layers", 1),
    )
    model.load_state_dict(state["state_dict"])
    model.eval()
    engine = TargetPoseEngine(ROOT, cfg)
    directory = ROOT / "outputs/demos" / time.strftime("camera_%Y%m%d_%H%M%S")
    directory.mkdir(parents=True)
    (directory / "config.json").write_text(
        json.dumps(
            dict(checkpoint=rel(args.checkpoint), threshold=state["threshold"], config=cfg),
            indent=2,
        ),
        encoding="utf-8",
    )
    history = deque(maxlen=info.get("history_window", 30))
    alarm_history = deque(maxlen=1800)
    start = time.perf_counter()
    next_time = 0.0
    last_valid = None
    segment = -1
    count = 0
    writer = None
    log = (directory / "predictions.csv").open("w", newline="", encoding="utf-8")
    csvout = csv.writer(log)
    csvout.writerow(
        ["seconds", "pose_valid", "state", "adl", "falling", "fallen", "alarm_score", "threshold"]
    )
    last_capture = None
    last_scores = None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            now = time.perf_counter() - start
            if args.headless:
                # A file is replayed on its own timeline instead of the wall clock.
                now = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
            if now < next_time:
                if not args.headless and cv2.waitKey(1) & 255 == ord("q"):
                    break
                continue
            next_time = now + 1 / cfg["sample_hz"]
            tick = time.perf_counter()
            if (
                engine.selector is not None
                and engine.selector.last_seen is not None
                and now - engine.selector.last_seen > engine.settings["retention_seconds"]
            ):
                engine.reset()
                history.clear()
                alarm_history.clear()
                last_valid = None
                segment = -1
                count = 0
            candidate, detail = engine.update(frame, now)
            valid = candidate is not None
            if valid:
                if last_valid is None or now - last_valid > cfg["max_gap_seconds"] + 1e-8:
                    segment += 1
                last_valid = now
            history.append(
                (
                    now,
                    candidate.keypoints.copy() if valid else np.zeros((17, 3), np.float32),
                    candidate.box.copy() if valid else np.zeros(4, np.float32),
                    valid,
                    segment if valid else -1,
                    (
                        scene_features(frame, candidate.box)[0]
                        if valid and info["scene"]
                        else np.zeros(8, np.float32)
                    ),
                )
            )
            rows = list(history)
            times = np.array([r[0] for r in rows])
            observed = np.array([r[3] for r in rows])
            ids = np.array([r[4] for r in rows])
            scores = None
            status = "WARMUP"
            alarm_score = np.nan
            w = cfg["window"]
            scheduled = count >= w - 1 and (count - (w - 1)) % cfg["stride"] == 0
            if scheduled and len(rows) >= w:
                good = np.flatnonzero(observed[-w:]) + len(rows) - w
                usable = (
                    valid
                    and observed[-w:].mean() >= cfg["min_valid_fraction"]
                    and len(np.unique(ids[-w:][ids[-w:] >= 0])) == 1
                    and np.diff(np.r_[times[-w], times[good], times[-1]]).max(initial=0)
                    <= cfg["max_gap_seconds"] + 1e-8
                )
                if usable:
                    keys = np.array([r[1] for r in rows])
                    boxes = np.array([r[2] for r in rows])
                    h, width = frame.shape[:2]
                    kp, pose, motion = body_features(
                        keys, boxes, times, observed, ids, width, h, cfg["keypoint_conf"]
                    )
                    record = dict(
                        times=times,
                        valid=observed,
                        ids=ids,
                        keypoints=kp,
                        pose=pose,
                        motion=motion,
                        scene=np.array([r[5] for r in rows]),
                        labels=np.zeros(len(rows), int),
                        ends=np.array([len(rows) - 1]),
                        meta=dict(window=w),
                    )
                    x, _, _, lengths = window_arrays(
                        [record],
                        info["model"],
                        info["scene"],
                        info["feature_set"],
                        history=info.get("history_window", w),
                        time_features=info.get("input_schema", 1) >= 2,
                        return_lengths=True,
                    )
                    x, _ = scale_windows(x, state["scaler"])
                    scores = probabilities(model, x, lengths)[0]
                    alarm_score = float(scores[1:].sum())
                else:
                    status = "POSE GAP / LOW QUALITY"
            if scheduled or not valid:
                alarm_history.append((now, alarm_score))
            # Between two decision ticks the last decision stays on screen.
            if scheduled and scores is not None:
                last_scores = scores
            if not valid:
                last_scores = None
            if scores is None and last_scores is not None:
                scores = last_scores
            _, states = alarm_onsets(
                [z[0] for z in alarm_history],
                [z[1] for z in alarm_history],
                state["threshold"],
                cfg["alarm_confirm_seconds"],
                cfg["alarm_release_seconds"],
                cfg["alarm_off_ratio"],
            )
            active = bool(states and states[-1])
            if active:
                status = "ALARM"
            elif scores is not None:
                status = "SUSPECT" if float(scores[1:].sum()) >= state["threshold"] else "NORMAL"
            elif len(rows) >= w and status == "WARMUP":
                status = "WAITING FOR DECISION"
            overlay = frame.copy()
            if valid:
                b = candidate.box.astype(int)
                color = (0, 0, 255) if active else (0, 210, 220)
                cv2.rectangle(overlay, tuple(b[:2]), tuple(b[2:]), color, 2)
                for a, b in LINKS:
                    if min(candidate.keypoints[a, 2], candidate.keypoints[b, 2]) >= cfg["keypoint_conf"]:
                        cv2.line(
                            overlay,
                            tuple(candidate.keypoints[a, :2].astype(int)),
                            tuple(candidate.keypoints[b, :2].astype(int)),
                            (0, 240, 255),
                            2,
                        )
            canvas = np.hstack([frame, overlay])
            canvas = cv2.resize(canvas, (1280, 480))
            canvas = np.vstack([canvas, np.zeros((170, 1280, 3), np.uint8)])
            color = (30, 30, 255) if active else (70, 210, 120)
            cv2.putText(
                canvas,
                f'{status} | history {len(rows)}/{w} | alarm threshold {state["threshold"]:.2f}',
                (20, 510),
                0,
                0.7,
                color,
                2,
            )
            for j, label in enumerate(["ADL", "FALLING", "FALLEN"]):
                x0 = 20 + j * 420
                value = float(scores[j]) if scores is not None else 0.0
                cv2.putText(
                    canvas,
                    f"{label}: {100*value:.1f}%" if scores is not None else f"{label}: waiting",
                    (x0, 550),
                    0,
                    0.65,
                    (240, 240, 240),
                    1,
                )
                cv2.rectangle(canvas, (x0, 565), (x0 + 390, 580), (65, 65, 65), -1)
                cv2.rectangle(canvas, (x0, 565), (x0 + int(390 * value), 580), color, -1)
            hz = 0 if last_capture is None else 1 / max(now - last_capture, 1e-6)
            last_capture = now
            cv2.putText(
                canvas,
                f"Observed rate {hz:.1f} Hz / target 15 | q: quit | r: reset | processing {time.perf_counter()-tick:.2f}s",
                (20, 620),
                0,
                0.6,
                (210, 210, 210),
                1,
            )
            if writer is None:
                writer = cv2.VideoWriter(
                    str(directory / "demo.mp4"),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    cfg["sample_hz"],
                    (1280, 650),
                )
            writer.write(canvas)
            csvout.writerow(
                [
                    now,
                    valid,
                    status,
                    *(scores.tolist() if scores is not None else [""] * 3),
                    alarm_score,
                    state["threshold"],
                ]
            )
            log.flush()
            if args.headless:
                key = -1
            else:
                cv2.imshow("Fall detection - original / tracking", canvas)
                key = cv2.waitKey(1) & 255
            if key == ord("q"):
                break
            if key == ord("r"):
                engine.reset()
                history.clear()
                alarm_history.clear()
                last_valid = None
                segment = -1
                count = -1
                last_scores = None
            count += 1
    finally:
        cap.release()
        log.close()
        if writer is not None:
            writer.release()
        if not args.headless:
            cv2.destroyAllWindows()
        print("Saved:", rel(directory))


if __name__ == "__main__":
    main()
