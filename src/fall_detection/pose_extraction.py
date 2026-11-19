"""Extract observed target poses with masks, continuity segments and provenance."""

import csv
import json
import math
import time
from pathlib import Path

import numpy as np

from .common import ROOT, digest, fingerprint, rel, without_code_hashes, write_json
from .target_pose import DEFAULTS, TargetPoseEngine


def extract(
    cfg, output=None, split=None, video_id=None, limit=None, seed_file=None, engine=None, write_summary=True
):
    """Extract one target person's pose per sampled frame and write `<video_id>.npz` + `.json`.

    The NPZ holds timestamps, 17 keypoints, box, a validity mask and continuity-segment ids.
    Missing observations are kept as missing (valid=False); nothing is interpolated.
    """
    import cv2

    directory = Path(cfg["source_project"]) / "data/manifests"
    with (directory / "videos.csv").open(encoding="utf-8-sig", newline="") as f:
        videos = list(csv.DictReader(f))
    if split:
        videos = [v for v in videos if v["split"] == split]
    if video_id:
        videos = [v for v in videos if v["video_id"] == video_id]
        if not videos:
            raise ValueError("Video not found in selected split")
    if limit is not None:
        if limit < 1:
            raise ValueError("Limit must be positive")
        videos = videos[:limit]
    if not videos:
        raise ValueError("No videos selected")
    output = Path(output) if output else ROOT / "data/pose_cache"
    output.mkdir(parents=True, exist_ok=True)
    seeds = json.loads(Path(seed_file).read_text(encoding="utf-8-sig")) if seed_file else {}
    settings = dict(DEFAULTS, **cfg.get("target", {}))
    signature = dict(
        schema=1,
        pose_weights_sha256=digest(ROOT / cfg["pose_weights"]),
        reid_weights_sha256=digest(ROOT / cfg["reid_weights"]),
        selector_sha256=digest(ROOT / "src/fall_detection/target_pose.py"),
        extractor_sha256=digest(Path(__file__)),
        settings=settings,
        sample_hz=cfg["sample_hz"],
        pose_imgsz=cfg["pose_imgsz"],
        detector_conf=0.1,
        continuity_gap_seconds=cfg["max_gap_seconds"],
        preprocessing=cfg["preprocessing"],
        preprocessing_sha256=digest(ROOT / "src/fall_detection/preprocessing.py"),
        manifests={name: digest(directory / name) for name in ("videos.csv", "segments.csv")},
    )
    if not np.isfinite(cfg["sample_hz"]) or cfg["sample_hz"] <= 0:
        raise ValueError("Invalid sample rate")
    reports = []
    for v in videos:
        source = Path(cfg["source_project"]) / v["video_path"]
        source_hash = digest(source)
        metadata = dict(
            signature=signature,
            source_sha256=source_hash,
            video_id=v["video_id"],
            seed=seeds.get(v["video_id"]),
        )
        cache_signature = fingerprint(metadata)
        target = output / (v["video_id"] + ".npz")
        sidecar = target.with_suffix(".json")
        if target.exists() or sidecar.exists():
            if not target.exists() or not sidecar.exists():
                raise ValueError(f"Incomplete existing cache: {target}")
            existing = json.loads(sidecar.read_text())
            # Reuse a cache when every setting, weight and input matches; source-code hashes are
            # provenance only, so reformatting the code does not invalidate existing caches.
            same_inputs = all(existing[key] == metadata[key] for key in ("source_sha256", "video_id", "seed"))
            same_settings = without_code_hashes(existing["signature"]) == without_code_hashes(signature)
            if not (same_inputs and same_settings) or existing.get("artifact_sha256") != digest(target):
                raise ValueError(
                    f"Existing cache differs: {target}. Choose a fresh --output; old cache is preserved."
                )
            reports.append(existing["quality"])
            print("cached", v["video_id"], flush=True)
            continue
        if engine is None:
            engine = TargetPoseEngine(ROOT, cfg)
        engine.reset(seeds.get(v["video_id"]))
        started = time.perf_counter()
        image_stats = []
        cap = cv2.VideoCapture(str(source))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if not cap.isOpened() or not math.isfinite(fps) or fps <= 0:
            raise ValueError(f"Cannot read {source}")
        times = []
        keys = []
        boxes = []
        valid = []
        segments = []
        reasons = []
        roi = []
        conf = []
        frame_index = 0
        next_time = 0.0
        previous = -math.inf
        last_valid = None
        segment = -1
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                pts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
                t = pts if math.isfinite(pts) and pts > previous else frame_index / fps
                if t <= previous:
                    raise ValueError("Nonmonotonic source timestamps")
                previous = t
                frame_index += 1
                if t + 1e-8 < next_time:
                    continue
                next_time = (math.floor(t * cfg["sample_hz"] + 1e-8) + 1) / cfg["sample_hz"]
                c, detail = engine.update(frame, t)
                observed = c is not None
                image_stats.append(detail["image_metrics"])
                if observed:
                    if last_valid is None or t - last_valid > cfg["max_gap_seconds"] + 1e-8:
                        segment += 1
                    last_valid = t
                times.append(t)
                keys.append(c.keypoints if observed else np.zeros((17, 3), np.float32))
                boxes.append(c.box if observed else np.zeros(4, np.float32))
                valid.append(observed)
                segments.append(segment if observed else -1)
                reasons.append(detail["reason"])
                roi.append(observed and c.source == "roi")
                conf.append(c.confidence if observed else 0.0)
                height, width = frame.shape[:2]
        finally:
            cap.release()
        if not times:
            raise ValueError(f"Empty source {source}")
        temporary = target.with_suffix(".partial.npz")
        np.savez_compressed(
            temporary,
            times=np.asarray(times),
            keypoints=np.asarray(keys, np.float32),
            bboxes=np.asarray(boxes, np.float32),
            valid=np.asarray(valid, bool),
            person_ids=np.where(valid, 0, -1),
            track_ids=np.asarray(segments, np.int64),
            reasons=np.asarray(reasons),
            roi_recovered=np.asarray(roi, bool),
            confidence=np.asarray(conf, np.float32),
            width=width,
            height=height,
        )
        temporary.replace(target)
        from .data import endpoints

        usable_windows = len(endpoints(np.asarray(times), np.asarray(valid), np.asarray(segments), cfg))
        quality = dict(
            video_id=v["video_id"],
            observations=len(times),
            valid=int(sum(valid)),
            valid_fraction=float(np.mean(valid)),
            roi_recovered=int(sum(roi)),
            continuity_segments=segment + 1,
            usable_windows=usable_windows,
            requires_review=bool(
                not any(valid)
                or np.mean(valid) < cfg["min_valid_fraction"]
                or usable_windows == 0
                or any(r == "expired_requires_review" for r in reasons)
            ),
        )
        quality["runtime_seconds"] = time.perf_counter() - started
        quality["image_metrics_mean"] = {
            key: float(np.mean([row[key] for row in image_stats])) for key in image_stats[0]
        }
        metadata.update(
            cache_signature=cache_signature,
            artifact_sha256=digest(target),
            quality=quality,
            semantics="Observed poses only; no interpolation or freeze. valid marks observations; person_ids=logical target; track_ids=continuity segments. No labels used for selection.",
        )
        write_json(sidecar, metadata)
        reports.append(quality)
        print(json.dumps(quality), flush=True)
    if write_summary:
        write_json(
            output / "quality_summary.json",
            dict(
                videos=reports,
                selection=dict(split=split, video_id=video_id, limit=limit),
                limitation="Accepted pose fraction is not verified person recall or pose accuracy; manually review suspicious clips before training.",
            ),
        )
    return dict(
        output=rel(output),
        videos=len(reports),
        observations=sum(r["observations"] for r in reports),
        valid=sum(r["valid"] for r in reports),
        requires_review=sum(r["requires_review"] for r in reports),
    )
