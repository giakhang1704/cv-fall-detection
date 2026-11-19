"""Build per-video feature caches from pose caches (pose features, RGB embedding, scene vector).

For every sampled observation the cache stores normalised keypoints, body geometry and motion,
the 8-D Canny-Hough scene vector and, optionally, a 576-D MobileNetV3 embedding of the person
crop, together with the 3-state label and the valid decision endpoints.
"""

import json
import math
from pathlib import Path

import cv2
import numpy as np

from ..common import digest, fingerprint, rel, without_code_hashes, write_json
from ..data import endpoints, labels_at, source_tables
from ..preprocessing import apply
from .features import body_features, scene_features


def crop_tensor(frame, box, variant, size=224, margin=0.15):
    """Crop the person box with a margin, apply the preprocessing variant, letterbox to a square
    image and normalise with ImageNet mean/std. Returns a CHW float32 array.
    """
    b = np.asarray(box, float)
    pad = (b[2:] - b[:2]) * margin
    a = np.maximum(0, np.floor(b[:2] - pad)).astype(int)
    z = np.minimum(frame.shape[1::-1], np.ceil(b[2:] + pad)).astype(int)
    if min(z - a) < 2:
        raise ValueError("Invalid crop")
    im = apply(frame[a[1] : z[1], a[0] : z[0]], variant)
    h, w = im.shape[:2]
    scale = size / max(h, w)
    im = cv2.resize(im, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
    canvas = np.full((size, size, 3), 114, np.uint8)
    hh, ww = im.shape[:2]
    y = (size - hh) // 2
    x = (size - ww) // 2
    canvas[y : y + hh, x : x + ww] = im
    rgb = np.ascontiguousarray(canvas[:, :, ::-1].transpose(2, 0, 1)).astype(np.float32) / 255
    return (rgb - np.array([0.485, 0.456, 0.406], np.float32)[:, None, None]) / np.array(
        [0.229, 0.224, 0.225], np.float32
    )[:, None, None]


def prepare(
    cfg,
    pose_dir,
    output,
    splits=("train", "val", "test"),
    rgb=False,
    rgb_preprocessing="none",
    allow_partial=False,
    limit=None,
):
    """Build the prepared dataset from pose caches.

    For every video: samples timestamps, labels, pose features, decision endpoints and, optionally,
    MobileNetV3 embeddings and the scene vector. Writes one .npz per video and an index.json.
    Existing outputs are reused only if all settings and inputs are identical.
    """
    videos, segments, hashes = source_tables(cfg)
    videos = videos[videos.split.isin(splits)]
    if limit:
        videos = videos.head(limit)
    pose_dir = Path(pose_dir)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    missing = [v.video_id for v in videos.itertuples() if not (pose_dir / (v.video_id + ".npz")).exists()]
    if missing and not allow_partial:
        raise ValueError(
            f"{len(missing)}/{len(videos)} pose caches missing. Wait for extraction. Examples: {missing[:3]}"
        )
    videos = videos[~videos.video_id.isin(missing)]
    encoder = None
    reports = []
    created = 0
    settings = dict(
        schema=1,
        manifests=hashes,
        rgb=rgb,
        rgb_preprocessing=rgb_preprocessing,
        window=cfg["window"],
        stride=cfg["stride"],
        max_gap=cfg["max_gap_seconds"],
        min_valid_fraction=cfg["min_valid_fraction"],
        image_size=cfg["image_size"],
        crop_margin=cfg["crop_margin"],
        feature_code=digest(Path(__file__).with_name("features.py")),
        prepare_code=digest(Path(__file__)),
        models_code=digest(Path(__file__).with_name("models.py")),
    )
    if not len(videos):
        raise ValueError("No completed pose caches in requested scope")
    first_meta = json.loads((pose_dir / (videos.iloc[0].video_id + ".json")).read_text(encoding="utf-8"))
    policy = dict(first_meta["signature"])
    policy.pop("extractor_sha256", None)
    settings["pose_policy"] = policy
    settings["pose_preprocessing"] = first_meta["signature"]["preprocessing"]
    if rgb:
        import torch

        from .models import ImageEncoder

        torch.set_num_threads(2)
        encoder = ImageEncoder()
        import hashlib

        encoder_hash = hashlib.sha256()
        for name, value in sorted(encoder.state_dict().items()):
            encoder_hash.update(name.encode())
            encoder_hash.update(value.cpu().numpy().tobytes())
        settings["encoder_sha256"] = encoder_hash.hexdigest()
        settings["encoder"] = (
            "torchvision MobileNetV3-small ImageNet pretrained, frozen features+global average pooling (576 features)"
        )
    for v in videos.itertuples():
        raw = pose_dir / (v.video_id + ".npz")
        meta = json.loads(raw.with_suffix(".json").read_text(encoding="utf-8"))
        current_policy = dict(meta["signature"])
        current_policy.pop("extractor_sha256", None)
        if current_policy != policy:
            raise ValueError("Mixed pose policies in prepared dataset")
        if meta["artifact_sha256"] != digest(raw) or meta["signature"]["manifests"] != hashes:
            raise ValueError("Pose cache integrity/manifest mismatch")
        source = Path(cfg["source_project"]) / v.video_path
        if digest(source) != meta["source_sha256"]:
            raise ValueError("Video content changed")
        signature = fingerprint(dict(settings=settings, pose_sha256=meta["artifact_sha256"]))
        target = out / (v.video_id + ".npz")
        sidecar = target.with_suffix(".json")
        if sidecar.exists():
            prior = json.loads(sidecar.read_text())
            # Source-code hashes are provenance only; every setting and input must still match.
            same = (
                without_code_hashes(prior["settings"]) == without_code_hashes(settings)
                and prior["pose_sha256"] == meta["artifact_sha256"]
            )
            if not same or prior["artifact_sha256"] != digest(target):
                raise ValueError(f"Prepared cache changed: {target}; choose a new output")
            reports.append(prior)
            continue
        with np.load(raw, allow_pickle=False) as d:
            times = d["times"]
            keys = d["keypoints"]
            boxes = d["bboxes"]
            valid = d["valid"]
            ids = d["track_ids"]
            width = int(d["width"])
            height = int(d["height"])
        keypoints, pose, motion = body_features(
            keys, boxes, times, valid, ids, width, height, cfg["keypoint_conf"]
        )
        scene = np.zeros((len(times), 8), np.float32)
        embedding = np.zeros((len(times), 576), np.float32) if rgb else np.empty((len(times), 0), np.float32)
        cap = cv2.VideoCapture(str(source))
        i = 0
        index = 0
        previous = -math.inf
        batch = []
        batch_indices = []
        preview_done = False

        def flush():
            """Encode the buffered person crops with MobileNetV3 in one batch."""
            if not batch:
                return
            import torch

            with torch.inference_mode():
                embedding[batch_indices] = encoder(torch.from_numpy(np.stack(batch))).numpy()
            batch.clear()
            batch_indices.clear()

        try:
            while i < len(times):
                ok, frame = cap.read()
                if not ok:
                    raise ValueError("Source ended before cached timestamps")
                pts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
                t = pts if np.isfinite(pts) and pts > previous else index / float(v.fps)
                previous = t
                index += 1
                if t < times[i] - 1e-6:
                    continue
                if abs(t - times[i]) > max(0.002, 0.51 / float(v.fps)):
                    raise ValueError("Pose and RGB timestamps do not align")
                if valid[i]:
                    scene[i], edges, lines, scale = scene_features(frame, boxes[i])
                    if rgb:
                        batch.append(
                            crop_tensor(
                                frame, boxes[i], rgb_preprocessing, cfg["image_size"], cfg["crop_margin"]
                            )
                        )
                        batch_indices.append(i)
                        if len(batch) >= 16:
                            flush()
                    if not preview_done:
                        display = cv2.resize(frame, (edges.shape[1], edges.shape[0]))
                        display[edges > 0] = [0, 180, 255]
                        for x1, y1, x2, y2 in lines:
                            cv2.line(display, (x1, y1), (x2, y2), (40, 230, 90), 1)
                        b = (boxes[i] * scale).astype(int)
                        cv2.rectangle(display, tuple(b[:2]), tuple(b[2:]), (255, 120, 40), 2)
                        path = out / "geometry_previews"
                        path.mkdir(exist_ok=True)
                        cv2.imwrite(str(path / (v.video_id + ".jpg")), display)
                        preview_done = True
                i += 1
            if rgb:
                flush()
        finally:
            cap.release()
        labels = labels_at(times, segments[segments.video_id == v.video_id])
        ends = endpoints(times, valid, ids, cfg)
        temporary = target.with_suffix(".partial.npz")
        np.savez_compressed(
            temporary,
            times=times,
            valid=valid,
            ids=ids,
            keypoints=keypoints,
            pose=pose,
            motion=motion,
            scene=scene,
            rgb=embedding,
            labels=labels,
            ends=ends,
        )
        temporary.replace(target)
        report = dict(
            signature=signature,
            artifact_sha256=digest(target),
            settings=settings,
            video_id=v.video_id,
            split=v.split,
            dataset=v.dataset,
            subject=v.global_subject,
            source_sha256=meta["source_sha256"],
            pose_sha256=meta["artifact_sha256"],
            windows=len(ends),
            duration=float(v.duration),
            window=int(cfg["window"]),
            rgb=rgb,
            intervals=segments[segments.video_id == v.video_id][["start", "end", "mapped_label"]].to_dict(
                "records"
            ),
            source_intervals=segments[segments.video_id == v.video_id][
                ["start", "end", "label", "mapped_label"]
            ].to_dict("records"),
        )
        write_json(sidecar, report)
        reports.append(report)
        print("Prepared", v.video_id, len(ends), flush=True)
        created += 1
    if created == 0 and (out / "index.json").exists():
        # Everything was reused: keep the existing index (checkpoints store its SHA-256).
        index = json.loads((out / "index.json").read_text(encoding="utf-8"))
        return dict(
            output=rel(out),
            videos=len(index["records"]),
            missing=len(index["missing"]),
            windows=sum(r["windows"] for r in index["records"]),
        )
    write_json(
        out / "index.json",
        dict(
            records=reports,
            missing=missing,
            partial=bool(missing) or limit is not None,
            settings=settings,
            limitation="Common pose-available crop protocol. RGB branches are not independent of the pose detector. Geometry describes image edges/lines, not calibrated floor.",
        ),
    )
    return dict(
        output=rel(out),
        videos=len(reports),
        missing=len(missing),
        windows=sum(r["windows"] for r in reports),
    )


def load_records(directory, split):
    """Load the prepared records of one split and check each file against its recorded SHA-256."""
    directory = Path(directory)
    index = json.loads((directory / "index.json").read_text())
    result = []
    for meta in index["records"]:
        if meta["split"] != split:
            continue
        path = directory / (meta["video_id"] + ".npz")
        if digest(path) != meta["artifact_sha256"]:
            raise ValueError("Prepared cache hash mismatch")
        with np.load(path, allow_pickle=False) as a:
            r = {k: a[k].copy() for k in a.files}
        r["meta"] = meta
        result.append(r)
    if not result:
        raise ValueError(f"No {split} records")
    return result, index


def window_arrays(
    records,
    kind,
    scene=False,
    feature_set="motion",
    training=False,
    cap=48,
    history=None,
    time_features=False,
    return_lengths=False,
):
    """Build fixed-length input windows (and labels) for a given model type.

    Selects the feature set (pose, RGB, optional scene vector), keeps an observation flag in the
    last channel and, when training, samples at most ``cap`` windows per video balanced across
    the three states.
    """
    xs = []
    ys = []
    refs = []
    lengths = []
    for j, r in enumerate(records):
        if kind in ("cnn2d", "cnn_lstm"):
            if r["rgb"].shape[1] == 0:
                raise ValueError("Run prepare --rgb before training CNN branches")
            base = r["rgb"]
        elif kind in ("rf", "svm"):
            if feature_set == "dynamics":
                from .features import dynamics_features

                base = np.c_[r["motion"], dynamics_features(r["motion"], r["times"], r["valid"], r["ids"])]
            else:
                base = r["keypoints"] if feature_set == "keypoints" else r["motion"]
        elif feature_set == "dynamics":
            from .features import dynamics_features

            extra = dynamics_features(r["motion"], r["times"], r["valid"], r["ids"])
            base = np.c_[r["pose"], extra]
        else:
            base = r["pose"] if feature_set == "motion" else r["keypoints"]
        if scene:
            base = np.c_[base, r["scene"]]
        w = r["meta"]["window"]
        ends = r["ends"]
        maximum = history or w
        if maximum < w:
            raise ValueError("History cannot be shorter than decision warmup window")
        if training:
            ends = ends[r["labels"][ends] >= 0]
            # Stratified deterministic cap per video; no overlapping-frame random split.
            if len(ends) > cap:
                groups = [ends[r["labels"][ends] == label] for label in (0, 1, 2)]
                chosen = []
                for g in groups:
                    if len(g):
                        chosen.extend(
                            g[np.linspace(0, len(g) - 1, min(len(g), max(1, cap // 3))).astype(int)]
                        )
                remaining = np.setdiff1d(ends, chosen)
                needed = min(cap - len(chosen), len(remaining))
                if needed:
                    chosen.extend(remaining[np.linspace(0, len(remaining) - 1, needed).astype(int)])
                ends = np.sort(chosen)
        for e in ends:
            anchor = int(e) - w + 1
            start = max(0, int(e) - maximum + 1)
            if maximum > w:
                identity = r["ids"][e]
                foreign = np.flatnonzero(
                    (r["ids"][start : e + 1] >= 0) & (r["ids"][start : e + 1] != identity)
                )
                if len(foreign):
                    start = max(start, min(anchor, start + int(foreign[-1]) + 1))
                own = np.flatnonzero(r["ids"][start : e + 1] == identity)
                if len(own):
                    start = max(start, min(anchor, start + int(own[0])))
            valid = r["valid"][start : e + 1]
            a = base[start : e + 1].copy()
            a[~valid] = 0
            if kind in ("rf", "svm"):
                a = a[valid]
                a = np.r_[a.mean(axis=0), a.std(axis=0), a.min(axis=0), a.max(axis=0), valid.mean()]
            else:
                if time_features and kind != "cnn2d":
                    ticks = r["times"][start : e + 1]
                    deltas = np.r_[0, np.diff(ticks)]
                    ages = ticks - ticks[-1]
                    a = np.c_[a, deltas, ages]
                    a[~valid] = 0
                a = np.c_[a, valid.astype(np.float32)]
                lengths.append(len(a))
                a = np.pad(a, ((0, maximum - len(a)), (0, 0)))
            xs.append(a)
            ys.append(r["labels"][e])
            refs.append((j, int(e)))
    if not xs:
        raise ValueError("No usable windows")
    result = (np.stack(xs).astype(np.float32), np.asarray(ys, np.int64), refs)
    return (
        (*result, np.asarray(lengths, np.int64)) if return_lengths and kind not in ("rf", "svm") else result
    )
