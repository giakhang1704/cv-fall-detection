"""Dataset manifests, label lookup and the causal decision-endpoint schedule."""

import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from .common import ROOT, digest, write_json


def source_tables(cfg):
    """Load videos.csv and segments.csv and check them.

    Rejects duplicate videos, subjects appearing in more than one split, unknown labels and
    invalid intervals. Returns (videos, segments, manifest SHA-256 hashes).
    """
    source = Path(cfg["source_project"])
    videos = pd.read_csv(source / "data/manifests/videos.csv")
    segments = pd.read_csv(source / "data/manifests/segments.csv")
    if videos.video_id.duplicated().any() or videos.video_path.duplicated().any():
        raise ValueError("Duplicate video identifiers/paths")
    if (videos.groupby("global_subject").split.nunique() > 1).any():
        raise ValueError("Subject leakage")
    if not set(videos.split) <= {"train", "val", "test"}:
        raise ValueError("Invalid split")
    if not set(segments.video_id) <= set(videos.video_id):
        raise ValueError("Unknown annotation video")
    if not segments.mapped_label.isin([0, 1, 2]).all():
        raise ValueError("Invalid labels")
    if (segments.end <= segments.start).any() or not np.isfinite(segments[["start", "end"]]).all().all():
        raise ValueError("Invalid annotation intervals")
    hashes = {name: digest(source / "data/manifests" / name) for name in ("videos.csv", "segments.csv")}
    return videos, segments, hashes


def freeze_data(cfg):
    """Verify every video file exists and has unique content, snapshot the manifests and
    write an audit report (data/manifest_snapshot/audit.json).
    """
    videos, segments, hashes = source_tables(cfg)
    destination = ROOT / "data/manifest_snapshot"
    destination.mkdir(parents=True, exist_ok=True)
    for name, h in hashes.items():
        target = destination / name
        if target.exists() and digest(target) != h:
            raise ValueError("Manifest changed; create a new project data snapshot explicitly")
        if not target.exists():
            shutil.copyfile(Path(cfg["source_project"]) / "data/manifests" / name, target)
    sources = {}
    for v in videos.itertuples():
        path = Path(cfg["source_project"]) / v.video_path
        if not path.exists():
            raise FileNotFoundError(path)
        sha = digest(path)
        if sha in sources:
            raise ValueError(f"Duplicate content: {v.video_id} and {sources[sha]}")
        sources[sha] = v.video_id
    report = dict(
        videos=len(videos),
        datasets=videos.groupby("dataset").size().to_dict(),
        splits=videos.groupby("split").size().to_dict(),
        manifests=hashes,
        source_project=cfg["source_project"],
        video_content_hashes=sources,
        note="Subject-disjoint split and reviewed labels.",
    )
    write_json(destination / "audit.json", report)
    return report


def labels_at(times, segments):
    """Return the 3-state label (0 ADL, 1 Falling, 2 Fallen) at each timestamp.

    Timestamps covered by no interval or by overlapping intervals get -1 and are ignored.
    """
    y = np.full(len(times), -1, dtype=np.int64)
    count = np.zeros(len(times), int)
    for r in segments.itertuples():
        selected = (times >= r.start) & (times < r.end)
        y[selected] = r.mapped_label
        count[selected] += 1
    y[count != 1] = -1
    return y


def endpoints(times, valid, ids, cfg):
    """Return the observation indices where the classifier may make a decision.

    A window of ``cfg['window']`` observations ending at index ``end`` (every ``cfg['stride']``)
    is kept only if the last observation has a pose, enough observations have a pose, it
    contains a single target identity and no gap is longer than ``cfg['max_gap_seconds']``.
    """
    result = []
    w = cfg["window"]
    for end in range(w - 1, len(times), cfg["stride"]):
        start = end - w + 1
        seen = ids[start : end + 1]
        seen = seen[seen >= 0]
        good = np.flatnonzero(valid[start : end + 1]) + start
        if (
            not valid[end]
            or valid[start : end + 1].mean() < cfg["min_valid_fraction"]
            or len(np.unique(seen)) != 1
        ):
            continue
        if (
            np.diff(np.r_[times[start], times[good], times[end]]).max(initial=0)
            > cfg["max_gap_seconds"] + 1e-8
        ):
            continue
        result.append(end)
    return np.array(result, dtype=np.int64)
