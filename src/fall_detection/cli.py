"""Command-line interface of the project (entry point: ``python run.py COMMAND``).

Commands
--------
status        Summarise the dataset, pose caches and trained checkpoints.
audit         Verify video hashes and the subject split; freeze the manifests.
extract-pose  Run YOLO26n-pose + target selection and write one pose cache per video.
prepare       Build features (pose, optional RGB embedding, scene vector) from pose caches.
train         Train one classifier on a prepared dataset.
evaluate      Evaluate a checkpoint on the validation or test split.
compare       Tabulate and plot several evaluated runs.
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from .common import ROOT, config, digest
from .preprocessing import VARIANTS

MODELS = ["rf", "svm", "cnn2d", "cnn_lstm", "pose_lstm", "pose_transformer"]
DEFAULT_POSE_DIR = ROOT / "outputs/preprocessing_full/pose_cache/none"


def build_parser():
    """Build the argparse parser with one sub-command per pipeline stage."""
    parser = argparse.ArgumentParser(prog="run.py", description="Fall detection from video")
    parser.add_argument("--config", type=Path, help="JSON file overriding configs/default.json")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("status", help="Summarise data, pose caches and checkpoints")
    sub.add_parser("audit", help="Verify video hashes and subject split; freeze manifests")

    p = sub.add_parser("extract-pose", help="Extract the target person's pose for each video")
    p.add_argument("--output", type=Path, help="Cache directory (default: outputs/preprocessing_full/pose_cache/none)")
    p.add_argument("--split", choices=["train", "val", "test"])
    p.add_argument("--video-id")
    p.add_argument("--limit", type=int)
    p.add_argument("--seed-file", type=Path, help='JSON {"video_id": {"box": [x1,y1,x2,y2], "time": t}}')

    p = sub.add_parser("prepare", help="Build features from pose caches")
    p.add_argument("--pose-dir", type=Path, default=DEFAULT_POSE_DIR)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--split", nargs="+", choices=["train", "val", "test"], default=["train", "val", "test"])
    p.add_argument(
        "--rgb", action="store_true", help="Also compute MobileNetV3 embeddings of the person crop"
    )
    p.add_argument("--rgb-preprocessing", choices=list(VARIANTS), default="none")
    p.add_argument("--allow-partial", action="store_true")
    p.add_argument("--limit", type=int)

    p = sub.add_parser("train", help="Train one classifier")
    p.add_argument("--data", type=Path, required=True, help="Directory written by `prepare`")
    p.add_argument("--model", choices=MODELS, required=True)
    p.add_argument("--scene", action="store_true", help="Append the 8-D Canny-Hough scene vector")
    p.add_argument("--feature-set", choices=["keypoints", "motion", "dynamics"], default="motion")
    p.add_argument("--run", type=Path, required=True, help="New, empty output directory")
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--epochs", type=int)
    p.add_argument("--history", type=int, help="Observations per sequence (default: window)")
    p.add_argument("--loss", choices=["ce", "weighted_ce", "focal"], default="weighted_ce")
    p.add_argument("--focal-gamma", type=float, default=2.0)
    p.add_argument("--transformer-dim", type=int, default=32)
    p.add_argument("--heads", type=int, default=2)
    p.add_argument("--layers", type=int, default=1)

    p = sub.add_parser("evaluate", help="Evaluate a checkpoint")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--split", choices=["val", "test"], default="test")

    p = sub.add_parser("compare", help="Compare evaluated runs")
    p.add_argument("--runs", nargs="+", type=Path, required=True)
    p.add_argument("--split", choices=["val", "test"], default="test")
    p.add_argument("--output", type=Path, required=True)
    return parser


def status(cfg):
    """Summarise the manifests, pose caches and trained checkpoints without modifying anything."""
    directory = Path(cfg["source_project"]) / "data/manifests"
    with (directory / "videos.csv").open(encoding="utf-8-sig", newline="") as stream:
        videos = list(csv.DictReader(stream))
    with (directory / "segments.csv").open(encoding="utf-8-sig", newline="") as stream:
        segments = list(csv.DictReader(stream))
    return dict(
        videos=len(videos),
        datasets=dict(Counter(r["dataset"] for r in videos)),
        splits=dict(Counter(r["split"] for r in videos)),
        annotated_intervals=len(segments),
        manifest_sha256={name: digest(directory / name) for name in ("videos.csv", "segments.csv")},
        pose_caches=len(list(DEFAULT_POSE_DIR.glob("*.npz"))),
        pose_model=cfg["pose_weights"],
        checkpoints=sorted(p.parent.name for p in (ROOT / "outputs/runs").glob("*/best.*")),
    )


def main(argv=None):
    """Parse the command line, run the selected command and print its result as JSON."""
    args = build_parser().parse_args(argv)
    cfg = config(args.config)
    if args.command == "status":
        result = status(cfg)
    elif args.command == "audit":
        from .data import freeze_data

        result = freeze_data(cfg)
    elif args.command == "extract-pose":
        from .pose_extraction import extract

        result = extract(cfg, args.output or DEFAULT_POSE_DIR, args.split, args.video_id, args.limit, args.seed_file)
    elif args.command == "prepare":
        from .learning.prepare import prepare

        result = prepare(
            cfg,
            args.pose_dir,
            args.output,
            args.split,
            args.rgb,
            args.rgb_preprocessing,
            args.allow_partial,
            args.limit,
        )
    elif args.command == "train":
        from .learning.training import train

        result = train(
            cfg,
            args.data,
            args.model,
            args.run,
            args.scene,
            args.feature_set,
            args.epochs,
            args.seed,
            history=args.history,
            loss_name=args.loss,
            focal_gamma=args.focal_gamma,
            transformer_dim=args.transformer_dim,
            heads=args.heads,
            layers=args.layers,
        )
    elif args.command == "evaluate":
        from .learning.training import evaluate_checkpoint

        report = evaluate_checkpoint(args.checkpoint, args.split)
        result = {k: v for k, v in report.items() if k != "clips"}
    else:
        from .learning.plots import compare

        result = compare(args.runs, args.split, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
