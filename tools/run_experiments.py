"""Run explicitly selected experiment phases after required pose caches exist."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fall_detection.common import config, digest, write_json
from fall_detection.learning.plots import compare
from fall_detection.learning.prepare import prepare
from fall_detection.learning.training import evaluate_checkpoint, train


def main():
    """Prepare features if needed, then train and evaluate every configuration of the selected phase."""
    p = argparse.ArgumentParser()
    p.add_argument(
        "--phase",
        choices=["baselines", "scene", "pose_preprocessing", "rgb_preprocessing", "pose_features"],
        default="baselines",
    )
    p.add_argument("--seeds", nargs="+", type=int, default=[17])
    p.add_argument("--evaluate-test", action="store_true")
    args = p.parse_args()
    cfg = config()
    pose_root = ROOT / "outputs/preprocessing_full/pose_cache"
    cache_root = ROOT / "outputs/prepared"
    run_root = ROOT / "outputs/runs"
    jobs = []
    if args.phase in ("baselines", "scene"):
        data = cache_root / "original_rgb"
        prepare(cfg, pose_root / "none", data, rgb=True)
        for model in ("rf", "cnn2d", "cnn_lstm", "pose_lstm"):
            jobs.append(
                (
                    data,
                    model,
                    args.phase == "scene",
                    "motion",
                    model + ("_scene" if args.phase == "scene" else ""),
                )
            )
    elif args.phase == "pose_features":
        data = cache_root / "original_rgb"
        prepare(cfg, pose_root / "none", data, rgb=True)
        for features in ("keypoints", "motion"):
            jobs.append((data, "pose_lstm", False, features, "pose_lstm_" + features))
    else:
        for variant in ("none", "gaussian", "hist_equalization", "clahe", "gamma_bright"):
            rgb = args.phase == "rgb_preprocessing"
            data = cache_root / (("rgb_" if rgb else "pose_") + variant)
            prepare(
                cfg,
                pose_root / ("none" if rgb else variant),
                data,
                rgb=rgb,
                rgb_preprocessing=variant if rgb else "none",
            )
            model = "cnn2d" if rgb else "pose_lstm"
            jobs.append((data, model, False, "motion", model + "_" + variant))
    runs = []
    for data, model, scene, features, name in jobs:
        for seed in args.seeds:
            run = run_root / f"{name}_s{seed}"
            checkpoint = run / ("best.joblib" if model == "rf" else "best.pt")
            if checkpoint.exists():
                previous = json.loads((run / "config.json").read_text())
                if (
                    previous["prepared_index_sha256"] != digest(data / "index.json")
                    or previous["model"] != model
                    or previous["scene"] != scene
                    or previous["seed"] != seed
                    or previous["feature_set"] != features
                ):
                    raise ValueError(f"Existing run differs: {run}")
            else:
                train(cfg, data, model, run, scene, features, seed=seed)
            if args.evaluate_test:
                evaluate_checkpoint(checkpoint, "test")
            runs.append(run)
    split = "test" if args.evaluate_test else "val"
    rows = compare(runs, split, ROOT / "outputs/comparisons" / args.phase / split)
    write_json(
        ROOT / "outputs/comparisons" / args.phase / "protocol.json",
        dict(
            phase=args.phase,
            seeds=args.seeds,
            split=split,
            test_used_for_selection=False,
            runs=[str(r) for r in runs],
            note="Choose configurations on validation before explicitly requesting test evaluation. Do not tune from test. Earlier project test was already observed.",
        ),
    )
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
