"""Controlled temporal ablations; one factor at a time, same endpoints and split."""

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
    """Train Pose-LSTM / Pose-Transformer variants that change one temporal factor at a time."""
    p = argparse.ArgumentParser()
    p.add_argument("--factor", choices=["model", "features", "history", "loss"], default="model")
    p.add_argument("--data", type=Path, default=ROOT / "outputs/prepared/temporal_common")
    p.add_argument(
        "--models",
        nargs="+",
        choices=["pose_lstm", "pose_transformer"],
        default=["pose_lstm", "pose_transformer"],
    )
    p.add_argument("--history", type=int, default=30)
    p.add_argument("--feature-set", choices=["keypoints", "motion", "dynamics"], default="motion")
    p.add_argument("--loss", choices=["ce", "weighted_ce", "focal"], default="weighted_ce")
    p.add_argument("--seeds", nargs="+", type=int, default=[17])
    p.add_argument("--evaluate-test", action="store_true")
    args = p.parse_args()
    cfg = config()
    data = args.data
    prepare(cfg, ROOT / "outputs/preprocessing_full/pose_cache/none", data, rgb=False)
    features = ["keypoints", "motion", "dynamics"] if args.factor == "features" else [args.feature_set]
    histories = [30, 45, 90] if args.factor == "history" else [args.history]
    losses = ["ce", "weighted_ce", "focal"] if args.factor == "loss" else [args.loss]
    runs = []
    for model in args.models:
        for feature in features:
            for history in histories:
                for loss in losses:
                    for seed in args.seeds:
                        run = ROOT / "outputs/runs" / f"{model}_{feature}_h{history}_{loss}_s{seed}"
                        checkpoint = run / "best.pt"
                        if checkpoint.exists():
                            info = json.loads((run / "config.json").read_text())
                            if (
                                info["prepared_index_sha256"] != digest(data / "index.json")
                                or info["model"] != model
                                or info.get("history_window") != history
                                or info.get("loss_name") != loss
                                or info["feature_set"] != feature
                                or info["seed"] != seed
                                or info.get("scene", False)
                                or info.get("focal_gamma", 2.0) != 2.0
                                or info.get("transformer_dim", 32) != 32
                                or info.get("heads", 2) != 2
                                or info.get("layers", 1) != 1
                            ):
                                raise ValueError(f"Existing run differs: {run}")
                        else:
                            train(
                                cfg,
                                data,
                                model,
                                run,
                                feature_set=feature,
                                seed=seed,
                                history=history,
                                loss_name=loss,
                            )
                        if args.evaluate_test:
                            evaluate_checkpoint(checkpoint, "test")
                        runs.append(run)
    split = "test" if args.evaluate_test else "val"
    output = ROOT / "outputs/comparisons" / ("temporal_" + args.factor) / split
    rows = compare(runs, split, output)
    write_json(
        output / "protocol.json",
        dict(
            factor=args.factor,
            models=args.models,
            histories=histories,
            features=features,
            losses=losses,
            seeds=args.seeds,
            first_decision_observations=cfg["window"],
            test_used_for_selection=False,
            causal=True,
            clip_and_event_protocol="same as baseline",
            note="Keep non-varied factors fixed; do not choose settings from test.",
        ),
    )
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
