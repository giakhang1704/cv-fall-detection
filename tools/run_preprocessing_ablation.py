"""Full-dataset, resumable preprocessing/pose ablation. Does not select on test."""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fall_detection.common import config, digest, write_json
from fall_detection.pose_extraction import extract
from fall_detection.preprocessing import VARIANTS
from fall_detection.target_pose import TargetPoseEngine


def main():
    """Extract pose caches for every video under each preprocessing variant.

    Resumable; a lock prevents two workers from writing the same output directory.
    """
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=ROOT / "outputs/preprocessing_full")
    p.add_argument(
        "--variants",
        nargs="+",
        choices=list(VARIANTS),
        default=["none", "gaussian", "hist_equalization", "clahe", "gamma_bright"],
    )
    p.add_argument("--limit", type=int)
    p.add_argument("--split", choices=["train", "val", "test"])
    args = p.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / "worker.lock").open("a+b")
    lock.seek(0)
    lock.write(b"0")
    lock.flush()
    lock.seek(0)
    try:
        if sys.platform == "win32":
            import msvcrt

            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as error:
        raise RuntimeError("Another worker holds this output directory") from error
    cfg = config()
    start = time.perf_counter()
    with (Path(cfg["source_project"]) / "data/manifests/videos.csv").open(
        encoding="utf-8-sig", newline=""
    ) as f:
        videos = list(csv.DictReader(f))
    if args.split:
        videos = [v for v in videos if v["split"] == args.split]
    if args.limit:
        videos = videos[: args.limit]
    code = {str(p.relative_to(ROOT)): digest(p) for p in (ROOT / "src/fall_detection").glob("*.py")}
    plan = dict(
        variants={v: VARIANTS[v] for v in args.variants},
        videos=[v["video_id"] for v in videos],
        split=args.split,
        sample_hz=cfg["sample_hz"],
        code=code,
        manifests={
            n: digest(Path(cfg["source_project"]) / "data/manifests" / n)
            for n in ("videos.csv", "segments.csv")
        },
    )
    plan_path = out / "plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text(encoding="utf-8")) != plan:
        raise ValueError("Plan or source code changed: choose a fresh output directory")
    write_json(plan_path, plan)
    total = len(videos) * len(args.variants)
    done = failed = 0
    engine = None
    fields = [
        "video_id",
        "dataset",
        "split",
        "variant",
        "observations",
        "valid",
        "valid_fraction",
        "roi_recovered",
        "continuity_segments",
        "usable_windows",
        "requires_review",
        "runtime_seconds",
        "brightness",
        "contrast",
        "clipped_fraction",
        "laplacian_variance",
    ]
    # Rebuild the small summary each run; validated NPZ caches are retained.
    with (out / "summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        stream.flush()
        for video in videos:
            for variant in args.variants:
                if (out / "STOP").exists():
                    write_json(
                        out / "progress.json",
                        dict(status="stopped", completed=done, failed=failed, total=total, pid=os.getpid()),
                    )
                    return
                if any(digest(ROOT / name) != sha for name, sha in code.items()):
                    raise RuntimeError("Source code changed while running; preserving completed caches")
                active = dict(cfg, preprocessing=variant)
                directory = out / "pose_cache" / variant
                write_json(
                    out / "progress.json",
                    dict(
                        status="running",
                        completed=done,
                        failed=failed,
                        total=total,
                        video_id=video["video_id"],
                        variant=variant,
                        elapsed_seconds=time.perf_counter() - start,
                        pid=os.getpid(),
                    ),
                )
                try:
                    # Load models once for the full job; reset target at every video/variant.
                    if engine is None:
                        engine = TargetPoseEngine(ROOT, active)
                    engine.cfg = active
                    extract(
                        active,
                        output=directory,
                        video_id=video["video_id"],
                        engine=engine,
                        write_summary=False,
                    )
                    q = json.loads((directory / (video["video_id"] + ".json")).read_text(encoding="utf-8"))[
                        "quality"
                    ]
                    row = {k: q[k] for k in fields if k in q}
                    row.update(dataset=video["dataset"], split=video["split"], variant=variant)
                    row.update(q["image_metrics_mean"])
                    writer.writerow(row)
                    stream.flush()
                    done += 1
                    print(f'Completed {done}/{total} | {video["video_id"]} | {variant}', flush=True)
                except Exception as error:
                    failed += 1
                    write_json(
                        out / "errors" / f'{video["video_id"]}_{variant}.json',
                        dict(error=repr(error), video_id=video["video_id"], variant=variant),
                    )
                    print("ERROR", video["video_id"], variant, repr(error), flush=True)
    write_json(
        out / "progress.json",
        dict(
            status="complete" if not failed else "complete_with_errors",
            completed=done,
            failed=failed,
            total=total,
            elapsed_seconds=time.perf_counter() - start,
            pid=os.getpid(),
        ),
    )


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
