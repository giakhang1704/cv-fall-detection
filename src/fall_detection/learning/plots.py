"""Report figures: clip confusion matrix, losses, validation metrics and LR."""

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 130,
        "savefig.dpi": 220,
    }
)


def render(run, report, prefix="val"):
    """Save the confusion matrix and training curves (loss, validation metrics, learning rate) of a run."""
    run = Path(run)
    out = run / "figures"
    out.mkdir(exist_ok=True)

    def save(fig, path):
        """Save one figure, forcing integer epoch ticks and marking synthetic smoke tests."""
        from matplotlib.ticker import MaxNLocator

        for axis in fig.axes:
            if axis.get_xlabel() == "Epoch":
                axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        if report.get("synthetic", False):
            fig.suptitle("SYNTHETIC SMOKE TEST - NOT EXPERIMENTAL RESULTS", fontsize=8, color="crimson")
            fig.tight_layout(rect=(0, 0, 1, 0.94))
        fig.savefig(path)

    cm = np.asarray(report["clip"]["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    im = ax.imshow(cm, cmap="Blues", vmin=0)
    for i in range(2):
        for j in range(2):
            pct = 100 * cm[i, j] / max(cm[i].sum(), 1)
            ax.text(
                j,
                i,
                f"{cm[i,j]}\n{pct:.1f}%",
                ha="center",
                va="center",
                color="white" if cm[i, j] > cm.max() * 0.55 else "#17324d",
                fontsize=14,
            )
    ax.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["ADL", "Fall"],
        yticklabels=["ADL", "Fall"],
        xlabel="Predicted clip",
        ylabel="Ground-truth clip",
        title=f"{prefix.upper()} · clip confusion matrix",
    )
    fig.colorbar(im, ax=ax, label="Number of clips")
    fig.tight_layout()
    save(fig, out / f"{prefix}_confusion_matrix.png")
    plt.close(fig)
    path = run / "history.json"
    if not path.exists():
        return
    history = json.loads(path.read_text())
    epochs = [r["epoch"] for r in history]
    if "train_loss" in history[0]:
        fig, ax = plt.subplots(figsize=(6.5, 3.8))
        ax.plot(epochs, [r["train_loss"] for r in history], label="Train", color="#2463ae")
        ax.plot(epochs, [r["val_loss"] for r in history], label="Validation", color="#df7d16")
        ax.set(xlabel="Epoch", ylabel="Selected training objective", title="Learning curves")
        ax.legend(frameon=False)
        ax.grid(alpha=0.2)
        fig.tight_layout()
        save(fig, out / "loss.png")
        plt.close(fig)
        if "val_unweighted_ce" in history[0]:
            fig, ax = plt.subplots(figsize=(6.5, 3.8))
            ax.plot(epochs, [r["train_unweighted_ce"] for r in history], label="Train")
            ax.plot(epochs, [r["val_unweighted_ce"] for r in history], label="Validation")
            ax.set(
                xlabel="Epoch",
                ylabel="Unweighted state cross-entropy",
                title="Common diagnostic across loss functions",
            )
            ax.legend(frameon=False)
            ax.grid(alpha=0.2)
            fig.tight_layout()
            save(fig, out / "unweighted_ce.png")
            plt.close(fig)
        fig, ax = plt.subplots(figsize=(6.5, 3.8))
        for name, color in [("precision", "#2463ae"), ("recall", "#df7d16"), ("f1", "#7b4ab0")]:
            ax.plot(epochs, [r[f"val_clip_{name}"] for r in history], label=name.title(), color=color)
        ax.set(xlabel="Epoch", ylabel="Clip metric", ylim=(0, 1.02), title="Validation clip performance")
        ax.legend(frameon=False)
        ax.grid(alpha=0.2)
        fig.tight_layout()
        save(fig, out / "validation_clip_metrics.png")
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(6.5, 3.2))
        ax.step(epochs, [r["learning_rate"] for r in history], where="post")
        ax.set(xlabel="Epoch", ylabel="Learning rate", yscale="log")
        ax.grid(alpha=0.2)
        fig.tight_layout()
        save(fig, out / "learning_rate.png")
        plt.close(fig)


def compare(runs, split, output, _dataset=None):
    """Collect the metrics of several runs into comparison.csv and a bar chart (clip_comparison.png)."""
    import csv

    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    reference = None
    for directory in runs:
        directory = Path(directory)
        report = json.loads((directory / f"{split}_metrics.json").read_text())
        info = json.loads((directory / "config.json").read_text())
        if _dataset is not None:
            report = report["by_dataset"][_dataset]
        clips = {c["video_id"]: c["truth"] for c in report["clips"]}
        if reference is None:
            reference = clips
        elif clips != reference:
            raise ValueError("Runs do not evaluate the same clips/ground truth")
        if info.get("synthetic", False):
            raise ValueError("Synthetic smoke runs cannot be included in experiment comparisons")
        rows.append(
            dict(
                run=directory.name,
                model=info["model"],
                scene=info["scene"],
                feature_set=info["feature_set"],
                history_window=info.get("history_window", 30),
                loss_name=info.get("loss_name", "weighted_ce"),
                preprocessing=info.get("image_preprocessing", info["config"]["preprocessing"]),
                seed=info["seed"],
                clip_precision=report["clip"]["precision"],
                clip_recall=report["clip"]["recall"],
                clip_f1=report["clip"]["f1"],
                event_precision=report["event"]["precision"],
                event_recall=report["event"]["recall"],
                event_f1=report["event"]["f1"],
                mean_delay_seconds=report["event"]["mean_delay_seconds"],
                coverage=report["prediction_coverage"],
                lying_clips=report.get("hard_negative_lying", {}).get("clips", 0),
                lying_false_positive_clips=report.get("hard_negative_lying", {}).get(
                    "false_positive_clips", 0
                ),
            )
        )
    with (out / "comparison.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    fig, ax = plt.subplots(figsize=(max(7, len(rows) * 1.5), 4.2))
    x = np.arange(len(rows))
    width = 0.23
    for i, (metric, color) in enumerate([("precision", "#2463ae"), ("recall", "#df7d16"), ("f1", "#7b4ab0")]):
        values = [r["clip_" + metric] for r in rows]
        bars = ax.bar(x + (i - 1) * width, values, width, label=metric.title(), color=color)
        ax.bar_label(bars, labels=[f"{100*v:.1f}%" for v in values], padding=3, fontsize=8)
    names = {
        "rf": "Random Forest",
        "svm": "SVM",
        "cnn2d": "CNN2D",
        "cnn_lstm": "CNN2D + LSTM",
        "pose_lstm": "Pose + LSTM",
        "pose_transformer": "Pose + Transformer",
    }
    labels = []
    for r in rows:
        peers = [z for z in rows if z["model"] == r["model"]]
        parts = [names.get(r["model"], r["model"]) + (" + scene" if r["scene"] else "")]
        for key in ("feature_set", "history_window", "loss_name", "preprocessing", "seed"):
            if len({z[key] for z in peers}) > 1:
                parts.append(
                    f'{dict(feature_set="features",history_window="history",loss_name="loss",preprocessing="image",seed="seed")[key]}={r[key]}'
                )
        labels.append("\n".join(parts))
    from matplotlib.ticker import PercentFormatter

    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set(
        xticks=x,
        xticklabels=labels,
        ylim=(0, 1.15),
        ylabel="Clip metric",
        title=f"{split.upper()} · ADL / Fall",
    )
    ax.legend(frameon=False, ncols=3)
    ax.grid(axis="y", alpha=0.15)
    fig.autofmt_xdate(rotation=0)
    fig.tight_layout()
    fig.savefig(out / "clip_comparison.png")
    plt.close(fig)
    if _dataset is None:
        reports = [json.loads((Path(r) / f"{split}_metrics.json").read_text()) for r in runs]
        datasets = set(reports[0].get("by_dataset", {}))
        if any(set(r.get("by_dataset", {})) != datasets for r in reports):
            raise ValueError("Dataset reports differ; rerun evaluation for every checkpoint")
        for dataset in sorted(datasets):
            compare(runs, split, out / dataset, _dataset=dataset)
    return rows
