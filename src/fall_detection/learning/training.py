"""Shared training/validation protocol; test is loaded only by explicit evaluate."""

import copy
import random
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ..common import ROOT, digest, project_path, rel, write_json
from .metrics import evaluate, select_threshold
from .models import Classifier
from .prepare import load_records, window_arrays


def _relative_to_root(path):
    """Store data paths relative to the project so checkpoints work on any machine."""
    path = Path(path).resolve()
    try:
        return path.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def scale_windows(x, scaler=None):
    """Standardise features using only observed (pose present) entries; the scaler is fitted on
    the first call (train) and reused afterwards. The observation flag is kept unchanged.
    """
    if x.ndim != 3:
        raise ValueError("Expected N,T,D")
    observed = x[:, :, -1] > 0.5
    z = x[:, :, :-1]
    if scaler is None:
        if not observed.any():
            raise ValueError("No observed training data")
        scaler = StandardScaler().fit(z[observed])
    out = np.zeros_like(z)
    out[observed] = scaler.transform(z[observed])
    out = np.clip(out, -10, 10)
    return np.concatenate([out, x[:, :, -1:]], axis=2).astype(np.float32), scaler


def probabilities(model, x, lengths=None, batch=256):
    """Softmax probabilities of a neural classifier, computed in batches without gradients."""
    model.eval()
    result = []
    with torch.inference_mode():
        for i in range(0, len(x), batch):
            length = None if lengths is None else torch.from_numpy(lengths[i : i + batch])
            result.append(model(torch.from_numpy(x[i : i + batch]), length).softmax(1).numpy())
    return np.concatenate(result)


def classical_probabilities(model, x):
    """Probabilities of a scikit-learn classifier mapped to the 3 classes in fixed order."""
    raw = model.predict_proba(x)
    out = np.zeros((len(x), 3), np.float32)
    out[:, model.classes_.astype(int)] = raw
    return out


def check_splits(train, val):
    """Raise an error if a subject or a video appears in both train and validation."""
    a = {r["meta"]["subject"] for r in train}
    b = {r["meta"]["subject"] for r in val}
    if a & b:
        raise ValueError("Subject leakage between train and validation")
    if {r["meta"]["video_id"] for r in train} & {r["meta"]["video_id"] for r in val}:
        raise ValueError("Video leakage")


def train(
    cfg,
    data,
    kind,
    run,
    scene=False,
    feature_set="motion",
    epochs=None,
    seed=17,
    history=None,
    loss_name="weighted_ce",
    focal_gamma=2.0,
    transformer_dim=32,
    heads=2,
    layers=1,
):
    """Train one classifier on a prepared dataset and save the best checkpoint.

    The checkpoint and the alarm threshold are selected on validation by clip F1. The run
    directory stores best.pt (or best.joblib), config.json, history.json and validation metrics.
    """
    run = Path(run)
    if run.exists() and any(run.iterdir()):
        raise ValueError("Run directory is not empty; choose a new run name")
    run.mkdir(parents=True, exist_ok=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    tr, index = load_records(data, "train")
    va, val_index = load_records(data, "val")
    check_splits(tr, va)
    if index["partial"]:
        raise ValueError("Training on partial caches is disabled. Complete preparation first.")
    for key in ("window", "stride", "max_gap", "min_valid_fraction"):
        name = {"max_gap": "max_gap_seconds"}.get(key, key)
        if index["settings"][key] != cfg[name]:
            raise ValueError(f"Prepared/training setting differs: {key}")
    history = cfg["window"] if history is None else history
    neural = kind not in ("rf", "svm")
    training_arrays = window_arrays(
        tr,
        kind,
        scene,
        feature_set,
        training=True,
        history=history,
        time_features=True,
        return_lengths=neural,
    )
    validation_arrays = window_arrays(
        va, kind, scene, feature_set, history=history, time_features=True, return_lengths=neural
    )
    x, y, _ = training_arrays[:3]
    vx, _, vrefs = validation_arrays[:3]
    lengths = training_arrays[3] if neural else None
    vlengths = validation_arrays[3] if neural else None
    if set(y.tolist()) != {0, 1, 2}:
        raise ValueError("Training data must contain all three states")
    cfg = dict(cfg, preprocessing=index["settings"].get("pose_preprocessing", "none"))
    cfg_saved = dict(cfg, source_project=rel(cfg["source_project"]))
    prepared_signature = digest(Path(data) / "index.json")
    import importlib.metadata

    learning_code = {p.name: digest(p) for p in Path(__file__).parent.glob("*.py")}
    versions = {
        name: importlib.metadata.version(name) for name in ("torch", "torchvision", "scikit-learn", "numpy")
    }
    info = dict(
        image_preprocessing=(
            index["settings"].get("rgb_preprocessing", "none")
            if kind in ("cnn2d", "cnn_lstm")
            else cfg["preprocessing"]
        ),
        input_schema=2,
        history_window=history,
        loss_name=loss_name,
        focal_gamma=focal_gamma,
        transformer_dim=transformer_dim,
        heads=heads,
        layers=layers,
        learning_code=learning_code,
        versions=versions,
        model=kind,
        scene=scene,
        feature_set=feature_set,
        seed=seed,
        config=cfg_saved,
        prepared_index_sha256=prepared_signature,
        data=_relative_to_root(data),
        training_windows=len(y),
        validation_windows=len(vx),
        test_loaded=False,
        synthetic=bool(index.get("synthetic", False)),
        comparison_note="Common pose-available endpoints; image encoder frozen.",
    )
    write_json(run / "config.json", info)
    if kind in ("rf", "svm"):
        if loss_name != "weighted_ce":
            raise ValueError(
                "Loss selection applies only to neural models; classical models use balanced class weights"
            )
        info["loss_name"] = "class_weight_balanced"
        scaler = StandardScaler().fit(x)
        sx = scaler.transform(x)
        sv = scaler.transform(vx)
        if kind == "rf":
            model = RandomForestClassifier(
                n_estimators=300,
                max_depth=8,
                min_samples_leaf=5,
                class_weight="balanced",
                n_jobs=2,
                random_state=seed,
            )
        else:
            from sklearn.calibration import CalibratedClassifierCV
            from sklearn.model_selection import GroupKFold

            _, _, train_refs = window_arrays(tr, kind, scene, feature_set, training=True)
            groups = np.asarray([tr[j]["meta"]["subject"] for j, e in train_refs])
            if len(np.unique(groups)) < 3:
                raise ValueError("SVM calibration requires at least three training subjects")
            folds = list(GroupKFold(3).split(sx, y, groups))
            model = CalibratedClassifierCV(
                SVC(C=1.0, kernel="rbf", class_weight="balanced"), cv=folds, method="sigmoid"
            )
        info["model_parameters"] = model.get_params(deep=False)
        info["model_parameters"] = {
            key: value
            for key, value in info["model_parameters"].items()
            if isinstance(value, (str, int, float, bool, type(None)))
        }
        write_json(run / "config.json", info)
        start = time.perf_counter()
        model.fit(sx, y)
        vp = classical_probabilities(model, sv)
        report = select_threshold(va, vp, vrefs, cfg)
        checkpoint = dict(info=info, model=model, scaler=scaler, threshold=report["threshold"])
        joblib.dump(checkpoint, run / "best.joblib")
        write_json(
            run / "history.json",
            [dict(epoch=1, training_seconds=time.perf_counter() - start, val_clip_f1=report["clip"]["f1"])],
        )
    else:
        x, scaler = scale_windows(x)
        vx, _ = scale_windows(vx, scaler)
        model = Classifier(
            kind, x.shape[2], cfg["hidden_size"], cfg["dropout"], transformer_dim, heads, layers
        )
        opt = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
        info["trainable_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        write_json(run / "config.json", info)
        from .losses import make_loss

        criterion, weights = make_loss(loss_name, y, focal_gamma)
        info["loss_class_weights"] = weights.tolist()
        write_json(run / "config.json", info)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=3)
        loader = DataLoader(
            TensorDataset(torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(lengths)),
            batch_size=cfg["batch_size"],
            shuffle=True,
            generator=torch.Generator().manual_seed(seed),
        )
        history_rows = []
        best = (-1.0, -1.0)
        stale = 0
        best_report = None
        vy = np.asarray([va[j]["labels"][e] for j, e in vrefs])
        for epoch in range(1, (epochs or cfg["epochs"]) + 1):
            start = time.perf_counter()
            model.train()
            loss_sum = 0.0
            ce_sum = 0.0
            for a, b, lens in loader:
                opt.zero_grad()
                logits = model(a, lens)
                loss = criterion(logits, b)
                ce_sum += nn.functional.cross_entropy(logits.detach(), b, reduction="sum").item()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                loss_sum += loss.item() * len(b)
            vp = probabilities(model, vx, vlengths)
            report = select_threshold(va, vp, vrefs, cfg)
            score = (report["clip"]["f1"], report["clip"]["recall"])
            scheduler.step(score[0])
            known = vy >= 0
            val_loss = (
                float(
                    criterion(
                        torch.from_numpy(np.log(np.clip(vp[known], 1e-8, 1))), torch.from_numpy(vy[known])
                    ).item()
                )
                if known.any()
                else None
            )
            common_ce = (
                float(-np.log(np.clip(vp[np.flatnonzero(known), vy[known]], 1e-8, 1)).mean())
                if known.any()
                else None
            )
            row = dict(
                train_unweighted_ce=ce_sum / len(y),
                val_unweighted_ce=common_ce,
                epoch=epoch,
                train_loss=loss_sum / len(y),
                val_loss=val_loss,
                val_clip_precision=report["clip"]["precision"],
                val_clip_recall=report["clip"]["recall"],
                val_clip_f1=score[0],
                threshold=report["threshold"],
                learning_rate=opt.param_groups[0]["lr"],
                seconds=time.perf_counter() - start,
            )
            history_rows.append(row)
            print(
                f'Epoch {epoch:03d} loss={row["train_loss"]:.4f} val_clip_F1={score[0]:.4f} R={score[1]:.4f}',
                flush=True,
            )
            state = dict(
                info=info,
                input_dim=x.shape[2],
                state_dict=copy.deepcopy(model.state_dict()),
                scaler=scaler,
                threshold=report["threshold"],
                epoch=epoch,
            )
            torch.save(state, run / "last.pt")
            if score > best:
                best = score
                best_report = report
                stale = 0
                torch.save(state, run / "best.pt")
            else:
                stale += 1
            write_json(run / "history.json", history_rows)
            if stale >= cfg["patience"]:
                break
        report = best_report
    report.update(
        synthetic=info.get("synthetic", False),
        loss_name=info["loss_name"],
        history_window=history,
        model=kind,
    )
    write_json(run / "val_metrics.json", report)
    from .plots import render

    render(run, report)
    return dict(
        run=rel(run),
        validation=report["clip"],
        checkpoint="best.joblib" if kind in ("rf", "svm") else "best.pt",
    )


def evaluate_checkpoint(checkpoint, split="test"):
    """Evaluate a saved checkpoint on the validation or test split with its stored threshold."""
    path = Path(checkpoint)
    state = (
        joblib.load(path)
        if path.suffix == ".joblib"
        else torch.load(path, map_location="cpu", weights_only=False)
    )
    info = state["info"]
    data = project_path(info["data"])
    cfg = info["config"]
    torch.set_num_threads(2)
    if digest(data / "index.json") != info["prepared_index_sha256"]:
        raise ValueError("Prepared index changed since training")
    records, index = load_records(data, split)
    neural = info["model"] not in ("rf", "svm")
    arrays = window_arrays(
        records,
        info["model"],
        info["scene"],
        info["feature_set"],
        history=info.get("history_window", cfg["window"]),
        time_features=info.get("input_schema", 1) >= 2,
        return_lengths=neural,
    )
    x, _, refs = arrays[:3]
    lengths = arrays[3] if neural else None
    if path.suffix == ".joblib":
        p = classical_probabilities(state["model"], state["scaler"].transform(x))
    else:
        x, _ = scale_windows(x, state["scaler"])
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
        p = probabilities(model, x, lengths)
    report = evaluate(records, p, refs, state["threshold"], cfg)
    report.update(
        synthetic=info.get("synthetic", False),
        split=split,
        checkpoint_sha256=digest(path),
        model=info["model"],
        scene=info["scene"],
        feature_set=info["feature_set"],
        history_window=info.get("history_window", cfg["window"]),
        loss_name=info.get("loss_name", "weighted_ce"),
    )
    from .metrics import evaluate_by_dataset

    report["by_dataset"] = evaluate_by_dataset(records, p, refs, state["threshold"], cfg)
    report["aggregation"] = (
        "Overall metrics pool all clips/events; they are not an unweighted average of dataset metrics. Same validation-selected threshold across datasets."
    )
    output = path.parent / f"{split}_metrics.json"
    write_json(output, report)
    np.savez_compressed(
        path.parent / f"{split}_predictions.npz",
        probabilities=p,
        record_indices=np.asarray(refs),
        video_ids=np.asarray([r["meta"]["video_id"] for r in records]),
    )
    from .plots import render

    render(path.parent, report, prefix=split)
    for dataset, subset in report["by_dataset"].items():
        subset["synthetic"] = info.get("synthetic", False)
        render(path.parent, subset, prefix=f"{split}_{dataset}")
    return report
