"""Clip decisions and one-to-one event evaluation from causal confirmed alarms."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)


def episodes(intervals, tolerance=0.0):
    """Merge contiguous Falling/Fallen intervals into fall events [start, end].

    Two fall intervals are merged only if no ADL interval lies between them.
    """
    events = []
    for r in sorted((r for r in intervals if r["mapped_label"] in (1, 2)), key=lambda r: r["start"]):
        a, b = float(r["start"]), float(r["end"])
        intervening_adl = bool(events) and any(
            z["mapped_label"] == 0 and float(z["start"]) < a and float(z["end"]) > events[-1][1]
            for z in intervals
        )
        if events and a <= events[-1][1] + tolerance + 1e-7 and not intervening_adl:
            events[-1][1] = max(events[-1][1], b)
        else:
            events.append([a, b])
    return events


def alarm_onsets(times, scores, threshold, confirm=0.2, release=1.0, off_ratio=0.7):
    """Turn a score sequence into confirmed alarms (the alarm confirmation mechanism).

    An alarm starts when the score stays >= ``threshold`` for ``confirm`` seconds and ends when it
    stays below ``off_ratio * threshold`` for ``release`` seconds. A missing score (NaN) resets
    the timers. Returns (onset times, active state per step).
    """
    active = False
    above = below = None
    onsets = []
    states = []
    for t, p in zip(times, scores):
        if not np.isfinite(p):
            above = below = None
            states.append(active)
            continue
        if not active:
            if p >= threshold:
                if above is None:
                    above = t
                if t - above >= confirm - 1e-8:
                    active = True
                    onsets.append(float(t))
                    above = None
            else:
                above = None
        else:
            if p < threshold * off_ratio:
                if below is None:
                    below = t
                if t - below >= release - 1e-8:
                    active = False
                    below = None
            else:
                below = None
        states.append(active)
    return onsets, states


def evaluate(records, probabilities, refs, threshold, cfg):
    """Evaluate alarms against the labels at clip and event level.

    Clip level: a Fall clip is detected if it has at least one confirmed alarm. Event level:
    each alarm is matched one-to-one with a fall event; delay is measured from the event start.
    Also reports prediction coverage and false alarms on intentional-lying ADL clips.
    """
    lookup = {}
    for p, (j, e) in zip(probabilities, refs):
        lookup[(j, e)] = p
    yt = []
    yp = []
    clips = []
    tp = fp = fn = unknown = 0
    delays = []
    covered = scheduled = 0
    for j, r in enumerate(records):
        stride = cfg["stride"]
        w = r["meta"]["window"]
        # Endpoints are aligned with the actual window scheduler, including warmup ticks.
        ticks = np.arange((w - 1) % stride, len(r["times"]), stride)
        scores = np.array(
            [float(lookup[(j, int(e))][1:].sum()) if (j, int(e)) in lookup else np.nan for e in ticks]
        )
        replay = {int(e): s for e, s in zip(ticks, scores)}
        replay.update({int(e): np.nan for e in np.flatnonzero(~r["valid"])})
        replay_indices = sorted(replay)
        onsets, _ = alarm_onsets(
            r["times"][replay_indices],
            [replay[e] for e in replay_indices],
            threshold,
            cfg["alarm_confirm_seconds"],
            cfg["alarm_release_seconds"],
            cfg["alarm_off_ratio"],
        )
        intervals = r["meta"]["intervals"]
        events = episodes(intervals, 1 / cfg.get("sample_hz", 15.0))
        matched = set()
        matches = []
        false = []
        unclassified = []
        for onset in onsets:
            active_intervals = [z for z in intervals if float(z["start"]) <= onset < float(z["end"])]
            if len(active_intervals) != 1:
                unknown += 1
                unclassified.append(onset)
                continue
            choices = [i for i, (a, b) in enumerate(events) if i not in matched and a <= onset < b]
            if choices:
                i = choices[0]
                matched.add(i)
                delay = onset - events[i][0]
                delays.append(delay)
                matches.append(dict(onset=onset, event=i, delay_seconds=delay))
                tp += 1
            elif any(float(z["start"]) <= onset < float(z["end"]) for z in intervals):
                fp += 1
                false.append(onset)
            else:
                unknown += 1
                unclassified.append(onset)
        fn += len(events) - len(matched)
        truth = int(bool(events))
        prediction = int(bool(onsets))
        yt.append(truth)
        yp.append(prediction)
        annotated = r["labels"][ticks] >= 0
        scheduled += int(annotated.sum())
        covered += int((annotated & np.isfinite(scores)).sum())
        lying = any(
            str(z.get("label", "")).split(".")[0] in ("5", "6") and z.get("mapped_label") == 0
            for z in r["meta"].get("source_intervals", [])
        )
        clips.append(
            dict(
                has_intentional_lying_annotation=lying,
                has_prediction=bool(np.isfinite(scores).any()),
                video_id=r["meta"]["video_id"],
                truth=truth,
                prediction=prediction,
                outcome=("TP" if truth else "FP") if prediction else ("FN" if truth else "TN"),
                alarm_onsets=onsets,
                event_matches=matches,
                false_event_alarms=false,
                unknown_event_alarms=unclassified,
                events=events,
                prediction_coverage=(
                    float(np.mean(np.isfinite(scores)[annotated])) if annotated.any() else 0.0
                ),
            )
        )
    p, rec, f1, _ = precision_recall_fscore_support(yt, yp, average="binary", zero_division=0)
    ep = tp / max(tp + fp, 1)
    er = tp / max(tp + fn, 1)
    hard = [c for c in clips if c["truth"] == 0 and c["has_intentional_lying_annotation"]]
    hard_fp = sum(c["prediction"] for c in hard)
    subgroup = dict(
        clips=len(hard),
        false_positive_clips=hard_fp,
        false_positive_fraction=hard_fp / len(hard) if hard else None,
        clips_without_prediction=sum(not c["has_prediction"] for c in hard),
        mean_coverage=float(np.mean([c["prediction_coverage"] for c in hard])) if hard else None,
        note="Explicit source lie_down/lying ADL clips only, not exhaustive; generic ADL not assumed to be lying.",
    )
    return dict(
        hard_negative_lying=subgroup,
        threshold=float(threshold),
        clip=dict(
            count=len(yt),
            class_names=["ADL", "Fall"],
            confusion_matrix=confusion_matrix(yt, yp, labels=[0, 1]).tolist(),
            precision=float(p),
            recall=float(rec),
            f1=float(f1),
            accuracy=float(accuracy_score(yt, yp)),
        ),
        event=dict(
            tp=tp,
            fp=fp,
            fn=fn,
            unknown_alarms=unknown,
            precision=ep,
            recall=er,
            f1=2 * ep * er / max(ep + er, 1e-12),
            mean_delay_seconds=float(np.mean(delays)) if delays else None,
        ),
        prediction_coverage=covered / max(scheduled, 1),
        clips=clips,
        protocol="Clip Fall iff >=1 confirmed alarm. Event one-to-one onset inside annotated falling/contiguous fallen episode; delay from episode start, matched events only. Missing predictions cancel confirmation timers; no stale score updates.",
    )


def select_threshold(records, probs, refs, cfg):
    """Pick the alarm threshold from ``cfg['threshold_grid']`` with the best clip F1 (ties: recall, precision)."""
    reports = [evaluate(records, probs, refs, float(t), cfg) for t in cfg["threshold_grid"]]
    return max(
        reports, key=lambda r: (r["clip"]["f1"], r["clip"]["recall"], r["clip"]["precision"], -r["threshold"])
    )


def evaluate_by_dataset(records, probs, refs, threshold, cfg):
    """Use the same frozen global threshold for every source dataset."""
    result = {}
    for name in sorted({r["meta"]["dataset"] for r in records}):
        indices = [i for i, r in enumerate(records) if r["meta"]["dataset"] == name]
        mapping = {old: new for new, old in enumerate(indices)}
        positions = [i for i, (j, e) in enumerate(refs) if j in mapping]
        subset_refs = [(mapping[refs[i][0]], refs[i][1]) for i in positions]
        result[name] = evaluate(
            [records[i] for i in indices], np.asarray(probs)[positions], subset_refs, threshold, cfg
        )
    return result
