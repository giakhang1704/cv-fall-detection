"""Causal body descriptors and background edge/line geometry. No inferred floor."""

import cv2
import numpy as np

SCENE_NAMES = [
    "edge_density",
    "horizontal_length",
    "diagonal_length",
    "vertical_length",
    "line_density",
    "nearest_line_to_box_bottom",
    "horizontal_below_box",
    "scene_observed",
]
MOTION_NAMES = [
    "box_log_aspect",
    "box_height",
    "body_tilt",
    "shoulder_hip_length",
    "head_above_hip",
    "left_knee_bend",
    "right_knee_bend",
    "hip_visible",
    "hip_vy",
    "head_vy",
    "width_velocity",
    "height_velocity",
]


def scene_features(frame, box):
    """Compute the 8-D scene vector from Canny edges and Hough line segments of the background.

    Edges inside the (enlarged) person box are erased after Canny so the mask border does not
    create fake lines. Returns (features, edge map, line segments, resize scale).
    """
    h, w = frame.shape[:2]
    scale = min(1.0, 640 / max(h, w))
    im = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    hh, ww = im.shape[:2]
    b = np.asarray(box) * scale
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 1), 50, 150)
    # Mask AFTER detecting edges: a black rectangle must not generate artificial lines.
    mask = np.ones((hh, ww), np.uint8)
    margin = max(4, int(0.05 * max(b[2:] - b[:2])))
    x1, y1 = np.maximum(0, np.floor(b[:2] - margin)).astype(int)
    x2, y2 = np.minimum([ww, hh], np.ceil(b[2:] + margin)).astype(int)
    mask[y1:y2, x1:x2] = 0
    edges[mask == 0] = 0
    if np.count_nonzero(mask) < 0.1 * mask.size:
        return np.zeros(8, np.float32), edges, np.empty((0, 4), np.int32), scale
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=35, minLineLength=max(15, int(0.08 * ww)), maxLineGap=10
    )
    lines = np.empty((0, 4), np.int32) if lines is None else np.asarray(lines).reshape(-1, 4)
    diag = max(np.hypot(ww, hh), 1)
    orient = np.zeros(3)
    distances = []
    below = 0.0
    foot = np.array([(b[0] + b[2]) / 2, b[3]])
    for x1, y1, x2, y2 in lines:
        delta = np.array([x2 - x1, y2 - y1], float)
        length = np.linalg.norm(delta)
        angle = np.degrees(np.arctan2(abs(delta[1]), abs(delta[0]) + 1e-8))
        kind = 0 if angle < 15 else 2 if angle > 75 else 1
        orient[kind] += length / diag
        a = np.array([x1, y1], float)
        alpha = np.clip(np.dot(foot - a, delta) / max(length**2, 1e-8), 0, 1)
        distances.append(np.linalg.norm(foot - (a + alpha * delta)) / diag)
        if kind == 0 and (y1 + y2) / 2 >= foot[1]:
            below += length / diag
    features = np.r_[
        np.count_nonzero(edges) / max(np.count_nonzero(mask), 1),
        np.clip(orient, 0, 20),
        len(lines) / 100,
        min(distances, default=1.0),
        min(below, 20),
        1.0,
    ].astype(np.float32)
    return features, edges, lines, scale


def angle(a, b, c):
    """Cosine of the angle at point ``b`` formed by points ``a`` and ``c`` (used for knee bend)."""
    u = a - b
    v = c - b
    return float(np.dot(u, v) / max(np.linalg.norm(u) * np.linalg.norm(v), 1e-6))


def body_features(keys, boxes, times, valid, ids, width, height, keypoint_conf=0.3):
    """Compute per-observation pose features for one video.

    Keypoints are normalised around the hip centre and scaled by torso length; 12 geometric and
    velocity descriptors are added. Velocities are only computed between consecutive
    observations of the same identity. Returns (keypoints + visibility, keypoints + visibility
    + motion, motion).
    """
    n = len(times)
    xy = np.zeros((n, 34), np.float32)
    visibility = np.zeros((n, 17), np.float32)
    motion = np.zeros((n, 12), np.float32)
    hip_y = np.zeros(n)
    head_y = np.zeros(n)
    for i in np.flatnonzero(valid):
        k = keys[i]
        b = boxes[i]
        seen = k[:, 2] >= keypoint_conf
        visibility[i] = seen
        hip = k[[11, 12], :2][seen[[11, 12]]].mean(axis=0) if seen[[11, 12]].any() else (b[:2] + b[2:]) / 2
        shoulder = (
            k[[5, 6], :2][seen[[5, 6]]].mean(axis=0) if seen[[5, 6]].any() else np.array([hip[0], b[1]])
        )
        torso = shoulder - hip
        length = np.linalg.norm(torso)
        scale = max(length, 0.15 * np.linalg.norm(b[2:] - b[:2]), 1.0)
        norm = np.clip((k[:, :2] - hip) / scale, -5, 5)
        norm[~seen] = 0
        xy[i] = norm.reshape(-1)
        head = k[:5, :2][seen[:5]].mean(axis=0) if seen[:5].any() else shoulder
        bw, bh = np.maximum(b[2:] - b[:2], 1)
        knees = [
            angle(k[a, :2], k[b_, :2], k[c, :2]) if seen[[a, b_, c]].all() else 0.0
            for a, b_, c in [(11, 13, 15), (12, 14, 16)]
        ]
        motion[i, :8] = [
            np.clip(np.log(bw / bh), -3, 3),
            bh / height,
            abs(torso[0]) / max(length, 1),
            length / height,
            (hip[1] - head[1]) / height,
            *knees,
            float(seen[[11, 12]].mean()),
        ]
        hip_y[i] = hip[1] / height
        head_y[i] = head[1] / height
        if i and valid[i - 1] and ids[i] == ids[i - 1] and 0 < times[i] - times[i - 1] <= 0.25 + 1e-8:
            dt = times[i] - times[i - 1]
            motion[i, 8:] = np.clip(
                [
                    (hip_y[i] - hip_y[i - 1]) / dt,
                    (head_y[i] - head_y[i - 1]) / dt,
                    ((boxes[i, 2] - boxes[i, 0]) - (boxes[i - 1, 2] - boxes[i - 1, 0])) / width / dt,
                    ((boxes[i, 3] - boxes[i, 1]) - (boxes[i - 1, 3] - boxes[i - 1, 1])) / height / dt,
                ],
                -10,
                10,
            )
    # The observation mask stays explicit; missing coordinates are never repeated.
    base = np.c_[xy, visibility].astype(np.float32)
    return base, np.c_[base, motion].astype(np.float32), motion


DYNAMICS_NAMES = [
    "hip_acceleration_image",
    "head_acceleration_image",
    "torso_angular_velocity",
    "dynamics_observed",
]


def dynamics_features(motion, times, valid, ids, tau=0.15, max_gap=0.25):
    """Causal EMA velocities then acceleration; never differentiate through a gap/identity change.

    Velocities use image-height/second. Acceleration is not physical m/s^2.
    Body tilt is converted to an unsigned inclination angle (radians).
    """
    extra = np.zeros((len(times), 4), np.float32)
    previous = None
    for i in range(len(times)):
        if not valid[i]:
            previous = None
            continue
        current = np.r_[motion[i, 8:10], np.arcsin(np.clip(motion[i, 2], 0, 1))]
        if previous is not None:
            j, state, velocity = previous
            dt = float(times[i] - times[j])
            if ids[i] == ids[j] and 0 < dt <= max_gap + 1e-8:
                alpha = 1 - np.exp(-dt / tau)
                smoothed = state + alpha * (current - state)
                derivative = (smoothed - state) / dt
                extra[i] = [
                    np.clip(derivative[0], -20, 20),
                    np.clip(derivative[1], -20, 20),
                    np.clip(derivative[2], -10, 10),
                    1.0,
                ]
                previous = (i, smoothed, current)
                continue
        previous = (i, current.copy(), current.copy())
    return extra
