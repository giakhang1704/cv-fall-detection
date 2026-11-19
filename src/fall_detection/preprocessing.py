"""Deterministic BGR-in/BGR-out preprocessing ablations, one operation at a time."""

import cv2
import numpy as np

VARIANTS = {
    "none": {},
    "gaussian": {"ksize": 3, "sigma": 0.8},
    "median": {"ksize": 3},
    "bilateral": {"diameter": 5, "sigma_color": 50, "sigma_space": 50},
    "hist_equalization": {"channel": "Y in YCrCb"},
    "clahe": {"channel": "L in Lab", "clip_limit": 2.0, "tile_grid": 8},
    "gamma_bright": {"gamma": 0.7, "formula": "255*(I/255)^gamma"},
    "gamma_dark": {"gamma": 1.3, "formula": "255*(I/255)^gamma"},
    "grayscale": {"output": "gray repeated over three channels"},
    "hsv_value_equalization": {"channel": "V in HSV", "output": "converted back to BGR"},
}
LUTS = {
    name: np.round(255 * (np.arange(256) / 255.0) ** p["gamma"]).astype(np.uint8)
    for name, p in VARIANTS.items()
    if "gamma" in p
}


def apply(frame, variant="none"):
    """Apply one preprocessing variant (see ``VARIANTS``) to a uint8 BGR image; returns a BGR image."""
    if variant not in VARIANTS:
        raise ValueError(f"Unknown preprocessing: {variant}")
    if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("Expected uint8 BGR image")
    if variant == "none":
        return frame
    if variant == "gaussian":
        return cv2.GaussianBlur(frame, (3, 3), 0.8)
    if variant == "median":
        return cv2.medianBlur(frame, 3)
    if variant == "bilateral":
        return cv2.bilateralFilter(frame, 5, 50, 50)
    if variant in LUTS:
        return cv2.LUT(frame, LUTS[variant])
    if variant == "grayscale":
        return cv2.cvtColor(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    if variant == "hist_equalization":
        y = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        y[:, :, 0] = cv2.equalizeHist(y[:, :, 0])
        return cv2.cvtColor(y, cv2.COLOR_YCrCb2BGR)
    if variant == "clahe":
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def image_metrics(frame):
    """Simple image statistics: brightness, contrast, clipped-pixel fraction and sharpness."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (160, 90), interpolation=cv2.INTER_AREA)
    return dict(
        brightness=float(gray.mean()),
        contrast=float(gray.std()),
        clipped_fraction=float(np.mean((gray == 0) | (gray == 255))),
        laplacian_variance=float(cv2.Laplacian(gray, cv2.CV_32F).var()),
    )
