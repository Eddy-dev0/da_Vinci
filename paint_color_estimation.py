"""Utilities for estimating paint colors from image regions."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass
class ColorSample:
    hsv: Tuple[float, float, float]
    lab: Tuple[float, float, float]
    rgb: Tuple[int, int, int]


def _clip_bbox(bbox: Tuple[int, int, int, int], shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    h, w = shape[:2]
    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h))
    if x2 <= x1:
        x2 = min(w, x1 + 1)
    if y2 <= y1:
        y2 = min(h, y1 + 1)
    return x1, y1, x2, y2


def apply_white_balance(frame: np.ndarray, gray_roi: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray:
    """Apply simple gray-world white balance.

    If gray_roi is provided, it is used to estimate the neutral color.
    """
    if gray_roi is not None:
        x1, y1, x2, y2 = _clip_bbox(gray_roi, frame.shape)
        roi = frame[y1:y2, x1:x2]
    else:
        roi = frame

    if roi.size == 0:
        return frame

    roi_float = roi.astype(np.float32)
    means = roi_float.reshape(-1, 3).mean(axis=0)
    target = float(np.mean(means))
    scales = target / (means + 1e-6)
    balanced = frame.astype(np.float32) * scales
    return np.clip(balanced, 0, 255).astype(np.uint8)


def roi_sample(
    frame: np.ndarray,
    bbox: Tuple[int, int, int, int],
    padding: int = 8,
    use_median: bool = True,
) -> ColorSample:
    """Sample HSV/Lab/RGB from a padded region of interest."""
    x1, y1, x2, y2 = bbox
    padded = (x1 - padding, y1 - padding, x2 + padding, y2 + padding)
    x1, y1, x2, y2 = _clip_bbox(padded, frame.shape)

    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        raise ValueError("ROI is empty after clipping")

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)

    flat_hsv = hsv.reshape(-1, 3)
    flat_lab = lab.reshape(-1, 3)
    flat_bgr = roi.reshape(-1, 3)

    if use_median:
        hsv_value = np.median(flat_hsv, axis=0)
        lab_value = np.median(flat_lab, axis=0)
        bgr_value = np.median(flat_bgr, axis=0)
    else:
        hsv_value = np.mean(flat_hsv, axis=0)
        lab_value = np.mean(flat_lab, axis=0)
        bgr_value = np.mean(flat_bgr, axis=0)

    rgb_value = bgr_value[::-1]

    return ColorSample(
        hsv=(float(hsv_value[0]), float(hsv_value[1]), float(hsv_value[2])),
        lab=(float(lab_value[0]), float(lab_value[1]), float(lab_value[2])),
        rgb=(int(rgb_value[0]), int(rgb_value[1]), int(rgb_value[2])),
    )


def bbox_from_corners(corners: np.ndarray) -> Tuple[int, int, int, int]:
    """Convert marker corners to a bounding box."""
    pts = corners.reshape(-1, 2)
    x1 = int(np.min(pts[:, 0]))
    y1 = int(np.min(pts[:, 1]))
    x2 = int(np.max(pts[:, 0]))
    y2 = int(np.max(pts[:, 1]))
    return x1, y1, x2, y2
