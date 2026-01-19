"""Camera utilities for the vision system."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass
class CameraCalibration:
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray


def _parse_matrix(data: dict | list) -> np.ndarray:
    if isinstance(data, dict):
        return np.array(data.get("data", []), dtype=np.float32).reshape(data.get("rows", 0), data.get("cols", 0))
    return np.array(data, dtype=np.float32)


def load_calibration(path: str | Path) -> Optional[CameraCalibration]:
    """Load camera calibration from a JSON file.

    Supports either OpenCV YAML/JSON export (camera_matrix/dist_coeffs with data/rows/cols)
    or a compact format with raw arrays.
    """
    if not path:
        return None
    calib_path = Path(path)
    if not calib_path.exists():
        raise FileNotFoundError(f"Calibration file not found: {calib_path}")

    payload = json.loads(calib_path.read_text())
    camera_matrix = _parse_matrix(payload.get("camera_matrix", payload.get("K", [])))
    dist_coeffs = _parse_matrix(payload.get("dist_coeffs", payload.get("D", [])))

    if camera_matrix.size == 0 or dist_coeffs.size == 0:
        raise ValueError("Calibration file missing camera_matrix or dist_coeffs")

    return CameraCalibration(camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)


class Camera:
    def __init__(self, index: int, calibration: Optional[CameraCalibration] = None) -> None:
        self.index = index
        self.calibration = calibration
        self.cap = open_camera(index)

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        if not self.cap or not self.cap.isOpened():
            return False, None
        success, frame = self.cap.read()
        if not success or frame is None:
            return False, None
        if self.calibration is not None:
            frame = undistort(frame, self.calibration)
        return True, frame

    def release(self) -> None:
        if self.cap:
            self.cap.release()


def open_camera(index: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open camera index {index}")
    return cap


def undistort(frame: np.ndarray, calibration: CameraCalibration) -> np.ndarray:
    h, w = frame.shape[:2]
    new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
        calibration.camera_matrix, calibration.dist_coeffs, (w, h), 1, (w, h)
    )
    return cv2.undistort(frame, calibration.camera_matrix, calibration.dist_coeffs, None, new_camera_matrix)
