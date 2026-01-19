from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from robot_control.calibration import Calibration, Pose, clamp_angles


@dataclass
class ServoLimit:
    minimum: int = 0
    maximum: int = 180

    def clamp(self, angle: float) -> int:
        return int(max(self.minimum, min(self.maximum, round(angle))))


@dataclass
class CanvasBounds:
    width_mm: float
    height_mm: float

    def normalize(self, x_mm: float, y_mm: float) -> Tuple[float, float]:
        if self.width_mm <= 0 or self.height_mm <= 0:
            raise ValueError("Canvas bounds must be positive.")
        u = max(0.0, min(1.0, x_mm / self.width_mm))
        v = max(0.0, min(1.0, y_mm / self.height_mm))
        return u, v


class PoseMapper:
    def __init__(
        self,
        calibration: Calibration,
        bounds: CanvasBounds,
        logger: Optional[logging.Logger] = None,
        servo_limits: Optional[Sequence[ServoLimit]] = None,
    ) -> None:
        self.calibration = calibration
        self.bounds = bounds
        self.logger = logger or logging.getLogger(__name__)
        self.servo_limits = list(servo_limits) if servo_limits else None

    @classmethod
    def from_files(
        cls,
        calibration_path: Path,
        canvas_bounds: CanvasBounds,
        logger: Optional[logging.Logger] = None,
    ) -> "PoseMapper":
        calibration = Calibration.load(calibration_path)
        servo_limits = _load_servo_limits(calibration.canvas)
        return cls(calibration, canvas_bounds, logger=logger, servo_limits=servo_limits)

    def canvas_mm_to_servo_pose(
        self,
        x_mm: float,
        y_mm: float,
        mode: str = "hover",
    ) -> List[int]:
        u, v = self.bounds.normalize(x_mm, y_mm)
        corners = self._get_canvas_corners(mode)
        angles = _bilinear_interpolate_angles(corners, u, v)
        return self._clamp_angles(angles)

    def in_bounds(self, x_mm: float, y_mm: float) -> bool:
        return 0.0 <= x_mm <= self.bounds.width_mm and 0.0 <= y_mm <= self.bounds.height_mm

    def _get_canvas_corners(self, mode: str) -> Dict[str, Pose]:
        mode = mode.lower().strip()
        if mode not in {"hover", "touch"}:
            raise ValueError("mode must be 'hover' or 'touch'.")

        corners = {}
        for key in ("tl", "tr", "br", "bl"):
            pose = None
            if mode == "touch":
                pose = self._get_touch_corner_pose(key)
            if pose is None:
                pose = self._get_hover_corner_pose(key)
            if pose is None:
                raise ValueError(f"Missing canvas corner pose: {key} ({mode})")
            corners[key] = pose
        return corners

    def _get_hover_corner_pose(self, key: str) -> Optional[Pose]:
        canvas = self.calibration.canvas
        pose = canvas.get(f"corner_{key}_pose")
        if isinstance(pose, Pose):
            return pose
        named = self.calibration.named_poses.get(f"corner_{key}_hover_pose")
        if named:
            return named
        return None

    def _get_touch_corner_pose(self, key: str) -> Optional[Pose]:
        canvas = self.calibration.canvas
        pose = canvas.get(f"corner_{key}_touch_pose") or canvas.get(f"corner_{key}_pose_touch")
        if isinstance(pose, Pose):
            return pose
        named = self.calibration.named_poses.get(f"corner_{key}_touch_pose")
        if named:
            return named
        hover = self._get_hover_corner_pose(key)
        if hover:
            offset = _resolve_touch_offset(canvas)
            if offset:
                return Pose(angles=[a + b for a, b in zip(hover.angles, offset)])
        return None

    def _clamp_angles(self, angles: Iterable[float]) -> List[int]:
        angle_list = list(angles)
        if self.servo_limits:
            return [limit.clamp(angle) for limit, angle in zip(self.servo_limits, angle_list)]
        return clamp_angles([int(round(angle)) for angle in angle_list])


def _bilinear_interpolate_angles(corners: Dict[str, Pose], u: float, v: float) -> List[float]:
    tl = corners["tl"].angles
    tr = corners["tr"].angles
    br = corners["br"].angles
    bl = corners["bl"].angles
    angles: List[float] = []
    for idx in range(7):
        a00 = tl[idx]
        a10 = tr[idx]
        a11 = br[idx]
        a01 = bl[idx]
        top = a00 * (1 - u) + a10 * u
        bottom = a01 * (1 - u) + a11 * u
        angles.append(top * (1 - v) + bottom * v)
    return angles


def _resolve_touch_offset(canvas: Dict[str, Any]) -> Optional[List[int]]:
    offset = canvas.get("touch_offset")
    if offset is None:
        offset = canvas.get("touch_height_offset")
    if offset is None:
        return None
    if isinstance(offset, list) and len(offset) == 7:
        return [int(round(value)) for value in offset]
    if isinstance(offset, dict):
        angles = [0] * 7
        for key, value in offset.items():
            try:
                idx = int(key)
            except (TypeError, ValueError):
                continue
            if 0 <= idx < 7:
                angles[idx] = int(round(value))
        return angles
    if isinstance(offset, (int, float)):
        angles = [0] * 7
        angles[2] = int(round(offset))
        return angles
    return None


def _load_servo_limits(canvas: Dict[str, Any]) -> Optional[List[ServoLimit]]:
    raw_limits = canvas.get("servo_limits")
    if not raw_limits:
        return None
    limits: List[ServoLimit] = []
    if isinstance(raw_limits, list):
        for entry in raw_limits[:7]:
            if isinstance(entry, dict):
                minimum = int(entry.get("min", 0))
                maximum = int(entry.get("max", 180))
            else:
                minimum, maximum = 0, 180
            limits.append(ServoLimit(minimum=minimum, maximum=maximum))
    if len(limits) != 7:
        return None
    return limits
