"""Planar bilinear kinematics mapping for the robot arm."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence


@dataclass
class PlanarCalibration:
    canvas_width_mm: float
    canvas_height_mm: float
    corner_tl_hover: Sequence[float]
    corner_tr_hover: Sequence[float]
    corner_br_hover: Sequence[float]
    corner_bl_hover: Sequence[float]
    corner_tl_touch: Sequence[float]
    corner_tr_touch: Sequence[float]
    corner_br_touch: Sequence[float]
    corner_bl_touch: Sequence[float]


@dataclass
class PlanarKinematics:
    calibration: PlanarCalibration
    servo_limits: Sequence[tuple[float, float]] | None = None

    def canvas_to_servo_pose(
        self,
        x_mm: float,
        y_mm: float,
        z_state: str = "hover",
        pressure: float = 0.0,
    ) -> List[int]:
        width = self.calibration.canvas_width_mm
        height = self.calibration.canvas_height_mm
        if width <= 0 or height <= 0:
            raise ValueError("Canvas dimensions must be positive")

        u = max(0.0, min(1.0, x_mm / width))
        v = max(0.0, min(1.0, y_mm / height))

        hover_pose = _bilinear(
            u,
            v,
            self.calibration.corner_tl_hover,
            self.calibration.corner_tr_hover,
            self.calibration.corner_br_hover,
            self.calibration.corner_bl_hover,
        )
        touch_pose = _bilinear(
            u,
            v,
            self.calibration.corner_tl_touch,
            self.calibration.corner_tr_touch,
            self.calibration.corner_br_touch,
            self.calibration.corner_bl_touch,
        )

        if z_state == "hover":
            pose = hover_pose
        elif z_state == "touch":
            pose = touch_pose
        elif z_state == "pressure":
            pressure = max(0.0, min(1.0, pressure))
            pose = _lerp(hover_pose, touch_pose, pressure)
        else:
            raise ValueError(f"Unknown z_state: {z_state}")

        pose = [max(0.0, min(180.0, value)) for value in pose]
        if self.servo_limits:
            pose = [
                max(limit[0], min(limit[1], value))
                for value, limit in zip(pose, self.servo_limits)
            ]
        return [int(round(value)) for value in pose]


def _bilinear(
    u: float,
    v: float,
    tl: Sequence[float],
    tr: Sequence[float],
    br: Sequence[float],
    bl: Sequence[float],
) -> List[float]:
    tl_list = list(tl)
    tr_list = list(tr)
    br_list = list(br)
    bl_list = list(bl)
    if not (len(tl_list) == len(tr_list) == len(br_list) == len(bl_list) == 7):
        raise ValueError("Each corner pose must have 7 servo angles")
    values: List[float] = []
    for idx in range(7):
        value = (
            (1 - u) * (1 - v) * tl_list[idx]
            + u * (1 - v) * tr_list[idx]
            + u * v * br_list[idx]
            + (1 - u) * v * bl_list[idx]
        )
        values.append(value)
    return values


def _lerp(a: Iterable[float], b: Iterable[float], t: float) -> List[float]:
    return [float(av + (bv - av) * t) for av, bv in zip(a, b)]
