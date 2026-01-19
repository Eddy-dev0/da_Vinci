from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from robot_control.calibration import BrushSlot, Calibration, NamedPose, Pose

from .pose_mapping import PoseMapper
from .serial_handshake import SerialHandshake


@dataclass
class StrokeConfig:
    resample_mm: float = 3.0
    reload_paint_mm: Optional[float] = None
    dry_run: bool = False


class ExecutorActions:
    def __init__(
        self,
        calibration: Calibration,
        inventory: Dict[str, Any],
        mapper: PoseMapper,
        serial: SerialHandshake,
        logger: Optional[logging.Logger] = None,
        stroke_config: Optional[StrokeConfig] = None,
    ) -> None:
        self.calibration = calibration
        self.inventory = inventory
        self.mapper = mapper
        self.serial = serial
        self.logger = logger or logging.getLogger(__name__)
        self.stroke_config = stroke_config or StrokeConfig()

    def select_tool(self, tool_type: Optional[str], tool_id: Optional[str]) -> None:
        pick_pose = self._resolve_tool_pick_pose(tool_type, tool_id)
        if not pick_pose:
            self.logger.warning("No tool pick pose found for %s/%s", tool_type, tool_id)
            return
        self._move_safe()
        self._move_named_pose("tool_rack_pose")
        self._set_gripper(open_grip=True, reference_pose=pick_pose)
        self._move_pose(pick_pose)
        self._set_gripper(open_grip=False, reference_pose=pick_pose)
        self._move_named_pose("tool_rack_pose")
        self._move_safe()

    def load_paint(self, paint_id: Optional[str]) -> None:
        dip_pose = self._resolve_paint_dip_pose(paint_id)
        if not dip_pose:
            self.logger.warning("No paint dip pose for paint %s", paint_id)
            return
        self._move_safe()
        self._move_pose(dip_pose)
        self._dip_motion(dip_pose, offset_key="paint_dip_offset")
        self._move_named_pose("paint_wipe_pose")
        self._move_safe()

    def clean_tool(self) -> None:
        if not self.calibration.water_cleaning_station_pose:
            self.logger.warning("No water cleaning station pose configured")
            return
        self._move_safe()
        self._move_pose(self.calibration.water_cleaning_station_pose)
        self._dip_motion(self.calibration.water_cleaning_station_pose, offset_key="cleaning_dip_offset")
        if self.calibration.paper_towel_pose:
            self._move_pose(self.calibration.paper_towel_pose)
        self._move_safe()

    def stroke(self, points_mm: Sequence[Sequence[float]]) -> None:
        if not points_mm:
            return
        filtered = [
            (float(x), float(y))
            for x, y in points_mm
            if self.mapper.in_bounds(float(x), float(y))
        ]
        if not filtered:
            self.logger.warning("Stroke skipped: all points outside canvas bounds")
            return
        for x, y in points_mm:
            if not self.mapper.in_bounds(float(x), float(y)):
                self.logger.warning("Point out of bounds: %.2f, %.2f", x, y)
                break

        resampled = _resample_polyline(filtered, self.stroke_config.resample_mm)
        if not resampled:
            return
        first = resampled[0]
        self._move_pose(self.mapper.canvas_mm_to_servo_pose(*first, mode="hover"))
        if not self.stroke_config.dry_run:
            self._move_pose(self.mapper.canvas_mm_to_servo_pose(*first, mode="touch"))
        distance_since_reload = 0.0
        last = first
        for point in resampled[1:]:
            mode = "hover" if self.stroke_config.dry_run else "touch"
            self._move_pose(self.mapper.canvas_mm_to_servo_pose(*point, mode=mode))
            distance_since_reload += _distance_mm(last, point)
            last = point
            if (
                self.stroke_config.reload_paint_mm
                and distance_since_reload >= self.stroke_config.reload_paint_mm
            ):
                self.logger.info("Reloading paint mid-stroke after %.1fmm", distance_since_reload)
                self.load_paint(None)
                distance_since_reload = 0.0
        self._move_pose(self.mapper.canvas_mm_to_servo_pose(*resampled[-1], mode="hover"))

    def _move_pose(self, pose: Pose | Iterable[int]) -> None:
        if isinstance(pose, Pose):
            angles = pose.angles
        else:
            angles = list(pose)
        self.serial.send_frame(angles)

    def _move_named_pose(self, name: str) -> None:
        pose = self.calibration.named_poses.get(name)
        if pose:
            self._move_pose(pose)

    def _move_safe(self) -> None:
        if self.calibration.safe_travel_pose:
            self._move_pose(self.calibration.safe_travel_pose)

    def _dip_motion(self, base_pose: Pose, offset_key: str) -> None:
        offset = _resolve_offset(self.calibration.canvas, offset_key)
        if not offset:
            return
        down_pose = Pose(angles=[a + b for a, b in zip(base_pose.angles, offset)])
        self._move_pose(down_pose)
        self._move_pose(base_pose)

    def _set_gripper(self, open_grip: bool, reference_pose: Pose) -> None:
        angle = _resolve_gripper_angle(self.calibration.canvas, open_grip)
        if angle is None:
            pose_name = "gripper_open_pose" if open_grip else "gripper_closed_pose"
            named_pose = self.calibration.named_poses.get(pose_name)
            if named_pose:
                self._move_pose(named_pose)
                return
            self.logger.warning("No gripper angle configured")
            return
        angles = list(reference_pose.angles)
        angles[-1] = angle
        self._move_pose(angles)

    def _resolve_tool_pick_pose(self, tool_type: Optional[str], tool_id: Optional[str]) -> Optional[Pose]:
        if tool_id:
            for brush in self.calibration.brushes:
                if brush.name == tool_id or brush.name == f"{tool_type}:{tool_id}":
                    if brush.pick_pose:
                        return brush.pick_pose
        if tool_type:
            for brush in self.calibration.brushes:
                if brush.name == tool_type and brush.pick_pose:
                    return brush.pick_pose
        for entry in self.inventory.get("tools", []):
            if tool_id and str(entry.get("id")) != str(tool_id):
                continue
            if tool_type and str(entry.get("tool_type")) != str(tool_type):
                continue
            pose_name = entry.get("pickup_pose_name") or entry.get("pick_pose_name")
            if pose_name:
                pose = self.calibration.named_poses.get(pose_name)
                if pose:
                    return pose
        return None

    def _resolve_paint_dip_pose(self, paint_id: Optional[str]) -> Optional[Pose]:
        if paint_id:
            for paint in self.calibration.paints:
                if paint.name == paint_id:
                    return paint.pose
        for entry in self.inventory.get("paints", []):
            if paint_id and str(entry.get("id")) != str(paint_id):
                continue
            pose_name = entry.get("dip_pose_name") or entry.get("pickup_pose_name")
            if pose_name:
                pose = self.calibration.named_poses.get(pose_name)
                if pose:
                    return pose
        return None


def _resolve_offset(canvas: Dict[str, Any], key: str) -> Optional[List[int]]:
    raw = canvas.get(key)
    if raw is None:
        return None
    if isinstance(raw, list) and len(raw) == 7:
        return [int(round(value)) for value in raw]
    if isinstance(raw, (int, float)):
        offset = [0] * 7
        offset[2] = int(round(raw))
        return offset
    return None


def _resolve_gripper_angle(canvas: Dict[str, Any], open_grip: bool) -> Optional[int]:
    key = "gripper_open_angle" if open_grip else "gripper_closed_angle"
    raw = canvas.get(key)
    if raw is None:
        return None
    return int(round(raw))


def _resample_polyline(points: Sequence[Tuple[float, float]], spacing_mm: float) -> List[Tuple[float, float]]:
    if not points:
        return []
    if spacing_mm <= 0:
        return list(points)
    resampled = [points[0]]
    remaining = spacing_mm
    prev = points[0]
    for current in points[1:]:
        segment = _distance_mm(prev, current)
        if segment == 0:
            continue
        direction = ((current[0] - prev[0]) / segment, (current[1] - prev[1]) / segment)
        while segment >= remaining:
            new_point = (prev[0] + direction[0] * remaining, prev[1] + direction[1] * remaining)
            resampled.append(new_point)
            prev = new_point
            segment = _distance_mm(prev, current)
            remaining = spacing_mm
        remaining -= segment
        prev = current
    if resampled[-1] != points[-1]:
        resampled.append(points[-1])
    return resampled


def _distance_mm(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    return (dx * dx + dy * dy) ** 0.5
