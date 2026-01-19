"""Robot arm backend: execute PaintCode on an Arduino-controlled arm."""
from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from robot_control.serial_controller import RobotArmSerial, RobotArmSerialError

from painterslicer.machines.robotarm_backend.kinematics_planar import PlanarCalibration, PlanarKinematics
from painterslicer.machines.robotarm_backend.paintcode_parser import PaintCodeParser, PaintStep
from painterslicer.machines.robotarm_backend.preview import PreviewConfig, render_preview
from painterslicer.machines.robotarm_backend.toolchain import Toolchain

try:
    import yaml
except ImportError:  # pragma: no cover - optional dependency
    yaml = None


@dataclass
class BackendSettings:
    port: str = "COM3"
    baud: int = 9600
    speed: int = 40
    step_mm: float = 3.0
    timeout_s: float = 5.0
    retry_count: int = 3
    retry_delay_s: float = 0.5
    canvas_width_mm: float = 400.0
    canvas_height_mm: float = 300.0
    paintcode_normalized: bool = False
    servo_limits: Optional[List[Tuple[float, float]]] = None
    preview_pixels_per_mm: float = 2.0
    allow_out_of_bounds: bool = False


@dataclass
class ExecutionResult:
    points_sent: int
    tool_changes: int
    duration_s: float


class RobotArmBackend:
    def __init__(
        self,
        settings: BackendSettings,
        calibration_path: Path,
        inventory_path: Optional[Path] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.settings = settings
        self.calibration_path = calibration_path
        self.inventory_path = inventory_path
        self.logger = logger or logging.getLogger(__name__)
        self.toolchain = Toolchain(_load_json(inventory_path) if inventory_path else None)
        self.serial = RobotArmSerial(port=settings.port, baud=settings.baud, timeout_s=settings.timeout_s, logger=self.logger)

    def run_paintcode(
        self,
        paintcode: str | Path,
        dry_run: bool = False,
        preview_path: Optional[Path] = None,
    ) -> ExecutionResult:
        steps = self._load_steps(paintcode)
        calibration = _load_planar_calibration(self.calibration_path, self.settings)
        kinematics = PlanarKinematics(calibration, servo_limits=self.settings.servo_limits)

        if preview_path:
            preview_config = PreviewConfig(
                canvas_width_mm=self.settings.canvas_width_mm,
                canvas_height_mm=self.settings.canvas_height_mm,
                pixels_per_mm=self.settings.preview_pixels_per_mm,
            )
            render_preview(steps, preview_config, preview_path)

        if not dry_run:
            self._connect_robot()
            self.serial.set_speed(self.settings.speed)

        current_pos: Optional[Tuple[float, float]] = None
        z_state = "hover"
        pressure = 0.0
        tool_changes = 0
        points_sent = 0
        start_time = time.monotonic()

        for step in steps:
            if step.command == "TOOL":
                tool = PaintCodeParser.tool_value(step)
                if tool:
                    self.toolchain.select_tool(tool)
                    tool_changes += 1
                continue
            if step.command == "PRESSURE":
                value = PaintCodeParser.pressure_value(step)
                if value is not None:
                    pressure = max(0.0, min(1.0, value))
                    z_state = "pressure"
                continue
            if step.command == "Z_UP":
                z_state = "hover"
                continue
            if step.command == "Z_DOWN":
                z_state = "touch"
                continue
            if step.command == "Z":
                value = PaintCodeParser.z_value(step)
                if value is not None:
                    pressure = max(0.0, min(1.0, value))
                    z_state = "pressure"
                continue
            if step.command in {"CLEAN", "WASH_STATION", "WIPE"}:
                self.toolchain.clean_tool()
                continue
            if step.command != "MOVE":
                continue

            x, y = PaintCodeParser.move_args(step)
            x_mm, y_mm = self._convert_coords(x, y)
            self._validate_bounds(x_mm, y_mm)
            if current_pos is None:
                points = [(x_mm, y_mm)]
            else:
                points = _resample_line(current_pos, (x_mm, y_mm), self.settings.step_mm)
            current_pos = (x_mm, y_mm)
            for point in points:
                pose = kinematics.canvas_to_servo_pose(point[0], point[1], z_state=z_state, pressure=pressure)
                points_sent += 1
                if not dry_run:
                    self._send_frame_with_retry(pose)

        duration = time.monotonic() - start_time
        if not dry_run:
            self.serial.close()
        return ExecutionResult(points_sent=points_sent, tool_changes=tool_changes, duration_s=duration)

    def _load_steps(self, paintcode: str | Path) -> List[PaintStep]:
        if isinstance(paintcode, Path):
            text = paintcode.read_text(encoding="utf-8")
        else:
            text = paintcode
        return PaintCodeParser(text).parse()

    def _convert_coords(self, x: float, y: float) -> Tuple[float, float]:
        if self.settings.paintcode_normalized or (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            return x * self.settings.canvas_width_mm, y * self.settings.canvas_height_mm
        return x, y

    def _validate_bounds(self, x: float, y: float) -> None:
        if self.settings.allow_out_of_bounds:
            return
        if not (0.0 <= x <= self.settings.canvas_width_mm and 0.0 <= y <= self.settings.canvas_height_mm):
            raise ValueError(f"MOVE out of bounds: ({x:.2f}, {y:.2f})")

    def _connect_robot(self) -> None:
        try:
            self.serial.connect()
        except Exception as exc:
            raise RobotArmSerialError(f"Failed to connect to {self.settings.port}: {exc}") from exc

    def _send_frame_with_retry(self, pose: Iterable[int]) -> None:
        for attempt in range(1, self.settings.retry_count + 1):
            try:
                self.serial.send_frame(pose)
                return
            except RobotArmSerialError as exc:
                if attempt >= self.settings.retry_count:
                    raise
                self.logger.warning("Retry %s/%s after error: %s", attempt, self.settings.retry_count, exc)
                time.sleep(self.settings.retry_delay_s)


def _load_planar_calibration(path: Path, settings: BackendSettings) -> PlanarCalibration:
    data = _load_json(path)
    if "corner_tl_hover" in data:
        return PlanarCalibration(
            canvas_width_mm=data.get("canvas_width_mm", settings.canvas_width_mm),
            canvas_height_mm=data.get("canvas_height_mm", settings.canvas_height_mm),
            corner_tl_hover=data["corner_tl_hover"],
            corner_tr_hover=data["corner_tr_hover"],
            corner_br_hover=data["corner_br_hover"],
            corner_bl_hover=data["corner_bl_hover"],
            corner_tl_touch=data["corner_tl_touch"],
            corner_tr_touch=data["corner_tr_touch"],
            corner_br_touch=data["corner_br_touch"],
            corner_bl_touch=data["corner_bl_touch"],
        )

    canvas = (data.get("canvas") or {})
    hover = {
        "tl": _pose_from_canvas(canvas, "corner_tl_pose"),
        "tr": _pose_from_canvas(canvas, "corner_tr_pose"),
        "br": _pose_from_canvas(canvas, "corner_br_pose"),
        "bl": _pose_from_canvas(canvas, "corner_bl_pose"),
    }
    touch_offset = canvas.get("touch_height_offset") or [0, 0, 0, 0, 0, 0, 0]
    touch = {key: [angle + offset for angle, offset in zip(pose, touch_offset)] for key, pose in hover.items()}

    return PlanarCalibration(
        canvas_width_mm=data.get("canvas_width_mm", settings.canvas_width_mm),
        canvas_height_mm=data.get("canvas_height_mm", settings.canvas_height_mm),
        corner_tl_hover=hover["tl"],
        corner_tr_hover=hover["tr"],
        corner_br_hover=hover["br"],
        corner_bl_hover=hover["bl"],
        corner_tl_touch=touch["tl"],
        corner_tr_touch=touch["tr"],
        corner_br_touch=touch["br"],
        corner_bl_touch=touch["bl"],
    )


def _pose_from_canvas(canvas: Dict[str, Any], key: str) -> List[float]:
    pose = canvas.get(key)
    if isinstance(pose, dict):
        pose = pose.get("angles")
    if not pose or len(pose) != 7:
        raise ValueError(f"Calibration missing {key} pose")
    return [float(angle) for angle in pose]


def _resample_line(start: Tuple[float, float], end: Tuple[float, float], step_mm: float) -> List[Tuple[float, float]]:
    sx, sy = start
    ex, ey = end
    dx = ex - sx
    dy = ey - sy
    distance = math.hypot(dx, dy)
    if distance <= 0:
        return [(ex, ey)]
    steps = max(1, int(distance / step_mm))
    return [
        (sx + dx * t / steps, sy + dy * t / steps)
        for t in range(1, steps + 1)
    ]


def _load_json(path: Optional[Path]) -> Dict[str, Any]:
    if not path:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"Calibration/inventory file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_settings(path: Path) -> BackendSettings:
    if not path.exists():
        return BackendSettings()
    raw = path.read_text(encoding="utf-8")
    if yaml is None:
        raise RuntimeError("PyYAML is required to load settings.yaml")
    data = yaml.safe_load(raw) or {}
    return BackendSettings(**data.get("robotarm", data))
