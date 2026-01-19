from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from robot_control.calibration import Calibration
from robot_control.serial_controller import RobotArmSerialError, RobotArmTimeoutError

from .actions import ExecutorActions, StrokeConfig
from .pose_mapping import CanvasBounds, PoseMapper
from .serial_handshake import connect_controller


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute stroke_plan.json with a 7-servo arm")
    parser.add_argument("--port", required=True, help="Serial port (e.g., COM3 or /dev/ttyUSB0)")
    parser.add_argument("--baud", type=int, default=9600, help="Serial baud rate")
    parser.add_argument("--speed", type=int, default=30, help="Speed 0..100")
    parser.add_argument("--max-speed", type=int, default=40, help="Safety max speed 0..100")
    parser.add_argument("--timeout", type=float, default=8.0, help="DONE timeout in seconds")
    parser.add_argument("--calibration", default="calibration.json", help="Path to calibration.json")
    parser.add_argument("--inventory", default="inventory.json", help="Path to inventory.json")
    parser.add_argument("--plan", default="stroke_plan.json", help="Path to stroke_plan.json")
    parser.add_argument("--resample-mm", type=float, default=3.0, help="Resample spacing in mm")
    parser.add_argument("--dry-run", action="store_true", help="Hover only, never touch canvas")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    return parser.parse_args()


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _canvas_bounds(plan: Dict[str, Any], calibration: Calibration) -> CanvasBounds:
    meta = plan.get("meta", {})
    canvas_mm = meta.get("canvas_mm", {})
    width = canvas_mm.get("width")
    height = canvas_mm.get("height")
    if not width or not height:
        canvas_cfg = calibration.canvas
        width = width or canvas_cfg.get("width_mm") or canvas_cfg.get("width")
        height = height or canvas_cfg.get("height_mm") or canvas_cfg.get("height")
    if not width or not height:
        raise ValueError("Canvas bounds missing from plan meta or calibration")
    return CanvasBounds(width_mm=float(width), height_mm=float(height))


def _execute_steps(actions: ExecutorActions, plan: Dict[str, Any]) -> None:
    steps = plan.get("steps") or []
    for idx, step in enumerate(steps):
        step_type = str(step.get("type", "")).upper()
        if step_type == "SELECT_TOOL":
            actions.select_tool(step.get("tool_type"), step.get("tool_id"))
        elif step_type == "LOAD_PAINT":
            actions.load_paint(step.get("paint_id"))
        elif step_type == "CLEAN_TOOL":
            actions.clean_tool()
        elif step_type == "STROKE":
            points = step.get("points_mm") or step.get("points") or []
            actions.stroke(points)
        else:
            actions.logger.warning("Unknown step %s at index %d", step_type, idx)


def main() -> int:
    args = _parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    logger = logging.getLogger("executor.run")

    calibration_path = Path(args.calibration)
    inventory_path = Path(args.inventory)
    plan_path = Path(args.plan)

    calibration = Calibration.load(calibration_path)
    inventory = _load_json(inventory_path)
    plan = _load_json(plan_path)

    bounds = _canvas_bounds(plan, calibration)
    mapper = PoseMapper(calibration, bounds, logger=logger)

    speed = min(args.speed, args.max_speed) if args.max_speed is not None else args.speed
    serial = connect_controller(args.port, args.baud, args.timeout, speed, logger)

    reload_distance = calibration.canvas.get("paint_reload_mm")
    stroke_cfg = StrokeConfig(
        resample_mm=args.resample_mm,
        reload_paint_mm=float(reload_distance) if reload_distance else None,
        dry_run=args.dry_run,
    )
    actions = ExecutorActions(
        calibration=calibration,
        inventory=inventory,
        mapper=mapper,
        serial=serial,
        logger=logger,
        stroke_config=stroke_cfg,
    )

    try:
        _execute_steps(actions, plan)
    except RobotArmTimeoutError:
        logger.error("Timeout while executing plan, aborting")
        return 2
    except RobotArmSerialError as exc:
        logger.error("Serial error while executing plan: %s", exc)
        return 1
    finally:
        serial.controller.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
