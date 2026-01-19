"""Inventory scanning pipeline for paint pots and tools."""
from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from paint_color_estimation import apply_white_balance, bbox_from_corners, roi_sample
from robot_control.calibration import Calibration, NamedPose, Pose
from vision.aruco_canvas import CanvasConfig, compute_homography, detect_markers, pixel_to_canvas
from vision.camera import Camera, CameraCalibration, load_calibration

try:
    from robot_control.serial_controller import RobotArmSerial
except ImportError:
    RobotArmSerial = None


@dataclass
class MarkerMetadata:
    category: str
    name: Optional[str] = None
    tool_type: Optional[str] = None
    size_mm: Optional[float] = None
    pickup_pose_name: Optional[str] = None
    dip_pose_name: Optional[str] = None
    rack_slot: Optional[str] = None


@dataclass
class MarkerObservation:
    centers: List[np.ndarray]
    hsv_samples: List[Tuple[float, float, float]]
    rgb_samples: List[Tuple[int, int, int]]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inventory scan using ArUco markers and color sampling.")
    parser.add_argument("--calibration", default="calibration.json", help="Path to calibration.json")
    parser.add_argument("--camera-index", type=int, default=0, help="Camera index")
    parser.add_argument("--camera-calibration", help="Path to camera calibration JSON")
    parser.add_argument("--frames-per-pose", type=int, default=5, help="Frames to capture per scan pose")
    parser.add_argument("--wait-after-move", type=float, default=0.5, help="Seconds to wait after moving")
    parser.add_argument("--use-robot", action="store_true", help="Send scan poses to the robot arm")
    parser.add_argument("--robot-port", help="Serial port for robot arm")
    parser.add_argument("--robot-baud", type=int, default=9600, help="Serial baud")
    parser.add_argument("--robot-speed", type=int, default=40, help="Robot speed 0..100")
    parser.add_argument("--marker-map", help="Optional JSON mapping marker IDs to inventory metadata")
    parser.add_argument("--default-category", choices=["paint", "tool", "unknown"], default="paint")
    parser.add_argument("--canvas-width-mm", type=float, help="Canvas width in mm")
    parser.add_argument("--canvas-height-mm", type=float, help="Canvas height in mm")
    parser.add_argument("--canvas-marker-ids", help="JSON mapping for canvas markers: {""tl"":0,""tr"":1,...}")
    parser.add_argument("--white-balance-gray-roi", help="Gray reference ROI as x1,y1,x2,y2")
    parser.add_argument("--fallback-detection", action="store_true", help="Enable fallback detection without ArUco")
    parser.add_argument("--output", default="inventory.json", help="Output inventory JSON")
    return parser.parse_args()


def _load_marker_map(path: Optional[str]) -> Dict[int, MarkerMetadata]:
    if not path:
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    mapping: Dict[int, MarkerMetadata] = {}
    for entry in payload.get("paints", []):
        mapping[int(entry["id"])] = MarkerMetadata(
            category="paint",
            name=entry.get("name"),
            pickup_pose_name=entry.get("pickup_pose_name"),
            dip_pose_name=entry.get("dip_pose_name"),
        )
    for entry in payload.get("tools", []):
        mapping[int(entry["id"])] = MarkerMetadata(
            category="tool",
            name=entry.get("name"),
            tool_type=entry.get("tool_type"),
            size_mm=entry.get("size_mm"),
            pickup_pose_name=entry.get("pickup_pose_name"),
            rack_slot=entry.get("rack_slot"),
        )
    return mapping


def _parse_gray_roi(raw: Optional[str]) -> Optional[Tuple[int, int, int, int]]:
    if not raw:
        return None
    parts = [int(p.strip()) for p in raw.split(",") if p.strip()]
    if len(parts) != 4:
        raise ValueError("Gray ROI must be x1,y1,x2,y2")
    return parts[0], parts[1], parts[2], parts[3]


def _parse_canvas_marker_ids(raw: Optional[str]) -> Optional[Dict[str, int]]:
    if not raw:
        return None
    payload = json.loads(raw)
    return {str(key): int(value) for key, value in payload.items()}


def _sharpness(frame: np.ndarray) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _capture_best_frame(camera: Camera, frames: int) -> np.ndarray:
    best_frame = None
    best_score = -1.0
    for _ in range(frames):
        success, frame = camera.read_frame()
        if not success or frame is None:
            continue
        score = _sharpness(frame)
        if score > best_score:
            best_score = score
            best_frame = frame
    if best_frame is None:
        raise RuntimeError("Unable to capture frames from camera")
    return best_frame


def _move_robot(pose: Pose, controller: RobotArmSerial, speed: Optional[int]) -> None:
    if speed is not None:
        controller.set_speed(speed)
    controller.send_frame(pose.angles)


def _fallback_palette(frame: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
    small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    pixels = small.reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.2)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
    palette = []
    for idx, center in enumerate(centers):
        mask = labels.flatten() == idx
        if not np.any(mask):
            continue
        indices = np.where(mask)[0]
        coords = np.column_stack(np.unravel_index(indices, (small.shape[0], small.shape[1])))
        center_yx = coords.mean(axis=0)
        bgr = center.tolist()
        rgb = [int(bgr[2]), int(bgr[1]), int(bgr[0])]
        hsv = cv2.cvtColor(np.array([[bgr]], dtype=np.uint8), cv2.COLOR_BGR2HSV)[0, 0]
        palette.append(
            {
                "id": f"cluster_{idx}",
                "name": f"cluster_{idx}",
                "hsv": [float(hsv[0]), float(hsv[1]), float(hsv[2])],
                "rgb": rgb,
                "pixel": [float(center_yx[1] * 4), float(center_yx[0] * 4)],
            }
        )
    return palette


def _fallback_tools(frame: np.ndarray) -> List[Dict[str, Any]]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blur, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    tools = []
    for idx, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area < 500:
            continue
        rect = cv2.minAreaRect(contour)
        (cx, cy), (w, h), _ = rect
        if min(w, h) == 0:
            continue
        aspect = max(w, h) / min(w, h)
        tool_type = "BRUSH" if aspect > 3 else "TOOL"
        tools.append({"id": f"tool_{idx}", "tool_type": tool_type, "pixel": [float(cx), float(cy)]})
    return tools


def _build_inventory(
    observations: Dict[int, MarkerObservation],
    marker_map: Dict[int, MarkerMetadata],
    default_category: str,
    homography: Optional[np.ndarray],
) -> Dict[str, Any]:
    paints: List[Dict[str, Any]] = []
    tools: List[Dict[str, Any]] = []

    for marker_id, obs in observations.items():
        center = np.mean(np.stack(obs.centers, axis=0), axis=0)
        hsv = np.median(np.array(obs.hsv_samples), axis=0) if obs.hsv_samples else None
        rgb = np.median(np.array(obs.rgb_samples), axis=0) if obs.rgb_samples else None
        position = {"pixel": [float(center[0]), float(center[1])]}
        if homography is not None:
            canvas = pixel_to_canvas((float(center[0]), float(center[1])), homography)
            position["canvas_mm"] = [float(canvas[0]), float(canvas[1])]

        metadata = marker_map.get(marker_id)
        category = metadata.category if metadata else default_category

        if category == "paint":
            payload: Dict[str, Any] = {
                "id": int(marker_id),
                "name": metadata.name if metadata and metadata.name else f"paint_{marker_id}",
                **position,
            }
            if hsv is not None:
                payload["hsv"] = [float(hsv[0]), float(hsv[1]), float(hsv[2])]
            if rgb is not None:
                payload["rgb"] = [int(rgb[0]), int(rgb[1]), int(rgb[2])]
            if metadata:
                if metadata.pickup_pose_name:
                    payload["pickup_pose_name"] = metadata.pickup_pose_name
                if metadata.dip_pose_name:
                    payload["dip_pose_name"] = metadata.dip_pose_name
            paints.append(payload)
        elif category == "tool":
            payload = {
                "id": int(marker_id),
                "tool_type": metadata.tool_type if metadata and metadata.tool_type else "UNKNOWN",
                **position,
            }
            if metadata:
                if metadata.size_mm is not None:
                    payload["size_mm"] = metadata.size_mm
                if metadata.pickup_pose_name:
                    payload["pickup_pose_name"] = metadata.pickup_pose_name
                if metadata.rack_slot:
                    payload["rack_slot"] = metadata.rack_slot
            tools.append(payload)

    return {"paints": paints, "tools": tools}


def _scan_pose_names(calibration: Calibration) -> List[NamedPose]:
    return calibration.scan_poses


def main() -> int:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger("inventory_scan")

    calibration = Calibration.load(Path(args.calibration))
    marker_map = _load_marker_map(args.marker_map)

    camera_calibration: Optional[CameraCalibration] = None
    if args.camera_calibration:
        camera_calibration = load_calibration(args.camera_calibration)

    gray_roi = _parse_gray_roi(args.white_balance_gray_roi)

    canvas_config = None
    homography = None
    if args.canvas_width_mm and args.canvas_height_mm:
        canvas_config = CanvasConfig(
            width_mm=args.canvas_width_mm,
            height_mm=args.canvas_height_mm,
            marker_ids=_parse_canvas_marker_ids(args.canvas_marker_ids),
        )

    controller = None
    if args.use_robot:
        if RobotArmSerial is None:
            raise RuntimeError("robot_control.serial_controller is unavailable")
        if not args.robot_port:
            raise ValueError("--robot-port is required when --use-robot is set")
        controller = RobotArmSerial(port=args.robot_port, baud=args.robot_baud)
        controller.connect()

    observations: Dict[int, MarkerObservation] = {}

    camera = Camera(args.camera_index, calibration=camera_calibration)
    try:
        for scan_pose in _scan_pose_names(calibration):
            logger.info("Scanning pose %s", scan_pose.name)
            if controller is not None:
                _move_robot(scan_pose.pose, controller, args.robot_speed)
                time.sleep(args.wait_after_move)

            frame = _capture_best_frame(camera, args.frames_per_pose)
            if gray_roi is not None:
                frame = apply_white_balance(frame, gray_roi)

            detection = detect_markers(frame)
            if detection.ids.size == 0:
                logger.warning("No markers detected at pose %s", scan_pose.name)
                if args.fallback_detection:
                    fallback = _fallback_palette(frame)
                    tools = _fallback_tools(frame)
                    payload = {"paints": fallback, "tools": tools}
                    Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
                    return 0
                continue

            if canvas_config is not None:
                centers = {
                    int(marker_id): corners.reshape(4, 2).mean(axis=0)
                    for marker_id, corners in zip(detection.ids.flatten(), detection.corners)
                }
                homography = compute_homography(centers, canvas_config)

            for marker_id, corners in zip(detection.ids.flatten(), detection.corners):
                bbox = bbox_from_corners(corners)
                sample = roi_sample(frame, bbox)
                obs = observations.setdefault(
                    int(marker_id),
                    MarkerObservation(centers=[], hsv_samples=[], rgb_samples=[]),
                )
                obs.centers.append(corners.reshape(4, 2).mean(axis=0))
                obs.hsv_samples.append(sample.hsv)
                obs.rgb_samples.append(sample.rgb)

        inventory = _build_inventory(observations, marker_map, args.default_category, homography)
        Path(args.output).write_text(json.dumps(inventory, indent=2), encoding="utf-8")
        logger.info("Inventory written to %s", args.output)
    finally:
        camera.release()
        if controller is not None:
            controller.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
