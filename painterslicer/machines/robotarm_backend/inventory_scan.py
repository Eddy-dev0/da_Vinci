"""Inventory scanning for tools and paint pots using ArUco markers."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from vision.aruco_canvas import DetectionResult, detect_markers
from vision.camera import Camera, load_calibration


@dataclass
class InventoryScanSettings:
    camera_index: int
    output_path: Path
    marker_map_path: Optional[Path] = None
    camera_calibration: Optional[Path] = None
    frames_per_pose: int = 5


def scan_inventory(settings: InventoryScanSettings) -> Dict[str, Any]:
    marker_map = _load_marker_map(settings.marker_map_path)
    camera = Camera(settings.camera_index, calibration=load_calibration(settings.camera_calibration) if settings.camera_calibration else None)
    observations: Dict[int, List[np.ndarray]] = {}
    colors: Dict[int, List[np.ndarray]] = {}

    try:
        for _ in range(settings.frames_per_pose):
            success, frame = camera.read_frame()
            if not success or frame is None:
                continue
            detection = detect_markers(frame)
            _accumulate(detection, frame, observations, colors)
    finally:
        camera.release()

    inventory = _build_inventory(observations, colors, marker_map)
    settings.output_path.write_text(json.dumps(inventory, indent=2))
    return inventory


def _accumulate(
    detection: DetectionResult,
    frame: np.ndarray,
    observations: Dict[int, List[np.ndarray]],
    colors: Dict[int, List[np.ndarray]],
) -> None:
    if detection.ids.size == 0:
        return
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lab_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    for marker_id, corners in zip(detection.ids.flatten(), detection.corners):
        marker_id = int(marker_id)
        center = corners.reshape(4, 2).mean(axis=0)
        observations.setdefault(marker_id, []).append(center)
        sample = _sample_color(hsv_frame, lab_frame, center)
        if sample:
            colors.setdefault(marker_id, []).append(sample)


def _sample_color(
    hsv_frame: np.ndarray,
    lab_frame: np.ndarray,
    center: np.ndarray,
    size: int = 10,
) -> Optional[np.ndarray]:
    x, y = int(center[0]), int(center[1])
    h, w = hsv_frame.shape[:2]
    x1 = max(0, x - size)
    x2 = min(w, x + size)
    y1 = max(0, y - size)
    y2 = min(h, y + size)
    if x1 >= x2 or y1 >= y2:
        return None
    hsv_roi = hsv_frame[y1:y2, x1:x2]
    lab_roi = lab_frame[y1:y2, x1:x2]
    hsv = np.median(hsv_roi.reshape(-1, 3), axis=0)
    lab = np.median(lab_roi.reshape(-1, 3), axis=0)
    return np.concatenate([hsv, lab])


def _load_marker_map(path: Optional[Path]) -> Dict[int, Dict[str, Any]]:
    if not path:
        return {}
    return {int(item["id"]): item for item in json.loads(path.read_text(encoding="utf-8")).get("markers", [])}


def _build_inventory(
    observations: Dict[int, List[np.ndarray]],
    colors: Dict[int, List[np.ndarray]],
    marker_map: Dict[int, Dict[str, Any]],
) -> Dict[str, Any]:
    paints: List[Dict[str, Any]] = []
    tools: List[Dict[str, Any]] = []

    for marker_id, centers in observations.items():
        meta = marker_map.get(marker_id, {})
        category = meta.get("category", "paint")
        center = np.mean(np.stack(centers, axis=0), axis=0)
        color_samples = colors.get(marker_id, [])
        hsv_lab = np.median(np.stack(color_samples, axis=0), axis=0) if color_samples else None
        entry = {
            "id": marker_id,
            "name": meta.get("name", f"marker_{marker_id}"),
            "pixel": [float(center[0]), float(center[1])],
        }
        if hsv_lab is not None:
            entry["hsv"] = [float(v) for v in hsv_lab[:3]]
            entry["lab"] = [float(v) for v in hsv_lab[3:]]

        if category == "tool":
            entry.update(
                {
                    "tool_type": meta.get("tool_type", "ROUND"),
                    "size_mm": meta.get("size_mm", 10.0),
                    "pickup_pose_name": meta.get("pickup_pose_name"),
                    "place_pose_name": meta.get("place_pose_name"),
                }
            )
            tools.append(entry)
        else:
            entry.update(
                {
                    "dip_pose_name": meta.get("dip_pose_name"),
                    "wipe_pose_name": meta.get("wipe_pose_name"),
                }
            )
            paints.append(entry)

    return {"paints": paints, "tools": tools}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan inventory markers for paints/tools.")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--output", default="inventory.json")
    parser.add_argument("--marker-map", help="JSON with marker metadata")
    parser.add_argument("--camera-calibration")
    parser.add_argument("--frames-per-pose", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    settings = InventoryScanSettings(
        camera_index=args.camera,
        output_path=Path(args.output),
        marker_map_path=Path(args.marker_map) if args.marker_map else None,
        camera_calibration=Path(args.camera_calibration) if args.camera_calibration else None,
        frames_per_pose=args.frames_per_pose,
    )
    inventory = scan_inventory(settings)
    print(json.dumps(inventory, indent=2))


if __name__ == "__main__":
    main()
