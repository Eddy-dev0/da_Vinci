"""Canvas detection using ArUco markers and homography estimation."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np

from vision.aruco_canvas import CanvasConfig, ObservationAccumulator, compute_homography, detect_markers, draw_debug_overlay
from vision.camera import Camera, load_calibration


@dataclass
class VisionSettings:
    camera_index: int
    canvas_width_mm: float
    canvas_height_mm: float
    marker_ids: Dict[str, int]
    output_path: Path
    camera_calibration: Optional[Path] = None
    frames: int = 20


def scan_canvas(settings: VisionSettings) -> Path:
    calibration = load_calibration(settings.camera_calibration) if settings.camera_calibration else None
    camera = Camera(settings.camera_index, calibration=calibration)
    accumulator = ObservationAccumulator()

    try:
        for _ in range(settings.frames):
            success, frame = camera.read_frame()
            if not success or frame is None:
                continue
            detection = detect_markers(frame)
            if detection.ids.size:
                accumulator.add(detection.corners, detection.ids)

        centers = accumulator.averaged_centers()
        config = CanvasConfig(
            width_mm=settings.canvas_width_mm,
            height_mm=settings.canvas_height_mm,
            marker_ids=settings.marker_ids,
        )
        homography = compute_homography(centers, config)
        if homography is None:
            raise RuntimeError("Unable to compute homography (missing markers).")

        payload = {
            "homography": homography.tolist(),
            "canvas_width_mm": settings.canvas_width_mm,
            "canvas_height_mm": settings.canvas_height_mm,
            "marker_ids": settings.marker_ids,
        }
        settings.output_path.write_text(json.dumps(payload, indent=2))

        debug_frame = draw_debug_overlay(frame, detection, homography, config)
        debug_path = settings.output_path.with_suffix(".debug.png")
        cv2.imwrite(str(debug_path), debug_frame)
        return settings.output_path
    finally:
        camera.release()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan canvas markers and compute homography.")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--canvas-width", type=float, required=True)
    parser.add_argument("--canvas-height", type=float, required=True)
    parser.add_argument("--marker-ids", default='{"tl":0,"tr":1,"br":2,"bl":3}')
    parser.add_argument("--output", default="canvas_homography.json")
    parser.add_argument("--camera-calibration")
    parser.add_argument("--frames", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    marker_ids = {key: int(value) for key, value in json.loads(args.marker_ids).items()}
    settings = VisionSettings(
        camera_index=args.camera,
        canvas_width_mm=args.canvas_width,
        canvas_height_mm=args.canvas_height,
        marker_ids=marker_ids,
        output_path=Path(args.output),
        camera_calibration=Path(args.camera_calibration) if args.camera_calibration else None,
        frames=args.frames,
    )
    output_path = scan_canvas(settings)
    print(f"Saved homography to {output_path}")


if __name__ == "__main__":
    main()
