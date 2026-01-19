"""ArUco-based canvas detection and homography estimation."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from vision.camera import Camera, CameraCalibration, load_calibration


@dataclass
class CanvasConfig:
    width_mm: Optional[float] = None
    height_mm: Optional[float] = None
    marker_length_mm: Optional[float] = None
    marker_ids: Dict[str, int] | None = None


@dataclass
class DetectionResult:
    corners: List[np.ndarray]
    ids: np.ndarray


class ObservationAccumulator:
    def __init__(self) -> None:
        self.marker_centers: Dict[int, List[np.ndarray]] = {}
        self.marker_corners: Dict[int, List[np.ndarray]] = {}

    def add(self, corners: List[np.ndarray], ids: np.ndarray) -> None:
        for marker_id, marker_corners in zip(ids.flatten(), corners):
            center = marker_corners.reshape(4, 2).mean(axis=0)
            self.marker_centers.setdefault(int(marker_id), []).append(center)
            self.marker_corners.setdefault(int(marker_id), []).append(marker_corners)

    def averaged_centers(self) -> Dict[int, np.ndarray]:
        return {
            marker_id: np.mean(np.stack(samples, axis=0), axis=0)
            for marker_id, samples in self.marker_centers.items()
        }


DEFAULT_MARKER_IDS = {"tl": 0, "tr": 1, "br": 2, "bl": 3}


def detect_markers(frame: np.ndarray, dictionary: int = cv2.aruco.DICT_4X4_50) -> DetectionResult:
    aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    corners, ids, _ = detector.detectMarkers(frame)
    if ids is None:
        ids = np.array([], dtype=np.int32)
    return DetectionResult(corners=corners, ids=ids)


def estimate_canvas_size_from_markers(
    corners: List[np.ndarray],
    ids: np.ndarray,
    config: CanvasConfig,
    calibration: CameraCalibration,
) -> Optional[Tuple[float, float]]:
    if not config.marker_length_mm:
        return None
    if ids.size < 4:
        return None

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
        corners, config.marker_length_mm, calibration.camera_matrix, calibration.dist_coeffs
    )
    marker_positions = {int(marker_id): tvec.squeeze() for marker_id, tvec in zip(ids.flatten(), tvecs)}

    required_ids = config.marker_ids or DEFAULT_MARKER_IDS
    try:
        tl = marker_positions[required_ids["tl"]]
        tr = marker_positions[required_ids["tr"]]
        br = marker_positions[required_ids["br"]]
        bl = marker_positions[required_ids["bl"]]
    except KeyError:
        return None

    width = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2.0
    height = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2.0
    return float(width), float(height)


def compute_homography(
    marker_centers: Dict[int, np.ndarray],
    config: CanvasConfig,
) -> Optional[np.ndarray]:
    marker_ids = config.marker_ids or DEFAULT_MARKER_IDS
    required_keys = ["tl", "tr", "br", "bl"]
    if any(key not in marker_ids for key in required_keys):
        raise ValueError("marker_ids must contain tl/tr/br/bl")

    try:
        pixel_points = np.array(
            [
                marker_centers[marker_ids["tl"]],
                marker_centers[marker_ids["tr"]],
                marker_centers[marker_ids["br"]],
                marker_centers[marker_ids["bl"]],
            ],
            dtype=np.float32,
        )
    except KeyError:
        return None

    if config.width_mm is None or config.height_mm is None:
        return None

    canvas_points = np.array(
        [
            [0.0, 0.0],
            [config.width_mm, 0.0],
            [config.width_mm, config.height_mm],
            [0.0, config.height_mm],
        ],
        dtype=np.float32,
    )

    homography, _ = cv2.findHomography(pixel_points, canvas_points)
    return homography


def pixel_to_canvas(point: Tuple[float, float], homography: np.ndarray) -> Tuple[float, float]:
    src = np.array([[point]], dtype=np.float32)
    dst = cv2.perspectiveTransform(src, homography)
    return float(dst[0, 0, 0]), float(dst[0, 0, 1])


def canvas_to_pixel(point: Tuple[float, float], homography: np.ndarray) -> Tuple[float, float]:
    inv = np.linalg.inv(homography)
    return pixel_to_canvas(point, inv)


def draw_debug_overlay(
    frame: np.ndarray,
    detection: DetectionResult,
    homography: Optional[np.ndarray],
    config: CanvasConfig,
) -> np.ndarray:
    output = frame.copy()
    if detection.ids.size:
        cv2.aruco.drawDetectedMarkers(output, detection.corners, detection.ids)

    if homography is not None and config.width_mm and config.height_mm:
        canvas_rect = np.array(
            [
                [0.0, 0.0],
                [config.width_mm, 0.0],
                [config.width_mm, config.height_mm],
                [0.0, config.height_mm],
            ],
            dtype=np.float32,
        )
        inv_h = np.linalg.inv(homography)
        pixel_rect = cv2.perspectiveTransform(canvas_rect.reshape(-1, 1, 2), inv_h).reshape(-1, 2)
        pts = pixel_rect.astype(np.int32)
        cv2.polylines(output, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
        for idx, point in enumerate(pts):
            cv2.circle(output, tuple(point), 5, (0, 255, 255), -1)
            cv2.putText(
                output,
                f"{idx}",
                tuple(point + np.array([5, -5])),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )
    return output


def save_homography(path: Path, homography: np.ndarray, config: CanvasConfig) -> None:
    payload = {
        "homography": homography.tolist(),
        "canvas_width_mm": config.width_mm,
        "canvas_height_mm": config.height_mm,
        "marker_ids": config.marker_ids or DEFAULT_MARKER_IDS,
    }
    path.write_text(json.dumps(payload, indent=2))


def load_observations(path: Path) -> List[Tuple[List[np.ndarray], np.ndarray]]:
    payload = json.loads(path.read_text())
    observations = []
    for obs in payload.get("observations", []):
        corners = [np.array(corner, dtype=np.float32) for corner in obs.get("corners", [])]
        ids = np.array(obs.get("ids", []), dtype=np.int32)
        if corners and ids.size:
            observations.append((corners, ids))
    return observations


def parse_marker_ids(marker_ids: Optional[str]) -> Dict[str, int]:
    if not marker_ids:
        return DEFAULT_MARKER_IDS
    payload = json.loads(marker_ids)
    return {key: int(value) for key, value in payload.items()}


def run_detection(args: argparse.Namespace) -> int:
    config = CanvasConfig(
        width_mm=args.width,
        height_mm=args.height,
        marker_length_mm=args.marker_length,
        marker_ids=parse_marker_ids(args.marker_ids),
    )
    calibration = load_calibration(args.calib) if args.calib else None

    accumulator = ObservationAccumulator()
    if args.scan_poses:
        for corners, ids in load_observations(Path(args.scan_poses)):
            accumulator.add(corners, ids)

    camera = Camera(args.camera, calibration)

    homography = None
    try:
        while True:
            success, frame = camera.read_frame()
            if not success or frame is None:
                continue

            detection = detect_markers(frame)
            if detection.ids.size:
                accumulator.add(detection.corners, detection.ids)

            averaged = accumulator.averaged_centers()

            if (config.width_mm is None or config.height_mm is None) and calibration and config.marker_length_mm:
                size = estimate_canvas_size_from_markers(detection.corners, detection.ids, config, calibration)
                if size:
                    config.width_mm, config.height_mm = size

            homography = compute_homography(averaged, config)

            if homography is not None:
                save_homography(Path(args.output), homography, config)

            if args.show:
                overlay = draw_debug_overlay(frame, detection, homography, config)
                cv2.imshow("Canvas Detection", overlay)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
            else:
                if homography is not None:
                    break
    finally:
        camera.release()
        if args.show:
            cv2.destroyAllWindows()

    if homography is None:
        print("Canvas not found. Try adjusting camera or scanning multiple poses.")
        return 1

    print("Canvas detected.")
    print(f"Canvas size: {config.width_mm}mm x {config.height_mm}mm")
    print(f"Homography saved to {args.output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Detect canvas using ArUco markers.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--calib", type=str, default=None, help="Camera calibration JSON")
    parser.add_argument("--width", type=float, default=None, help="Canvas width in mm")
    parser.add_argument("--height", type=float, default=None, help="Canvas height in mm")
    parser.add_argument("--marker-length", type=float, default=None, help="Marker length in mm")
    parser.add_argument(
        "--marker-ids",
        type=str,
        default=None,
        help='JSON mapping for marker ids, e.g. "{\"tl\":0,\"tr\":1,\"br\":2,\"bl\":3}"',
    )
    parser.add_argument("--scan-poses", type=str, default=None, help="JSON file with stored marker observations")
    parser.add_argument("--output", type=str, default="canvas_homography.json", help="Output JSON file")
    parser.add_argument("--show", action="store_true", help="Show debug overlay window")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return run_detection(args)


if __name__ == "__main__":
    raise SystemExit(main())
