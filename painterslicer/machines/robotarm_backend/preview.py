"""Dry-run preview for PaintCode plans."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import cv2
import numpy as np

from painterslicer.machines.robotarm_backend.paintcode_parser import PaintCodeParser, PaintStep


@dataclass
class PreviewConfig:
    canvas_width_mm: float
    canvas_height_mm: float
    pixels_per_mm: float = 2.0
    stroke_color: Tuple[int, int, int] = (50, 80, 200)
    bg_color: Tuple[int, int, int] = (240, 240, 240)


def render_preview(steps: Iterable[PaintStep], config: PreviewConfig, output_path: Path) -> Path:
    width_px = int(config.canvas_width_mm * config.pixels_per_mm)
    height_px = int(config.canvas_height_mm * config.pixels_per_mm)
    canvas = np.full((height_px, width_px, 3), config.bg_color, dtype=np.uint8)

    points: List[Tuple[float, float]] = []
    for step in steps:
        if step.command == "MOVE":
            x_mm, y_mm = PaintCodeParser.move_args(step)
            points.append((x_mm, y_mm))

    if len(points) >= 2:
        for start, end in zip(points, points[1:]):
            sx = int(start[0] * config.pixels_per_mm)
            sy = int(start[1] * config.pixels_per_mm)
            ex = int(end[0] * config.pixels_per_mm)
            ey = int(end[1] * config.pixels_per_mm)
            cv2.line(canvas, (sx, sy), (ex, ey), config.stroke_color, 2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas)
    return output_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PaintCode dry-run preview.")
    parser.add_argument("paintcode", help="Path to PaintCode file")
    parser.add_argument("--canvas-width", type=float, required=True)
    parser.add_argument("--canvas-height", type=float, required=True)
    parser.add_argument("--pixels-per-mm", type=float, default=2.0)
    parser.add_argument("--output", default="preview.png")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    text = Path(args.paintcode).read_text(encoding="utf-8")
    steps = PaintCodeParser(text).parse()
    config = PreviewConfig(
        canvas_width_mm=args.canvas_width,
        canvas_height_mm=args.canvas_height,
        pixels_per_mm=args.pixels_per_mm,
    )
    output_path = render_preview(steps, config, Path(args.output))
    print(f"Preview saved to {output_path}")


if __name__ == "__main__":
    main()
