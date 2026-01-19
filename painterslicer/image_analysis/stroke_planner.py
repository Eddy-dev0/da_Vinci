"""Generate executable stroke plans from sliced layer masks."""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np


@dataclass
class ToolChoice:
    tool_type: str
    tool_id: Optional[str]
    brush_width_mm: float


@dataclass
class Stroke:
    points_mm: List[Tuple[float, float]]
    brush_width_mm: float
    pressure_hint: float
    speed_hint: float
    kind: str


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_mask(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not read mask: {path}")
    return (mask > 0).astype(np.uint8)


def _component_masks(mask: np.ndarray) -> Iterable[np.ndarray]:
    num_labels, labels = cv2.connectedComponents(mask, connectivity=8)
    for label_idx in range(1, num_labels):
        component = (labels == label_idx).astype(np.uint8)
        if np.any(component):
            yield component


def _largest_contour(mask: np.ndarray) -> Optional[np.ndarray]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def _simplify_contour(contour: np.ndarray) -> List[Tuple[int, int]]:
    if contour is None or contour.size == 0:
        return []
    perimeter = cv2.arcLength(contour, True)
    epsilon = max(1.0, 0.01 * perimeter)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    points = [(int(pt[0][0]), int(pt[0][1])) for pt in approx]
    if points and points[0] != points[-1]:
        points.append(points[0])
    return points


def _mask_bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask > 0)
    if ys.size == 0 or xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _scanline_segments(mask: np.ndarray, spacing_px: int) -> List[List[Tuple[int, int]]]:
    bbox = _mask_bbox(mask)
    if bbox is None:
        return []
    x_min, y_min, x_max, y_max = bbox
    segments: List[List[Tuple[int, int]]] = []
    direction = 1
    for y in range(y_min, y_max + 1, spacing_px):
        row = mask[y, x_min : x_max + 1]
        if not np.any(row):
            continue
        idx = np.where(row > 0)[0]
        if idx.size == 0:
            continue
        breaks = np.where(np.diff(idx) > 1)[0]
        starts = np.insert(idx[breaks + 1], 0, idx[0])
        ends = np.append(idx[breaks], idx[-1])
        for start, end in zip(starts, ends):
            x0 = x_min + int(start)
            x1 = x_min + int(end)
            line = [(x0, y), (x1, y)]
            if direction < 0:
                line = list(reversed(line))
            segments.append(line)
            direction *= -1
    return segments


def _points_px_to_mm(points: Iterable[Tuple[int, int]], px_per_mm: float) -> List[Tuple[float, float]]:
    return [(float(x) / px_per_mm, float(y) / px_per_mm) for x, y in points]


def _stroke_area_mm2(mask: np.ndarray, px_per_mm: float) -> float:
    area_px = float(np.count_nonzero(mask))
    return area_px / (px_per_mm ** 2)


def _nearest_neighbor_order(strokes: List[Stroke]) -> List[Stroke]:
    if len(strokes) <= 1:
        return strokes
    remaining = strokes[:]
    remaining.sort(key=lambda s: (s.points_mm[0][1], s.points_mm[0][0]))
    ordered = [remaining.pop(0)]
    while remaining:
        last = ordered[-1].points_mm[-1]
        next_idx = min(
            range(len(remaining)),
            key=lambda idx: (remaining[idx].points_mm[0][0] - last[0]) ** 2
            + (remaining[idx].points_mm[0][1] - last[1]) ** 2,
        )
        ordered.append(remaining.pop(next_idx))
    return ordered


def _tool_choices(required_tools: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    mapping: Dict[str, List[Dict[str, Any]]] = {}
    for tool in required_tools.get("required_tools", []):
        for layer in tool.get("layers", []):
            color_id = str(layer.get("color_id"))
            mapping.setdefault(color_id, []).append(tool)
    return mapping


def _pick_tool_for_layer(layer: Dict[str, Any], tool_map: Dict[str, List[Dict[str, Any]]]) -> ToolChoice:
    color_id = str(layer.get("color_id"))
    recommended = layer.get("recommended_brush_widths_mm", []) or []
    preferred_width = float(recommended[0]) if recommended else 5.0
    tools = tool_map.get(color_id, [])
    if not tools:
        return ToolChoice(tool_type="BRUSH", tool_id=None, brush_width_mm=preferred_width)
    selected = tools[0]
    tool_type = str(selected.get("tool_type") or "BRUSH")
    tool_id = None
    matched = selected.get("matched_inventory") or []
    if matched:
        sized = [m for m in matched if m.get("size_mm") is not None]
        if sized:
            tool_entry = min(sized, key=lambda m: abs(float(m.get("size_mm", 0.0)) - preferred_width))
            tool_id = str(tool_entry.get("id"))
            preferred_width = float(tool_entry.get("size_mm"))
        else:
            tool_id = str(matched[0].get("id"))
    return ToolChoice(tool_type=tool_type, tool_id=tool_id, brush_width_mm=preferred_width)


def _build_strokes_for_component(
    mask: np.ndarray,
    px_per_mm: float,
    brush_width_mm: float,
    large_area_threshold_mm2: float,
    pressure_fill: float,
    pressure_contour: float,
    speed_fill: float,
    speed_contour: float,
) -> Tuple[List[Stroke], List[Stroke]]:
    area_mm2 = _stroke_area_mm2(mask, px_per_mm)
    contour = _largest_contour(mask)
    contour_points = _simplify_contour(contour)
    contour_mm = _points_px_to_mm(contour_points, px_per_mm)
    contour_strokes: List[Stroke] = []
    if contour_mm:
        contour_strokes.append(
            Stroke(
                points_mm=contour_mm,
                brush_width_mm=brush_width_mm,
                pressure_hint=pressure_contour,
                speed_hint=speed_contour,
                kind="contour",
            )
        )

    if area_mm2 < large_area_threshold_mm2:
        return [], contour_strokes

    spacing_px = max(1, int(round(brush_width_mm * 0.6 * px_per_mm)))
    segments = _scanline_segments(mask, spacing_px)
    fill_strokes: List[Stroke] = []
    for segment in segments:
        points_mm = _points_px_to_mm(segment, px_per_mm)
        if len(points_mm) < 2:
            continue
        fill_strokes.append(
            Stroke(
                points_mm=points_mm,
                brush_width_mm=brush_width_mm,
                pressure_hint=pressure_fill,
                speed_hint=speed_fill,
                kind="fill",
            )
        )
    return fill_strokes, contour_strokes


def build_stroke_plan(
    plan_path: Path,
    required_tools_path: Path,
    output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    plan = _load_json(plan_path)
    required_tools = _load_json(required_tools_path)
    plan_dir = plan_path.parent
    px_per_mm = float(plan.get("px_per_mm", 3.0))
    canvas_mm = plan.get("canvas_mm", {}) or {}
    layers = plan.get("layers", [])
    layer_order = plan.get("layer_order") or [layer.get("color_id") for layer in layers]
    layer_map = {str(layer.get("color_id")): layer for layer in layers}
    tool_map = _tool_choices(required_tools)

    steps: List[Dict[str, Any]] = []
    last_paint_id: Optional[str] = None
    last_tool: Optional[str] = None

    for color_id in layer_order:
        layer = layer_map.get(str(color_id))
        if layer is None:
            continue
        mask_files = layer.get("mask_files") or {}
        mask_file = mask_files.get("fine") or next(iter(mask_files.values()), None)
        if not mask_file:
            continue
        mask_path = plan_dir / mask_file
        mask = _load_mask(mask_path)
        if not np.any(mask):
            continue

        tool = _pick_tool_for_layer(layer, tool_map)
        paint_id = str(layer.get("color_id"))
        if last_paint_id is not None and paint_id != last_paint_id:
            steps.append({"type": "CLEAN_TOOL"})
        if tool.tool_id is not None:
            tool_key = f"{tool.tool_type}:{tool.tool_id}"
        else:
            tool_key = tool.tool_type
        if tool_key != last_tool:
            select_payload = {"type": "SELECT_TOOL", "tool_type": tool.tool_type}
            if tool.tool_id is not None:
                select_payload["tool_id"] = tool.tool_id
            steps.append(select_payload)
        if paint_id != last_paint_id:
            steps.append({"type": "LOAD_PAINT", "paint_id": paint_id})

        strokes_fill: List[Stroke] = []
        strokes_contour: List[Stroke] = []
        large_area_threshold_mm2 = max(tool.brush_width_mm ** 2 * 4.0, 80.0)
        for component in _component_masks(mask):
            fill, contour = _build_strokes_for_component(
                component,
                px_per_mm,
                tool.brush_width_mm,
                large_area_threshold_mm2,
                pressure_fill=0.6,
                pressure_contour=0.8,
                speed_fill=40.0,
                speed_contour=30.0,
            )
            strokes_fill.extend(fill)
            strokes_contour.extend(contour)

        ordered_fill = _nearest_neighbor_order(strokes_fill)
        ordered_contour = _nearest_neighbor_order(strokes_contour)
        strokes = ordered_fill + ordered_contour

        passes = 2 if float(layer.get("coverage_ratio", 0.0)) > 0.35 else 1
        for _ in range(passes):
            for stroke in strokes:
                steps.append(
                    {
                        "type": "STROKE",
                        "points_mm": [[round(x, 3), round(y, 3)] for x, y in stroke.points_mm],
                        "brush_width_mm": round(stroke.brush_width_mm, 2),
                        "pressure_hint": round(stroke.pressure_hint, 2),
                        "speed_hint": round(stroke.speed_hint, 2),
                    }
                )

        last_paint_id = paint_id
        last_tool = tool_key

    payload = {
        "meta": {
            "canvas_mm": {
                "width": float(canvas_mm.get("width", 0.0)),
                "height": float(canvas_mm.get("height", 0.0)),
            },
            "px_per_mm": px_per_mm,
            "generated_at": datetime.utcnow().isoformat() + "Z",
        },
        "steps": steps,
    }

    if output_path is None:
        output_path = plan_dir / "stroke_plan.json"
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def build_stroke_plan_from_cli(args: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    parser = argparse.ArgumentParser(description="Generate stroke_plan.json from plan layers")
    parser.add_argument("--plan", required=True, help="Path to plan_layers.json")
    parser.add_argument("--tools", required=True, help="Path to required_tools.json")
    parser.add_argument("--output", help="Output path for stroke_plan.json")
    parsed = parser.parse_args(args=args)

    output_path = Path(parsed.output) if parsed.output else None
    return build_stroke_plan(Path(parsed.plan), Path(parsed.tools), output_path)


if __name__ == "__main__":
    build_stroke_plan_from_cli()
