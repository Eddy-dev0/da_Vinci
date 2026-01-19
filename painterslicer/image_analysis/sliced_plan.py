"""Generate palette-quantized, multi-resolution paint layers from a target image."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from uuid import uuid4

import cv2
import numpy as np
from PIL import Image


@dataclass
class CanvasSpec:
    width_mm: float
    height_mm: float
    px_per_mm: float
    fit_mode: str = "fit"
    background_rgb: Tuple[int, int, int] = (255, 255, 255)


@dataclass
class MultiResSpec:
    coarse_scale: float = 0.25
    medium_scale: float = 0.5
    blur_sigma: float = 1.2


@dataclass
class SlicerSpec:
    canvas: CanvasSpec
    multi_res: MultiResSpec = MultiResSpec()
    morphology: bool = True
    morph_kernel_mm: float = 0.6
    sort_mode: str = "area_dark"


@dataclass
class PaletteColor:
    color_id: str
    name: str
    rgb: Tuple[int, int, int]


def _load_inventory_palette(path: Path) -> List[PaletteColor]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    paints = payload.get("paints") or payload.get("palette") or payload.get("colors") or []
    palette: List[PaletteColor] = []
    for entry in paints:
        rgb = entry.get("rgb")
        if not rgb or len(rgb) != 3:
            continue
        color_id = str(entry.get("id") or entry.get("name") or f"color_{len(palette)}")
        name = str(entry.get("name") or color_id)
        palette.append(PaletteColor(color_id=color_id, name=name, rgb=tuple(int(c) for c in rgb)))
    if not palette:
        raise ValueError("inventory palette does not contain any usable rgb colors")
    return palette


def _resolve_px_per_mm(px_per_mm: Optional[float], dpi: Optional[float]) -> float:
    if px_per_mm is not None:
        return float(px_per_mm)
    if dpi is not None:
        return float(dpi) / 25.4
    return 3.0


def _resolve_canvas_spec(
    image_size: Tuple[int, int],
    width_mm: Optional[float],
    height_mm: Optional[float],
    px_per_mm: float,
    fit_mode: str,
    background_rgb: Tuple[int, int, int],
) -> CanvasSpec:
    if width_mm is None or height_mm is None:
        img_w, img_h = image_size
        width_mm = float(img_w) / px_per_mm
        height_mm = float(img_h) / px_per_mm
    return CanvasSpec(
        width_mm=float(width_mm),
        height_mm=float(height_mm),
        px_per_mm=float(px_per_mm),
        fit_mode=fit_mode,
        background_rgb=background_rgb,
    )


def _resize_to_canvas(img: Image.Image, canvas: CanvasSpec) -> Image.Image:
    target_w = max(1, int(round(canvas.width_mm * canvas.px_per_mm)))
    target_h = max(1, int(round(canvas.height_mm * canvas.px_per_mm)))
    img_w, img_h = img.size
    scale_fit = min(target_w / img_w, target_h / img_h)
    scale_fill = max(target_w / img_w, target_h / img_h)
    scale = scale_fit if canvas.fit_mode == "fit" else scale_fill
    new_w = max(1, int(round(img_w * scale)))
    new_h = max(1, int(round(img_h * scale)))
    resized = img.resize((new_w, new_h), resample=Image.LANCZOS)
    if canvas.fit_mode == "fit":
        background = Image.new("RGB", (target_w, target_h), color=canvas.background_rgb)
        offset_x = (target_w - new_w) // 2
        offset_y = (target_h - new_h) // 2
        background.paste(resized, (offset_x, offset_y))
        return background
    if new_w == target_w and new_h == target_h:
        return resized
    left = max(0, (new_w - target_w) // 2)
    upper = max(0, (new_h - target_h) // 2)
    right = left + target_w
    lower = upper + target_h
    return resized.crop((left, upper, right, lower))


def _rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2LAB)
    return lab.astype(np.float32)


def _quantize_to_palette(
    img_rgb: np.ndarray,
    palette: Sequence[PaletteColor],
    chunk: int = 150_000,
) -> Tuple[np.ndarray, np.ndarray]:
    img_lab = _rgb_to_lab(img_rgb)
    palette_rgb = np.array([c.rgb for c in palette], dtype=np.uint8)
    palette_lab = _rgb_to_lab(palette_rgb)
    flat_lab = img_lab.reshape(-1, 3)
    labels = np.empty(flat_lab.shape[0], dtype=np.int32)
    for start in range(0, flat_lab.shape[0], chunk):
        end = min(start + chunk, flat_lab.shape[0])
        block = flat_lab[start:end]
        diff = block[:, None, :] - palette_lab[None, :, :]
        dist = np.sum(diff * diff, axis=2)
        labels[start:end] = np.argmin(dist, axis=1)
    quant_rgb = palette_rgb[labels].reshape(img_rgb.shape)
    return labels.reshape(img_rgb.shape[:2]), quant_rgb


def _apply_morphology(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    if kernel_size <= 1:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
    return opened


def _multi_res_masks(mask: np.ndarray, spec: MultiResSpec) -> Dict[str, np.ndarray]:
    mask_f = mask.astype(np.float32)
    blurred = cv2.GaussianBlur(mask_f, (0, 0), spec.blur_sigma)

    def _down_up(scale: float) -> np.ndarray:
        h, w = mask.shape
        small = cv2.resize(blurred, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
        up = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
        return (up > 0.5).astype(np.uint8)

    coarse = _down_up(spec.coarse_scale)
    medium = _down_up(spec.medium_scale)
    fine = (mask > 0).astype(np.uint8)
    return {"coarse": coarse, "medium": medium, "fine": fine}


def _mask_bbox(mask: np.ndarray) -> Optional[Dict[str, int]]:
    ys, xs = np.where(mask > 0)
    if ys.size == 0 or xs.size == 0:
        return None
    return {
        "x_min": int(xs.min()),
        "y_min": int(ys.min()),
        "x_max": int(xs.max()),
        "y_max": int(ys.max()),
    }


def _recommended_brushes(coverage_ratio: float, canvas_area_mm2: float) -> List[float]:
    area_mm2 = max(coverage_ratio * canvas_area_mm2, 1e-6)
    side = math.sqrt(area_mm2)
    widths = [side / 12.0, side / 25.0, side / 50.0]
    clamped = [max(0.5, min(30.0, w)) for w in widths]
    unique = sorted({round(w, 2) for w in clamped}, reverse=True)
    return unique


def _layer_sort_key(
    mask_area: int,
    palette_lab: np.ndarray,
    palette_idx: int,
    sort_mode: str,
) -> Tuple[float, float]:
    area_key = -float(mask_area)
    luminance = float(palette_lab[palette_idx][0])
    if sort_mode == "area_light":
        return (area_key, -luminance)
    if sort_mode == "dark_only":
        return (0.0, luminance)
    return (area_key, luminance)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def build_sliced_plan(
    target_image: Path,
    inventory_path: Path,
    output_root: Path = Path("plans"),
    job_id: Optional[str] = None,
    width_mm: Optional[float] = None,
    height_mm: Optional[float] = None,
    px_per_mm: Optional[float] = None,
    dpi: Optional[float] = None,
    fit_mode: str = "fit",
    sort_mode: str = "area_dark",
    background_rgb: Tuple[int, int, int] = (255, 255, 255),
) -> Dict[str, Any]:
    palette = _load_inventory_palette(inventory_path)
    px_per_mm = _resolve_px_per_mm(px_per_mm, dpi)
    img = Image.open(target_image).convert("RGB")
    canvas = _resolve_canvas_spec(img.size, width_mm, height_mm, px_per_mm, fit_mode, background_rgb)
    spec = SlicerSpec(canvas=canvas, sort_mode=sort_mode)
    resized = _resize_to_canvas(img, canvas)
    resized_rgb = np.array(resized, dtype=np.uint8)

    labels, quant_rgb = _quantize_to_palette(resized_rgb, palette)
    quant_image = Image.fromarray(quant_rgb, mode="RGB")

    palette_lab = _rgb_to_lab(np.array([c.rgb for c in palette], dtype=np.uint8))
    total_pixels = labels.size
    kernel_px = max(1, int(round(spec.morph_kernel_mm * canvas.px_per_mm)))

    if job_id is None:
        job_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:6]
    output_dir = output_root / job_id
    _ensure_dir(output_dir)

    quant_path = output_dir / "quantized_image.png"
    quant_image.save(quant_path)

    layer_entries: List[Dict[str, Any]] = []
    used_palette: List[Dict[str, Any]] = []
    layer_order: List[str] = []

    for idx, color in enumerate(palette):
        mask = (labels == idx).astype(np.uint8)
        if spec.morphology:
            mask = _apply_morphology(mask, kernel_px)
        mask_area = int(np.count_nonzero(mask))
        if mask_area == 0:
            continue
        coverage = mask_area / total_pixels
        multi_masks = _multi_res_masks(mask, spec.multi_res)

        color_tag = str(color.color_id)
        used_palette.append({"id": color_tag, "name": color.name, "rgb": list(color.rgb)})
        mask_files: Dict[str, str] = {}
        for level, level_mask in multi_masks.items():
            filename = f"{color_tag}_{level}.png"
            Image.fromarray((level_mask * 255).astype(np.uint8), mode="L").save(output_dir / filename)
            mask_files[level] = filename
        layer_order.append(color_tag)

        layer_entries.append(
            {
                "color_id": color_tag,
                "color_name": color.name,
                "rgb": list(color.rgb),
                "coverage_ratio": float(round(coverage, 6)),
                "recommended_brush_widths_mm": _recommended_brushes(coverage, canvas.width_mm * canvas.height_mm),
                "mask_files": mask_files,
                "bbox_px": _mask_bbox(mask),
                "pixel_area": mask_area,
                "sort_key": _layer_sort_key(mask_area, palette_lab, idx, spec.sort_mode),
            }
        )

    layer_entries.sort(key=lambda entry: entry["sort_key"])
    layer_order = [entry["color_id"] for entry in layer_entries]
    for entry in layer_entries:
        entry.pop("sort_key", None)

    plan = {
        "job_id": job_id,
        "canvas_mm": {"width": canvas.width_mm, "height": canvas.height_mm},
        "px_per_mm": canvas.px_per_mm,
        "fit_mode": canvas.fit_mode,
        "quantized_image": quant_path.name,
        "palette": used_palette,
        "layer_order": layer_order,
        "layers": layer_entries,
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }

    plan_path = output_dir / "plan_layers.json"
    plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    return plan


def build_sliced_plan_from_cli(args: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    import argparse

    parser = argparse.ArgumentParser(description="Generate palette-quantized paint layers from an image")
    parser.add_argument("--target", required=True, help="Path to target image (png/jpg)")
    parser.add_argument("--inventory", required=True, help="Path to inventory.json")
    parser.add_argument("--width-mm", type=float, help="Canvas width in mm")
    parser.add_argument("--height-mm", type=float, help="Canvas height in mm")
    parser.add_argument("--px-per-mm", type=float, help="Pixels per mm for working resolution")
    parser.add_argument("--dpi", type=float, help="Optional DPI if px_per_mm not provided")
    parser.add_argument("--fit-mode", choices=["fit", "fill"], default="fit", help="Fit or fill")
    parser.add_argument("--sort-mode", default="area_dark", help="Layer sorting mode")
    parser.add_argument("--job-id", help="Optional job id")
    parser.add_argument("--output-root", default="plans", help="Output root directory")
    parsed = parser.parse_args(args=args)

    return build_sliced_plan(
        target_image=Path(parsed.target),
        inventory_path=Path(parsed.inventory),
        output_root=Path(parsed.output_root),
        job_id=parsed.job_id,
        width_mm=parsed.width_mm,
        height_mm=parsed.height_mm,
        px_per_mm=parsed.px_per_mm,
        dpi=parsed.dpi,
        fit_mode=parsed.fit_mode,
        sort_mode=parsed.sort_mode,
    )


if __name__ == "__main__":
    build_sliced_plan_from_cli()
