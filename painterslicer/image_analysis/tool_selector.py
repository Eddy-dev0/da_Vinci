"""Derive required brush/tool list from plan layers and inventory."""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np


class ToolTaxonomy(str, Enum):
    ROUND = "ROUND"
    FLAT = "FLAT"
    BRIGHT = "BRIGHT"
    FILBERT = "FILBERT"
    FAN = "FAN"
    ANGLE = "ANGLE"
    RIGGER_LINER = "RIGGER/LINER"
    MOP_WASH = "MOP/WASH"
    SPONGE = "SPONGE"
    PALETTE_KNIFE = "PALETTE_KNIFE"
    DAUBER = "DAUBER"
    SILICONE_SHAPER = "SILICONE_SHAPER"


TOOL_DESCRIPTIONS: Dict[ToolTaxonomy, str] = {
    ToolTaxonomy.ROUND: "rund: Detail, Linien, kleine Flächen",
    ToolTaxonomy.FLAT: "breite Flächen, klare Kanten",
    ToolTaxonomy.BRIGHT: "kurzer Flat: kräftiger Druck, deckende kurze Striche",
    ToolTaxonomy.FILBERT: "weichere Kanten, organische Formen, vielseitig",
    ToolTaxonomy.FAN: "Textur, weiches Blending, Gras/Haar-Effekte",
    ToolTaxonomy.ANGLE: "scharfe Kanten, Ecken, cut-in",
    ToolTaxonomy.RIGGER_LINER: "sehr lange feine Linien",
    ToolTaxonomy.MOP_WASH: "große weiche Flächen/Lasuren",
    ToolTaxonomy.SPONGE: "tupfen, Textur, Wolken/Poren/Background",
    ToolTaxonomy.PALETTE_KNIFE: "Spachtel/Knife für harte Texturen",
    ToolTaxonomy.DAUBER: "Stupfpinsel für harte Tupfer",
    ToolTaxonomy.SILICONE_SHAPER: "Gummi-Tool für weiche Push/Pull Effekte",
}


FALLBACK_MAP: Dict[ToolTaxonomy, Sequence[ToolTaxonomy]] = {
    ToolTaxonomy.MOP_WASH: (ToolTaxonomy.FLAT, ToolTaxonomy.FILBERT),
    ToolTaxonomy.FLAT: (ToolTaxonomy.FILBERT, ToolTaxonomy.BRIGHT),
    ToolTaxonomy.BRIGHT: (ToolTaxonomy.FLAT, ToolTaxonomy.FILBERT),
    ToolTaxonomy.FILBERT: (ToolTaxonomy.ROUND, ToolTaxonomy.FLAT),
    ToolTaxonomy.ROUND: (ToolTaxonomy.FILBERT,),
    ToolTaxonomy.ANGLE: (ToolTaxonomy.FLAT, ToolTaxonomy.ROUND),
    ToolTaxonomy.RIGGER_LINER: (ToolTaxonomy.ROUND,),
    ToolTaxonomy.FAN: (ToolTaxonomy.SPONGE, ToolTaxonomy.FILBERT),
    ToolTaxonomy.SPONGE: (ToolTaxonomy.FAN,),
    ToolTaxonomy.PALETTE_KNIFE: (ToolTaxonomy.FLAT,),
    ToolTaxonomy.DAUBER: (ToolTaxonomy.SPONGE,),
    ToolTaxonomy.SILICONE_SHAPER: (ToolTaxonomy.FILBERT,),
}


@dataclass
class LayerMetrics:
    color_id: str
    color_name: str
    rgb: Tuple[int, int, int]
    area_mm2: float
    thickness_mm: float
    sqrt_area_mm: float
    grainy: bool
    thin_long: bool
    circularity: float


@dataclass
class ToolRequirement:
    tool_type: ToolTaxonomy
    recommended_sizes_mm: List[float]
    usage: str
    layers: List[Dict[str, Any]]
    substitute: Optional[ToolTaxonomy] = None
    warning: Optional[str] = None
    matched_inventory: List[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool_type": self.tool_type.value,
            "description": TOOL_DESCRIPTIONS.get(self.tool_type, ""),
            "recommended_sizes_mm": self.recommended_sizes_mm,
            "usage": self.usage,
            "layers": self.layers,
            "substitute": self.substitute.value if self.substitute else None,
            "warning": self.warning,
            "matched_inventory": self.matched_inventory or [],
        }


def _normalize_tool_type(raw: str) -> Optional[ToolTaxonomy]:
    key = raw.strip().lower()
    if not key:
        return None
    mapping = {
        "round": ToolTaxonomy.ROUND,
        "flat": ToolTaxonomy.FLAT,
        "bright": ToolTaxonomy.BRIGHT,
        "filbert": ToolTaxonomy.FILBERT,
        "fan": ToolTaxonomy.FAN,
        "angle": ToolTaxonomy.ANGLE,
        "angled": ToolTaxonomy.ANGLE,
        "rigger": ToolTaxonomy.RIGGER_LINER,
        "liner": ToolTaxonomy.RIGGER_LINER,
        "mop": ToolTaxonomy.MOP_WASH,
        "wash": ToolTaxonomy.MOP_WASH,
        "sponge": ToolTaxonomy.SPONGE,
        "palette_knife": ToolTaxonomy.PALETTE_KNIFE,
        "knife": ToolTaxonomy.PALETTE_KNIFE,
        "dauber": ToolTaxonomy.DAUBER,
        "silicone": ToolTaxonomy.SILICONE_SHAPER,
        "shaper": ToolTaxonomy.SILICONE_SHAPER,
    }
    for token, tool in mapping.items():
        if token in key:
            return tool
    return None


def _load_plan(plan_path: Path) -> Dict[str, Any]:
    return json.loads(plan_path.read_text(encoding="utf-8"))


def _load_inventory(inventory_path: Path) -> Dict[str, Any]:
    return json.loads(inventory_path.read_text(encoding="utf-8"))


def _load_mask(mask_path: Path) -> np.ndarray:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not read mask: {mask_path}")
    return (mask > 0).astype(np.uint8)


def _circularity(area_px: float, perimeter_px: float) -> float:
    if perimeter_px <= 0:
        return 0.0
    return float(4.0 * math.pi * area_px / (perimeter_px ** 2))


def _analyze_mask(mask: np.ndarray, px_per_mm: float) -> Tuple[float, float, bool, bool, float]:
    area_px = float(np.count_nonzero(mask))
    area_mm2 = area_px / (px_per_mm ** 2)
    sqrt_area_mm = math.sqrt(max(area_mm2, 1e-6))
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    dist_vals = dist[mask > 0]
    median_dist_px = float(np.median(dist_vals)) if dist_vals.size else 0.0
    thickness_mm = 2.0 * median_dist_px / px_per_mm

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    perimeter_px = sum(cv2.arcLength(cnt, True) for cnt in contours)
    circularity = _circularity(area_px, perimeter_px)

    num_labels, _ = cv2.connectedComponents(mask)
    component_count = max(num_labels - 1, 0)
    avg_area_mm2 = area_mm2 / max(component_count, 1)
    grainy = component_count > 20 or (component_count > 5 and avg_area_mm2 < 12.0)

    perimeter_mm = perimeter_px / px_per_mm
    thin_long = thickness_mm < 3.0 and perimeter_mm > 80.0
    return area_mm2, thickness_mm, grainy, thin_long, circularity


def _desired_stroke_width_mm(thickness_mm: float, sqrt_area_mm: float) -> float:
    return float(min(thickness_mm, sqrt_area_mm))


def _recommend_sizes(width_mm: float) -> List[float]:
    widths = [width_mm * 0.8, width_mm, width_mm * 1.3]
    clamped = [max(0.5, min(50.0, w)) for w in widths]
    unique = sorted({round(w, 1) for w in clamped})
    return unique


def _select_tool_types(metrics: LayerMetrics) -> List[ToolTaxonomy]:
    width_mm = _desired_stroke_width_mm(metrics.thickness_mm, metrics.sqrt_area_mm)
    tools: List[ToolTaxonomy] = []
    if metrics.thin_long or width_mm < 3.0:
        tools.append(ToolTaxonomy.RIGGER_LINER)
    elif width_mm > 30.0:
        tools.append(ToolTaxonomy.SPONGE if metrics.grainy else ToolTaxonomy.MOP_WASH)
    elif 10.0 <= width_mm <= 30.0:
        if metrics.circularity < 0.35:
            tools.append(ToolTaxonomy.FILBERT)
        else:
            tools.append(ToolTaxonomy.FLAT)
    elif 3.0 <= width_mm < 10.0:
        if metrics.circularity < 0.3:
            tools.append(ToolTaxonomy.ANGLE)
        else:
            tools.append(ToolTaxonomy.ROUND)

    if metrics.grainy:
        tools.append(ToolTaxonomy.SPONGE)
        if width_mm >= 3.0:
            tools.append(ToolTaxonomy.FAN)

    if 3.0 <= width_mm < 10.0 and metrics.area_mm2 < 80.0:
        tools.append(ToolTaxonomy.BRIGHT)

    deduped: List[ToolTaxonomy] = []
    for tool in tools:
        if tool not in deduped:
            deduped.append(tool)
    return deduped


def _summarize_usage(tool: ToolTaxonomy, metrics: LayerMetrics) -> str:
    notes = ["Layer-Fläche {:.1f}mm²".format(metrics.area_mm2)]
    if metrics.grainy and tool in {ToolTaxonomy.SPONGE, ToolTaxonomy.FAN}:
        notes.append("Textur/Fragmentierung")
    if metrics.thin_long and tool == ToolTaxonomy.RIGGER_LINER:
        notes.append("lange dünne Konturen")
    return "; ".join(notes)


def _tool_inventory_index(inventory: Dict[str, Any]) -> Dict[ToolTaxonomy, List[Dict[str, Any]]]:
    tools: Dict[ToolTaxonomy, List[Dict[str, Any]]] = {}
    for entry in inventory.get("tools", []):
        raw_type = str(entry.get("tool_type", ""))
        normalized = _normalize_tool_type(raw_type)
        if normalized is None:
            continue
        tools.setdefault(normalized, []).append(entry)
    return tools


def _match_inventory(
    tool_type: ToolTaxonomy,
    recommended_sizes_mm: List[float],
    inventory_index: Dict[ToolTaxonomy, List[Dict[str, Any]]],
) -> Tuple[List[Dict[str, Any]], Optional[ToolTaxonomy], Optional[str]]:
    if tool_type in inventory_index:
        return inventory_index[tool_type], None, None
    for candidate in FALLBACK_MAP.get(tool_type, []):
        if candidate in inventory_index:
            warning = f"{tool_type.value} nicht verfügbar, ersetze mit {candidate.value}"
            return inventory_index[candidate], candidate, warning
    warning = f"{tool_type.value} nicht verfügbar und kein Ersatz im Inventory"
    return [], None, warning


def build_required_tools(
    plan_path: Path,
    inventory_path: Path,
    output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    plan = _load_plan(plan_path)
    inventory = _load_inventory(inventory_path)
    plan_dir = plan_path.parent
    px_per_mm = float(plan.get("px_per_mm", 3.0))
    layers = plan.get("layers", [])

    inventory_index = _tool_inventory_index(inventory)
    requirements: Dict[ToolTaxonomy, ToolRequirement] = {}
    missing: List[str] = []
    warnings: List[str] = []

    for layer in layers:
        mask_files = layer.get("mask_files") or {}
        mask_file = mask_files.get("fine") or next(iter(mask_files.values()), None)
        if not mask_file:
            continue
        mask_path = plan_dir / mask_file
        mask = _load_mask(mask_path)
        area_mm2, thickness_mm, grainy, thin_long, circularity = _analyze_mask(mask, px_per_mm)
        metrics = LayerMetrics(
            color_id=str(layer.get("color_id")),
            color_name=str(layer.get("color_name")),
            rgb=tuple(int(c) for c in layer.get("rgb", [0, 0, 0])),
            area_mm2=area_mm2,
            thickness_mm=thickness_mm,
            sqrt_area_mm=math.sqrt(max(area_mm2, 1e-6)),
            grainy=grainy,
            thin_long=thin_long,
            circularity=circularity,
        )
        tool_types = _select_tool_types(metrics)
        width_mm = _desired_stroke_width_mm(metrics.thickness_mm, metrics.sqrt_area_mm)
        recommended_sizes = _recommend_sizes(width_mm)

        for tool_type in tool_types:
            requirement = requirements.get(tool_type)
            layer_entry = {
                "color_id": metrics.color_id,
                "color_name": metrics.color_name,
                "rgb": list(metrics.rgb),
            }
            if requirement is None:
                matched_tools, substitute, warning = _match_inventory(
                    tool_type, recommended_sizes, inventory_index
                )
                if warning:
                    warnings.append(warning)
                    if not matched_tools:
                        missing.append(tool_type.value)
                requirement = ToolRequirement(
                    tool_type=tool_type,
                    recommended_sizes_mm=recommended_sizes,
                    usage=_summarize_usage(tool_type, metrics),
                    layers=[layer_entry],
                    substitute=substitute,
                    warning=warning,
                    matched_inventory=matched_tools,
                )
                requirements[tool_type] = requirement
            else:
                requirement.layers.append(layer_entry)
                requirement.recommended_sizes_mm = sorted(
                    {round(size, 1) for size in (requirement.recommended_sizes_mm + recommended_sizes)}
                )

    required_list = [req.to_dict() for req in requirements.values()]
    payload = {
        "plan": plan_path.name,
        "required_tools": required_list,
        "missing_tools": sorted(set(missing)),
        "warnings": warnings,
    }

    if output_path is None:
        output_path = plan_dir / "required_tools.json"
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def _render_console(payload: Dict[str, Any]) -> str:
    lines = ["Benötigte Tools (min.):"]
    for tool in payload.get("required_tools", []):
        sizes = ", ".join(f"{size}mm" for size in tool.get("recommended_sizes_mm", []))
        layers = tool.get("layers", [])
        layer_names = ", ".join(sorted({layer.get("color_name", "") for layer in layers}))
        usage = tool.get("usage", "")
        info = f"- {tool.get('tool_type')}: {sizes} | {usage} | Layer: {layer_names}"
        if tool.get("substitute"):
            info += f" | Ersatz: {tool.get('substitute')}"
        if tool.get("warning"):
            info += f" | WARNUNG: {tool.get('warning')}"
        lines.append(info)
    missing = payload.get("missing_tools", [])
    if missing:
        lines.append("Nicht verfügbar im Inventory -> bitte hinzufügen:")
        for entry in missing:
            lines.append(f"- {entry}")
    return "\n".join(lines)


def build_required_tools_from_cli(args: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    parser = argparse.ArgumentParser(description="Derive required tools from plan_layers.json")
    parser.add_argument("--plan", required=True, help="Path to plan_layers.json")
    parser.add_argument("--inventory", required=True, help="Path to inventory.json")
    parser.add_argument("--output", help="Output path for required_tools.json")
    parsed = parser.parse_args(args=args)

    output_path = Path(parsed.output) if parsed.output else None
    payload = build_required_tools(Path(parsed.plan), Path(parsed.inventory), output_path)
    print(_render_console(payload))
    return payload


if __name__ == "__main__":
    build_required_tools_from_cli()
