"""Toolchain helpers for robot arm painting."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from painterslicer.machines.robotarm_backend.paintcode_parser import PaintCodeParser, PaintStep


TOOL_TYPES = [
    "ROUND",
    "FLAT",
    "BRIGHT",
    "FILBERT",
    "FAN",
    "ANGLE",
    "RIGGER",
    "LINER",
    "MOP",
    "WASH",
    "SPONGE",
]


@dataclass
class ToolRequirement:
    tool_type: str
    size_class: str


class Toolchain:
    def __init__(self, inventory: Optional[Dict[str, Any]] = None) -> None:
        self.inventory = inventory or {"paints": [], "tools": []}
        self.current_tool: Optional[str] = None

    def select_tool(self, tool_id: str) -> None:
        self.current_tool = tool_id

    def clean_tool(self) -> None:
        return None

    def load_paint(self, paint_id: str) -> None:
        return None


def required_tools_from_plan(
    plan_or_paintcode: Iterable[PaintStep] | str,
    inventory_path: Optional[Path] = None,
) -> Tuple[List[ToolRequirement], Dict[str, List[str]]]:
    if isinstance(plan_or_paintcode, str):
        steps = PaintCodeParser(plan_or_paintcode).parse()
    else:
        steps = list(plan_or_paintcode)

    tool_names = [PaintCodeParser.tool_value(step) for step in steps if step.command == "TOOL"]
    tool_names = [name for name in tool_names if name]

    requirements: List[ToolRequirement] = []
    for name in tool_names:
        size_class = _size_class_from_name(name)
        tool_type = _tool_type_from_name(name)
        requirements.append(ToolRequirement(tool_type=tool_type, size_class=size_class))

    available = []
    missing = []
    inventory = _load_inventory(inventory_path)
    inventory_tools = inventory.get("tools", [])
    inventory_pairs = {(tool.get("tool_type"), tool.get("size_class")) for tool in inventory_tools}

    for requirement in requirements:
        if (requirement.tool_type, requirement.size_class) in inventory_pairs:
            available.append(f"{requirement.tool_type}:{requirement.size_class}")
        else:
            missing.append(f"{requirement.tool_type}:{requirement.size_class}")

    report = {"available": sorted(set(available)), "missing": sorted(set(missing))}
    return requirements, report


def _load_inventory(path: Optional[Path]) -> Dict[str, Any]:
    if not path:
        return {"paints": [], "tools": []}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {"paints": [], "tools": []}


def _size_class_from_name(name: str) -> str:
    lowered = name.lower()
    if "fine" in lowered or "small" in lowered:
        return "small"
    if "medium" in lowered or "mid" in lowered:
        return "medium"
    if "broad" in lowered or "large" in lowered or "big" in lowered:
        return "large"
    return "medium"


def _tool_type_from_name(name: str) -> str:
    lowered = name.lower()
    if "sponge" in lowered:
        return "SPONGE"
    if "fan" in lowered:
        return "FAN"
    if "filbert" in lowered:
        return "FILBERT"
    if "bright" in lowered:
        return "BRIGHT"
    if "flat" in lowered:
        return "FLAT"
    if "angle" in lowered:
        return "ANGLE"
    if "rigger" in lowered or "liner" in lowered:
        return "RIGGER"
    if "mop" in lowered or "wash" in lowered:
        return "MOP"
    return "ROUND"
