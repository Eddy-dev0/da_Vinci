"""PaintCode parsing utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class PaintStep:
    command: str
    args: Tuple[str, ...] = ()
    raw: str = ""


class PaintCodeParser:
    """Parse PaintCode text into structured steps."""

    def __init__(self, text: Optional[str] = None) -> None:
        self.text = text or ""

    def parse(self, text: Optional[str] = None) -> List[PaintStep]:
        payload = text if text is not None else self.text
        steps: List[PaintStep] = []
        for line in payload.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#") or stripped.startswith("//") or stripped.startswith(";"):
                continue
            tokens = stripped.split()
            command = tokens[0].upper()
            args = tuple(tokens[1:])
            if command in {"TOOL", "PRESSURE", "Z", "MOVE", "Z_UP", "Z_DOWN", "CLEAN", "WASH_STATION", "WIPE"}:
                steps.append(PaintStep(command=command, args=args, raw=stripped))
            else:
                steps.append(PaintStep(command=command, args=args, raw=stripped))
        return steps

    @staticmethod
    def move_args(step: PaintStep) -> Tuple[float, float]:
        if len(step.args) < 2:
            raise ValueError(f"MOVE requires two coordinates, got: {step.args}")
        return float(step.args[0]), float(step.args[1])

    @staticmethod
    def z_value(step: PaintStep) -> Optional[float]:
        if step.command == "Z" and step.args:
            return float(step.args[0])
        return None

    @staticmethod
    def pressure_value(step: PaintStep) -> Optional[float]:
        if step.command == "PRESSURE" and step.args:
            return float(step.args[0])
        return None

    @staticmethod
    def tool_value(step: PaintStep) -> Optional[str]:
        if step.command == "TOOL" and step.args:
            return step.args[0]
        return None


def iter_moves(steps: Iterable[PaintStep]) -> Iterable[Tuple[float, float]]:
    for step in steps:
        if step.command == "MOVE":
            yield PaintCodeParser.move_args(step)
