from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def clamp_angle(angle: int) -> int:
    return max(0, min(180, angle))


def clamp_angles(angles: Iterable[int]) -> List[int]:
    return [clamp_angle(angle) for angle in angles]


@dataclass
class Pose:
    angles: List[int]
    note: Optional[str] = None
    tool_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"angles": self.angles}
        if self.note:
            payload["note"] = self.note
        if self.tool_id:
            payload["tool_id"] = self.tool_id
        payload.update(self.metadata)
        return payload

    @classmethod
    def from_dict(cls, data: Any) -> "Pose":
        if isinstance(data, list):
            return cls(angles=clamp_angles([int(angle) for angle in data]))
        if not isinstance(data, dict):
            raise ValueError("Pose must be a list of angles or a dict with angles.")
        angles = data.get("angles")
        if angles is None:
            raise ValueError("Pose dict missing 'angles'.")
        note = data.get("note")
        tool_id = data.get("tool_id")
        metadata = {
            key: value
            for key, value in data.items()
            if key not in {"angles", "note", "tool_id"}
        }
        return cls(
            angles=clamp_angles([int(angle) for angle in angles]),
            note=note,
            tool_id=tool_id,
            metadata=metadata,
        )


@dataclass
class NamedPose:
    name: str
    pose: Pose

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "pose": self.pose.to_dict()}


@dataclass
class BrushSlot:
    name: str
    pick_pose: Optional[Pose] = None
    place_pose: Optional[Pose] = None

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"name": self.name}
        if self.pick_pose:
            payload["pick_pose"] = self.pick_pose.to_dict()
        if self.place_pose:
            payload["place_pose"] = self.place_pose.to_dict()
        return payload


@dataclass
class Calibration:
    home_pose: Optional[Pose] = None
    safe_travel_pose: Optional[Pose] = None
    canvas: Dict[str, Any] = field(default_factory=dict)
    paints: List[NamedPose] = field(default_factory=list)
    brushes: List[BrushSlot] = field(default_factory=list)
    scan_poses: List[NamedPose] = field(default_factory=list)
    water_cleaning_station_pose: Optional[Pose] = None
    paper_towel_pose: Optional[Pose] = None
    named_poses: Dict[str, Pose] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        canvas_payload: Dict[str, Any] = {}
        for key in (
            "corner_tl_pose",
            "corner_tr_pose",
            "corner_br_pose",
            "corner_bl_pose",
        ):
            pose = self.canvas.get(key)
            if pose:
                canvas_payload[key] = pose.to_dict()
        if "touch_height_offset" in self.canvas:
            canvas_payload["touch_height_offset"] = self.canvas["touch_height_offset"]
        if "press_depth" in self.canvas:
            canvas_payload["press_depth"] = self.canvas["press_depth"]

        return {
            "home_pose": self.home_pose.to_dict() if self.home_pose else None,
            "safe_travel_pose": self.safe_travel_pose.to_dict() if self.safe_travel_pose else None,
            "canvas": canvas_payload,
            "paints": [pose.to_dict() for pose in self.paints],
            "brushes": [brush.to_dict() for brush in self.brushes],
            "scan_poses": [pose.to_dict() for pose in self.scan_poses],
            "water_cleaning_station_pose": self.water_cleaning_station_pose.to_dict()
            if self.water_cleaning_station_pose
            else None,
            "paper_towel_pose": self.paper_towel_pose.to_dict() if self.paper_towel_pose else None,
            "named_poses": {name: pose.to_dict() for name, pose in self.named_poses.items()},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Calibration":
        canvas_data = data.get("canvas") or {}
        canvas: Dict[str, Any] = {}
        for key in (
            "corner_tl_pose",
            "corner_tr_pose",
            "corner_br_pose",
            "corner_bl_pose",
        ):
            if key in canvas_data and canvas_data[key] is not None:
                canvas[key] = Pose.from_dict(canvas_data[key])
        if "touch_height_offset" in canvas_data:
            canvas["touch_height_offset"] = canvas_data.get("touch_height_offset")
        if "press_depth" in canvas_data:
            canvas["press_depth"] = canvas_data.get("press_depth")

        paints = [
            NamedPose(name=entry["name"], pose=Pose.from_dict(entry["pose"]))
            for entry in data.get("paints") or []
        ]
        brushes = []
        for entry in data.get("brushes") or []:
            brushes.append(
                BrushSlot(
                    name=entry["name"],
                    pick_pose=Pose.from_dict(entry["pick_pose"])
                    if entry.get("pick_pose")
                    else None,
                    place_pose=Pose.from_dict(entry["place_pose"])
                    if entry.get("place_pose")
                    else None,
                )
            )
        scan_poses = [
            NamedPose(name=entry["name"], pose=Pose.from_dict(entry["pose"]))
            for entry in data.get("scan_poses") or []
        ]
        named_poses = {
            name: Pose.from_dict(pose_data)
            for name, pose_data in (data.get("named_poses") or {}).items()
        }

        return cls(
            home_pose=Pose.from_dict(data["home_pose"]) if data.get("home_pose") else None,
            safe_travel_pose=Pose.from_dict(data["safe_travel_pose"])
            if data.get("safe_travel_pose")
            else None,
            canvas=canvas,
            paints=paints,
            brushes=brushes,
            scan_poses=scan_poses,
            water_cleaning_station_pose=Pose.from_dict(data["water_cleaning_station_pose"])
            if data.get("water_cleaning_station_pose")
            else None,
            paper_towel_pose=Pose.from_dict(data["paper_towel_pose"])
            if data.get("paper_towel_pose")
            else None,
            named_poses=named_poses,
        )

    @classmethod
    def load(cls, path: Path) -> "Calibration":
        if not path.exists():
            return cls()
        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_dict(raw)

    def save(self, path: Path) -> None:
        payload = self.to_dict()
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def set_pose(self, name: str, pose: Pose) -> None:
        if name == "home_pose":
            self.home_pose = pose
            return
        if name == "safe_travel_pose":
            self.safe_travel_pose = pose
            return
        if name == "water_cleaning_station_pose":
            self.water_cleaning_station_pose = pose
            return
        if name == "paper_towel_pose":
            self.paper_towel_pose = pose
            return
        if name.startswith("canvas."):
            canvas_key = name.split(".", 1)[1]
            self.canvas[canvas_key] = pose
            return
        if name.startswith("paints."):
            paint_name = name.split(".", 1)[1]
            self._set_named_pose(self.paints, paint_name, pose)
            return
        if name.startswith("scan."):
            scan_name = name.split(".", 1)[1]
            self._set_named_pose(self.scan_poses, scan_name, pose)
            return
        if name.startswith("brushes."):
            brush_name, pose_kind = self._parse_brush_name(name)
            brush = self._get_or_create_brush(brush_name)
            if pose_kind == "pick":
                brush.pick_pose = pose
            elif pose_kind == "place":
                brush.place_pose = pose
            return
        self.named_poses[name] = pose

    def delete_pose(self, name: str) -> bool:
        if name == "home_pose" and self.home_pose:
            self.home_pose = None
            return True
        if name == "safe_travel_pose" and self.safe_travel_pose:
            self.safe_travel_pose = None
            return True
        if name == "water_cleaning_station_pose" and self.water_cleaning_station_pose:
            self.water_cleaning_station_pose = None
            return True
        if name == "paper_towel_pose" and self.paper_towel_pose:
            self.paper_towel_pose = None
            return True
        if name.startswith("canvas."):
            canvas_key = name.split(".", 1)[1]
            if canvas_key in self.canvas:
                del self.canvas[canvas_key]
                return True
        if name.startswith("paints."):
            return self._delete_named_pose(self.paints, name.split(".", 1)[1])
        if name.startswith("scan."):
            return self._delete_named_pose(self.scan_poses, name.split(".", 1)[1])
        if name.startswith("brushes."):
            brush_name, pose_kind = self._parse_brush_name(name)
            for brush in self.brushes:
                if brush.name == brush_name:
                    if pose_kind == "pick" and brush.pick_pose:
                        brush.pick_pose = None
                        return True
                    if pose_kind == "place" and brush.place_pose:
                        brush.place_pose = None
                        return True
            return False
        if name in self.named_poses:
            del self.named_poses[name]
            return True
        return False

    def find_pose(self, name: str) -> Optional[Pose]:
        if name == "home_pose":
            return self.home_pose
        if name == "safe_travel_pose":
            return self.safe_travel_pose
        if name == "water_cleaning_station_pose":
            return self.water_cleaning_station_pose
        if name == "paper_towel_pose":
            return self.paper_towel_pose
        if name.startswith("canvas."):
            return self.canvas.get(name.split(".", 1)[1])
        if name.startswith("paints."):
            return self._find_named_pose(self.paints, name.split(".", 1)[1])
        if name.startswith("scan."):
            return self._find_named_pose(self.scan_poses, name.split(".", 1)[1])
        if name.startswith("brushes."):
            brush_name, pose_kind = self._parse_brush_name(name)
            for brush in self.brushes:
                if brush.name == brush_name:
                    return brush.pick_pose if pose_kind == "pick" else brush.place_pose
            return None
        return self.named_poses.get(name)

    def list_pose_names(self) -> List[str]:
        names: List[str] = []
        for key in (
            "home_pose",
            "safe_travel_pose",
            "water_cleaning_station_pose",
            "paper_towel_pose",
        ):
            if getattr(self, key):
                names.append(key)
        for key in (
            "corner_tl_pose",
            "corner_tr_pose",
            "corner_br_pose",
            "corner_bl_pose",
        ):
            if key in self.canvas:
                names.append(f"canvas.{key}")
        names.extend(f"paints.{pose.name}" for pose in self.paints)
        names.extend(f"scan.{pose.name}" for pose in self.scan_poses)
        for brush in self.brushes:
            if brush.pick_pose:
                names.append(f"brushes.{brush.name}_pick_pose")
            if brush.place_pose:
                names.append(f"brushes.{brush.name}_place_pose")
        names.extend(sorted(self.named_poses.keys()))
        return sorted(names)

    def _set_named_pose(self, bucket: List[NamedPose], name: str, pose: Pose) -> None:
        for entry in bucket:
            if entry.name == name:
                entry.pose = pose
                return
        bucket.append(NamedPose(name=name, pose=pose))

    def _delete_named_pose(self, bucket: List[NamedPose], name: str) -> bool:
        for index, entry in enumerate(bucket):
            if entry.name == name:
                del bucket[index]
                return True
        return False

    def _find_named_pose(self, bucket: List[NamedPose], name: str) -> Optional[Pose]:
        for entry in bucket:
            if entry.name == name:
                return entry.pose
        return None

    def _get_or_create_brush(self, name: str) -> BrushSlot:
        for brush in self.brushes:
            if brush.name == name:
                return brush
        brush = BrushSlot(name=name)
        self.brushes.append(brush)
        return brush

    def _parse_brush_name(self, name: str) -> Tuple[str, str]:
        tail = name.split(".", 1)[1]
        if tail.endswith("_pick_pose"):
            return tail[: -len("_pick_pose")], "pick"
        if tail.endswith("_place_pose"):
            return tail[: -len("_place_pose")], "place"
        raise ValueError(
            "Brush pose names must end with _pick_pose or _place_pose."
        )
