# Robot Arm Backend (Arduino COM3)

## Überblick
Der `robotarm_backend` übersetzt PaintCode in Servo-Frames und nutzt die bestehende Arduino-Serial-Firmware.
Der Planner bleibt **PainterSlicer**, das Backend kümmert sich um Kinematik, Tool-Handling und Kamera-Scans.

## Setup
1. **Serial (COM3, 9600 Baud)**
   - Standard: `COM3` / `9600`
   - Einstellbar in `config/settings.yaml` unter `robotarm`.
2. **Kamera (Aruco Scan)**
   - Default: Kamera-Index `0`
   - Optional: Kamera-Kalibrierung als JSON (OpenCV Export).

## ArUco Marker IDs (Canvas)
Standard-IDs (an den Canvas-Ecken):
- TL = 0, TR = 1, BR = 2, BL = 3

## calibration.json Beispiel
`data/calibration.json` enthält die vier Ecken als Hover/Touch-Pose (je 7 Winkel). Beispiel:
```json
{
  "canvas_width_mm": 400,
  "canvas_height_mm": 300,
  "corner_tl_hover": [90,90,90,90,90,90,90],
  "corner_tr_hover": [95,85,90,90,90,90,90],
  "corner_br_hover": [100,80,90,90,90,90,90],
  "corner_bl_hover": [85,95,90,90,90,90,90],
  "corner_tl_touch": [90,90,80,90,90,90,90],
  "corner_tr_touch": [95,85,80,90,90,90,90],
  "corner_br_touch": [100,80,80,90,90,90,90],
  "corner_bl_touch": [85,95,80,90,90,90,90]
}
```

## inventory.json Beispiel
`data/inventory.json` listet Farben und Tools:
```json
{
  "paints": [
    {"id": 1, "name": "Primary Red", "hsv": [0,255,255], "lab": [136,208,195], "dip_pose_name": "paint_red_dip"}
  ],
  "tools": [
    {"id": 101, "tool_type": "ROUND", "size_class": "medium", "size_mm": 10.0, "pickup_pose_name": "brush_round_medium_pick"}
  ]
}
```

## Minimaler PaintCode (Hello Stroke)
```
TOOL round_medium
PRESSURE 0.7
Z_DOWN
MOVE 10 10
MOVE 100 10
MOVE 100 50
Z_UP
```

## CLI: Canvas Scan (Aruco)
```bash
python -m painterslicer.machines.robotarm_backend.vision_aruco \
  --camera 0 \
  --canvas-width 400 \
  --canvas-height 300 \
  --output canvas_homography.json
```

## CLI: Inventory Scan
```bash
python -m painterslicer.machines.robotarm_backend.inventory_scan \
  --camera 0 \
  --output data/inventory.json
```

## CLI: Dry-Run Preview
```bash
python -m painterslicer.machines.robotarm_backend.preview \
  job.paintcode \
  --canvas-width 400 \
  --canvas-height 300 \
  --output preview.png
```

## Backend Execution (Python)
```python
from pathlib import Path
from painterslicer.machines.robotarm_backend.backend import RobotArmBackend, load_settings

settings = load_settings(Path("config/settings.yaml"))
backend = RobotArmBackend(
    settings=settings,
    calibration_path=Path("data/calibration.json"),
    inventory_path=Path("data/inventory.json"),
)
result = backend.run_paintcode(Path("job.paintcode"), dry_run=True, preview_path=Path("preview.png"))
print(result)
```

## Windows Run Commands
```powershell
python -m painterslicer.machines.robotarm_backend.preview job.paintcode --canvas-width 400 --canvas-height 300 --output preview.png
python -m painterslicer.machines.robotarm_backend.vision_aruco --camera 0 --canvas-width 400 --canvas-height 300 --output canvas_homography.json
python -m painterslicer.machines.robotarm_backend.inventory_scan --camera 0 --output data/inventory.json
```

## Fehlerfälle
- **COM3 belegt**: Backend meldet einen klaren Serial-Fehler.
- **DONE Timeout**: serial_controller wirft `RobotArmTimeoutError`.
- **Marker fehlen**: Homographie wird nicht geschrieben.
- **MOVE außerhalb Canvas**: Backend wirft ValueError (wenn `allow_out_of_bounds=false`).
