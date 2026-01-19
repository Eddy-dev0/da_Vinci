# Robot Arm Serial Control

This module wraps the Arduino serial protocol so new tooling (vision, planning) can send motion
frames without touching the legacy Windows EXE.

## Serial Protocol

The Arduino firmware listens on a serial port (default assumed `9600` baud) with newline-delimited
commands:

| Command | Meaning |
| --- | --- |
| `MODE 1` | Enable automatic mode (frame execution). |
| `MODE 0` | Manual mode / stop automatic execution. |
| `SPD <0..100>` | Set global speed percentage. |
| `<index 1..7> <angle 0..180>` | Set servo angle for a given channel. |

**Frame behavior:** In automatic mode, the firmware waits until it has received seven servo
commands. Once the target positions are reached, the Arduino sends `DONE` back over serial.

## Servo Index Mapping

According to the firmware comments the index mapping is:

1. Garra
2. Muñeca1
3. Muñeca2
4. BrazoSup
5. BrazoInf
6. Antebrazo
7. Base

**Note:** The antebrazo has an `OPPOSITE_SERVO_CHANNEL` defined in the firmware, so its paired
servo automatically mirrors with `180 - angle`.

## CLI Usage

```bash
python -m robot_control.cli --port COM3 --home
python -m robot_control.cli --port COM3 --frame "90,80,90,100,90,90,90" --speed 60
```

## Calibration / Teach CLI

The teach CLI lets you jog the arm, save poses into a `calibration.json`, and recall them later.

```bash
python -m robot_control.teach_cli --port COM3 --calibration calibration.json
```

### Calibration JSON Schema

```json
{
  "home_pose": {"angles": [90, 90, 90, 90, 90, 90, 90], "note": "optional", "tool_id": "optional"},
  "safe_travel_pose": {"angles": [90, 90, 90, 90, 90, 90, 90]},
  "canvas": {
    "corner_tl_pose": {"angles": [90, 90, 90, 90, 90, 90, 90]},
    "corner_tr_pose": {"angles": [90, 90, 90, 90, 90, 90, 90]},
    "corner_br_pose": {"angles": [90, 90, 90, 90, 90, 90, 90]},
    "corner_bl_pose": {"angles": [90, 90, 90, 90, 90, 90, 90]},
    "touch_height_offset": 2.5,
    "press_depth": 1.2
  },
  "paints": [
    {"name": "paint_red_pose", "pose": {"angles": [90, 90, 90, 90, 90, 90, 90]}}
  ],
  "brushes": [
    {
      "name": "brush_1",
      "pick_pose": {"angles": [90, 90, 90, 90, 90, 90, 90]},
      "place_pose": {"angles": [90, 90, 90, 90, 90, 90, 90]}
    }
  ],
  "scan_poses": [
    {"name": "scan_front", "pose": {"angles": [90, 90, 90, 90, 90, 90, 90]}}
  ],
  "water_cleaning_station_pose": {"angles": [90, 90, 90, 90, 90, 90, 90]},
  "paper_towel_pose": {"angles": [90, 90, 90, 90, 90, 90, 90]},
  "named_poses": {
    "custom_pose": {"angles": [90, 90, 90, 90, 90, 90, 90]}
  }
}
```

### Teach CLI Commands (interactive)

- `list`: list saved poses.
- `save <pose_name>`: save current angles (supports `home_pose`, `safe_travel_pose`, `canvas.corner_tl_pose`,
  `paints.paint_red_pose`, `brushes.brush_1_pick_pose`, `scan.scan_front`, or any custom name).
- `go <pose_name>`: move to a saved pose and wait for `DONE`.
- `jog <servo_index> <+|-> [step]`: jog a servo by the step size.
- `step <degrees>`: set jog step size.
- `set <servo_index> <angle>`: set an individual servo angle (clamped 0..180).
- `frame <a1,a2,a3,a4,a5,a6,a7>`: send a full frame (clamped 0..180).
- `manual` / `auto`: toggle `MODE 0` or `MODE 1`.
- `note <pose_name> <text>`: store a note on a pose.
- `tool <pose_name> <tool_id>`: store a tool identifier on a pose.
- `set_canvas <touch_height_offset|press_depth> <value>`: update canvas metadata.
- `delete <pose_name>`: delete a pose.
- `export [path]`: write calibration JSON to a file.
- `angles`: print the last known angles.
- `quit` / `exit`: leave the CLI.

## Python API

```python
from robot_control import RobotArmSerial

controller = RobotArmSerial(port="COM3", baud=9600, timeout_s=5.0)
controller.connect()
controller.set_speed(60)
controller.send_frame([90, 80, 90, 100, 90, 90, 90])
controller.close()
```
