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

## Python API

```python
from robot_control import RobotArmSerial

controller = RobotArmSerial(port="COM3", baud=9600, timeout_s=5.0)
controller.connect()
controller.set_speed(60)
controller.send_frame([90, 80, 90, 100, 90, 90, 90])
controller.close()
```
