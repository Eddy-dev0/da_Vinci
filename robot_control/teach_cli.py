from __future__ import annotations

import argparse
import cmd
import json
import logging
from pathlib import Path
from typing import List, Optional

from .calibration import Calibration, Pose, clamp_angle, clamp_angles
from .serial_controller import RobotArmSerial, RobotArmSerialError, RobotArmTimeoutError


class CalibrationShell(cmd.Cmd):
    intro = "Robotic Arm Teach Mode. Type 'help' or '?' to list commands."
    prompt = "calibrate> "

    def __init__(
        self,
        controller: RobotArmSerial,
        calibration: Calibration,
        calibration_path: Path,
    ) -> None:
        super().__init__()
        self.controller = controller
        self.calibration = calibration
        self.calibration_path = calibration_path
        self.current_angles: List[int] = list(controller.home_angles)
        self.step_size = 5

    def do_list(self, arg: str) -> None:
        """List saved pose names."""
        names = self.calibration.list_pose_names()
        if not names:
            print("No poses saved yet.")
            return
        for name in names:
            print(name)

    def do_show(self, arg: str) -> None:
        """Show pose data for a pose name."""
        name = arg.strip()
        if not name:
            print("Usage: show <pose_name>")
            return
        pose = self.calibration.find_pose(name)
        if not pose:
            print(f"Pose '{name}' not found.")
            return
        print(json.dumps(pose.to_dict(), indent=2))

    def do_save(self, arg: str) -> None:
        """Save current angles as a named pose: save <pose_name>"""
        name = arg.strip()
        if not name:
            print("Usage: save <pose_name>")
            return
        pose = Pose(angles=clamp_angles(self.current_angles))
        self.calibration.set_pose(name, pose)
        self.calibration.save(self.calibration_path)
        print(f"Saved pose '{name}'.")

    def do_delete(self, arg: str) -> None:
        """Delete a pose: delete <pose_name>"""
        name = arg.strip()
        if not name:
            print("Usage: delete <pose_name>")
            return
        if self.calibration.delete_pose(name):
            self.calibration.save(self.calibration_path)
            print(f"Deleted pose '{name}'.")
        else:
            print(f"Pose '{name}' not found.")

    def do_go(self, arg: str) -> None:
        """Move robot to a saved pose: go <pose_name>"""
        name = arg.strip()
        if not name:
            print("Usage: go <pose_name>")
            return
        pose = self.calibration.find_pose(name)
        if not pose:
            print(f"Pose '{name}' not found.")
            return
        self._send_frame(pose.angles)

    def do_angles(self, arg: str) -> None:
        """Print the last known angles."""
        print(",".join(str(angle) for angle in self.current_angles))

    def do_manual(self, arg: str) -> None:
        """Set manual MODE 0 (manual control)."""
        self.controller.set_mode(False)
        print("Manual mode enabled (MODE 0).")

    def do_auto(self, arg: str) -> None:
        """Set automatic MODE 1."""
        self.controller.set_mode(True)
        print("Auto mode enabled (MODE 1).")

    def do_step(self, arg: str) -> None:
        """Set jog step size in degrees: step <1|5|10>"""
        try:
            step = int(arg.strip())
        except ValueError:
            print("Usage: step <integer>")
            return
        if step <= 0:
            print("Step size must be positive.")
            return
        self.step_size = step
        print(f"Step size set to {self.step_size}°.")

    def do_jog(self, arg: str) -> None:
        """Jog a servo by step size: jog <servo_index> <+|-> [step]"""
        parts = arg.split()
        if len(parts) < 2:
            print("Usage: jog <servo_index 1..7> <+|-> [step]")
            return
        try:
            servo_index = int(parts[0])
        except ValueError:
            print("Servo index must be an integer.")
            return
        if not 1 <= servo_index <= 7:
            print("Servo index must be 1..7.")
            return
        direction = parts[1]
        if direction not in {"+", "-"}:
            print("Direction must be + or -.")
            return
        step = self.step_size
        if len(parts) >= 3:
            try:
                step = int(parts[2])
            except ValueError:
                print("Step must be an integer.")
                return
        delta = step if direction == "+" else -step
        new_angle = clamp_angle(self.current_angles[servo_index - 1] + delta)
        self.current_angles[servo_index - 1] = new_angle
        self._send_servo(servo_index, new_angle)

    def do_set(self, arg: str) -> None:
        """Set a servo to a specific angle: set <servo_index> <angle>"""
        parts = arg.split()
        if len(parts) != 2:
            print("Usage: set <servo_index 1..7> <angle>")
            return
        try:
            servo_index = int(parts[0])
            angle = int(parts[1])
        except ValueError:
            print("Servo index and angle must be integers.")
            return
        if not 1 <= servo_index <= 7:
            print("Servo index must be 1..7.")
            return
        angle = clamp_angle(angle)
        self.current_angles[servo_index - 1] = angle
        self._send_servo(servo_index, angle)

    def do_frame(self, arg: str) -> None:
        """Send full frame: frame <a1,a2,a3,a4,a5,a6,a7>"""
        frame_arg = arg.strip()
        if not frame_arg:
            print("Usage: frame <a1,a2,a3,a4,a5,a6,a7>")
            return
        parts = [p.strip() for p in frame_arg.split(",") if p.strip()]
        if len(parts) != 7:
            print("Frame requires 7 comma-separated angles.")
            return
        try:
            angles = [int(value) for value in parts]
        except ValueError:
            print("Frame angles must be integers.")
            return
        angles = clamp_angles(angles)
        self._send_frame(angles)

    def do_export(self, arg: str) -> None:
        """Export calibration JSON to a path: export [path]"""
        path_str = arg.strip()
        target_path = Path(path_str) if path_str else self.calibration_path
        self.calibration.save(target_path)
        print(f"Exported calibration to {target_path}.")

    def do_note(self, arg: str) -> None:
        """Attach a note to a pose: note <pose_name> <note text>"""
        parts = arg.split(maxsplit=1)
        if len(parts) != 2:
            print("Usage: note <pose_name> <note text>")
            return
        name, note_text = parts
        pose = self.calibration.find_pose(name)
        if not pose:
            print(f"Pose '{name}' not found.")
            return
        pose.note = note_text
        self.calibration.set_pose(name, pose)
        self.calibration.save(self.calibration_path)
        print(f"Updated note for '{name}'.")

    def do_tool(self, arg: str) -> None:
        """Attach a tool id to a pose: tool <pose_name> <tool_id>"""
        parts = arg.split(maxsplit=1)
        if len(parts) != 2:
            print("Usage: tool <pose_name> <tool_id>")
            return
        name, tool_id = parts
        pose = self.calibration.find_pose(name)
        if not pose:
            print(f"Pose '{name}' not found.")
            return
        pose.tool_id = tool_id
        self.calibration.set_pose(name, pose)
        self.calibration.save(self.calibration_path)
        print(f"Updated tool_id for '{name}'.")

    def do_set_canvas(self, arg: str) -> None:
        """Set canvas values: set_canvas <touch_height_offset|press_depth> <value>"""
        parts = arg.split()
        if len(parts) != 2:
            print("Usage: set_canvas <touch_height_offset|press_depth> <value>")
            return
        key, value_str = parts
        if key not in {"touch_height_offset", "press_depth"}:
            print("Canvas key must be touch_height_offset or press_depth.")
            return
        try:
            value = float(value_str)
        except ValueError:
            print("Canvas value must be numeric.")
            return
        self.calibration.canvas[key] = value
        self.calibration.save(self.calibration_path)
        print(f"Updated canvas {key} to {value}.")

    def do_reset(self, arg: str) -> None:
        """Reset current angles to last home pose or defaults."""
        pose = self.calibration.home_pose
        if pose:
            self._send_frame(pose.angles)
        else:
            self._send_frame(list(self.controller.home_angles))

    def do_quit(self, arg: str) -> bool:
        """Quit the calibration shell."""
        return True

    def do_exit(self, arg: str) -> bool:
        """Quit the calibration shell."""
        return True

    def do_EOF(self, arg: str) -> bool:  # noqa: N802
        return True

    def _send_servo(self, servo_index: int, angle: int) -> None:
        try:
            self.controller.set_mode(False)
            self.controller.send_servo(servo_index, angle)
            print(f"Servo {servo_index} -> {angle}°")
        except RobotArmSerialError as exc:
            print(f"Serial error: {exc}")

    def _send_frame(self, angles: List[int]) -> None:
        try:
            elapsed = self.controller.send_frame(angles)
            self.current_angles = list(angles)
            print(f"Frame complete in {elapsed:.2f}s.")
        except RobotArmTimeoutError as exc:
            print(f"Timeout: {exc}")
        except RobotArmSerialError as exc:
            print(f"Serial error: {exc}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Robot Arm Calibration/Teach CLI")
    parser.add_argument("--port", required=True, help="Serial port (e.g., COM3 or /dev/ttyUSB0)")
    parser.add_argument("--baud", type=int, default=9600, help="Serial baud rate")
    parser.add_argument("--speed", type=int, default=60, help="Speed 0..100")
    parser.add_argument(
        "--calibration",
        type=Path,
        default=Path("calibration.json"),
        help="Calibration JSON path",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=5.0,
        help="DONE timeout in seconds",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    controller = RobotArmSerial(port=args.port, baud=args.baud, timeout_s=args.timeout)
    calibration = Calibration.load(args.calibration)
    if args.speed is not None:
        controller.home_angles = clamp_angles(controller.home_angles)
    try:
        controller.connect()
        controller.set_speed(args.speed)
        shell = CalibrationShell(controller, calibration, args.calibration)
        shell.cmdloop()
    except RobotArmSerialError as exc:
        logging.error("Serial error: %s", exc)
        return 1
    finally:
        controller.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
