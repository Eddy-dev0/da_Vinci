from __future__ import annotations

import argparse
import logging
import time

from .serial_controller import RobotArmSerial, RobotArmSerialError, RobotArmTimeoutError


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Robot Arm Serial CLI")
    parser.add_argument("--port", required=True, help="Serial port (e.g., COM3 or /dev/ttyUSB0)")
    parser.add_argument("--baud", type=int, default=9600, help="Serial baud rate")
    parser.add_argument("--speed", type=int, help="Speed 0..100")
    parser.add_argument("--timeout", type=float, default=5.0, help="DONE timeout in seconds")
    parser.add_argument("--home", action="store_true", help="Send home pose")
    parser.add_argument(
        "--frame",
        type=str,
        help='Comma-separated 7 angles, e.g. "90,80,70,90,90,90,90"',
    )
    return parser.parse_args()


def _parse_frame(frame: str) -> list[int]:
    parts = [p.strip() for p in frame.split(",") if p.strip()]
    angles = [int(p) for p in parts]
    if len(angles) != 7:
        raise ValueError("Frame must contain 7 comma-separated angles.")
    return angles


def main() -> int:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger("robot_control.cli")

    controller = RobotArmSerial(port=args.port, baud=args.baud, timeout_s=args.timeout)
    try:
        controller.connect()
        logger.info("Connected to %s", args.port)
        if args.speed is not None:
            controller.set_speed(args.speed)
        if args.home:
            start = time.monotonic()
            elapsed = controller.reset_home()
            logger.info("Home complete in %.2fs (elapsed %.2fs)", elapsed, time.monotonic() - start)
        if args.frame:
            angles = _parse_frame(args.frame)
            start = time.monotonic()
            elapsed = controller.send_frame(angles)
            logger.info("Frame complete in %.2fs (elapsed %.2fs)", elapsed, time.monotonic() - start)
    except RobotArmTimeoutError as exc:
        logger.error("Timeout: %s", exc)
        return 2
    except (RobotArmSerialError, ValueError) as exc:
        logger.error("Error: %s", exc)
        return 1
    finally:
        controller.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
