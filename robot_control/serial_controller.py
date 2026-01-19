from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

import serial


class RobotArmSerialError(RuntimeError):
    """Base error for serial communication issues."""


class RobotArmTimeoutError(RobotArmSerialError):
    """Raised when the arm does not respond in time."""


@dataclass
class RobotArmSerial:
    """High-level interface for the Robotic Arm Arduino serial protocol."""

    port: Optional[str] = None
    baud: int = 9600
    timeout_s: float = 5.0
    home_angles: Sequence[int] = field(default_factory=lambda: (90, 90, 90, 90, 90, 90, 90))
    logger: logging.Logger = field(default_factory=lambda: logging.getLogger(__name__))
    _serial: Optional[serial.Serial] = field(default=None, init=False, repr=False)
    _abort_wait: bool = field(default=False, init=False, repr=False)

    def connect(self, port: Optional[str] = None, baud: Optional[int] = None) -> None:
        """Open the serial connection."""
        if self._serial and self._serial.is_open:
            return
        self.port = port or self.port
        self.baud = baud or self.baud
        if not self.port:
            raise RobotArmSerialError("Serial port is not configured.")
        self.logger.info("Connecting to %s at %s baud", self.port, self.baud)
        self._serial = serial.Serial(self.port, self.baud, timeout=0.1)

    def reconnect(self) -> None:
        """Close and reopen the serial connection."""
        if self._serial and self._serial.is_open:
            self._serial.close()
        self._serial = None
        self.connect()

    def close(self) -> None:
        """Close the serial connection."""
        if self._serial and self._serial.is_open:
            self._serial.close()

    def set_mode(self, auto: bool) -> None:
        self._send_command(f"MODE {'1' if auto else '0'}")

    def set_speed(self, speed: int) -> None:
        if not 0 <= speed <= 100:
            raise ValueError("Speed must be between 0 and 100.")
        self._send_command(f"SPD {speed}")

    def send_servo(self, index: int, angle: int) -> None:
        if not 1 <= index <= 7:
            raise ValueError("Servo index must be 1..7.")
        if not 0 <= angle <= 180:
            raise ValueError("Servo angle must be 0..180.")
        self._send_command(f"{index} {angle}")

    def send_frame(self, angles: Iterable[int]) -> float:
        """Send seven servo angles and wait for DONE. Returns elapsed seconds."""
        angle_list = list(angles)
        if len(angle_list) != 7:
            raise ValueError("Frame must contain 7 servo angles.")
        self._abort_wait = False
        self.set_mode(True)
        for idx, angle in enumerate(angle_list, start=1):
            self.send_servo(idx, angle)
        return self._wait_for_done()

    def reset_home(self) -> float:
        """Send the default home pose and wait for DONE."""
        return self.send_frame(self.home_angles)

    def emergency_stop(self) -> None:
        """Stop automatic mode and abort any wait loop."""
        self._abort_wait = True
        try:
            self.set_speed(0)
        finally:
            self.set_mode(False)

    def _send_command(self, command: str) -> None:
        self._ensure_connected()
        message = f"{command}\n".encode("utf-8")
        self.logger.debug("Sending: %s", command)
        try:
            self._serial.write(message)
        except Exception as exc:  # pragma: no cover - passthrough error
            raise RobotArmSerialError(f"Failed to send command: {command}") from exc

    def _wait_for_done(self) -> float:
        self._ensure_connected()
        start = time.monotonic()
        while True:
            if self._abort_wait:
                raise RobotArmSerialError("Wait aborted by emergency stop.")
            if time.monotonic() - start > self.timeout_s:
                raise RobotArmTimeoutError("Timed out waiting for DONE.")
            line = self._readline()
            if not line:
                continue
            self.logger.debug("Received: %s", line)
            if line.strip().upper() == "DONE":
                return time.monotonic() - start

    def _readline(self) -> str:
        try:
            raw = self._serial.readline()
        except Exception as exc:  # pragma: no cover - passthrough error
            raise RobotArmSerialError("Failed to read from serial port.") from exc
        if not raw:
            return ""
        try:
            return raw.decode("utf-8", errors="ignore")
        except Exception:  # pragma: no cover - defensive
            return ""

    def _ensure_connected(self) -> None:
        if not self._serial or not self._serial.is_open:
            raise RobotArmSerialError("Serial connection is not open.")
