from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, Optional

from robot_control.serial_controller import RobotArmSerial, RobotArmSerialError, RobotArmTimeoutError


@dataclass
class SerialHandshake:
    controller: RobotArmSerial
    logger: logging.Logger

    def send_frame(self, angles: Iterable[int]) -> None:
        try:
            elapsed = self.controller.send_frame(angles)
            self.logger.debug("Move done in %.2fs", elapsed)
        except RobotArmTimeoutError as exc:
            self.logger.error("Move timed out: %s", exc)
            self._abort()
            raise
        except RobotArmSerialError as exc:
            self.logger.error("Serial error: %s", exc)
            self._abort()
            raise

    def emergency_stop(self) -> None:
        self._abort()

    def _abort(self) -> None:
        try:
            self.controller.emergency_stop()
        except RobotArmSerialError as exc:
            self.logger.error("Failed to stop controller: %s", exc)


def connect_controller(
    port: str,
    baud: int,
    timeout_s: float,
    speed: Optional[int],
    logger: logging.Logger,
) -> SerialHandshake:
    controller = RobotArmSerial(port=port, baud=baud, timeout_s=timeout_s, logger=logger)
    controller.connect()
    if speed is not None:
        controller.set_speed(speed)
    return SerialHandshake(controller=controller, logger=logger)
