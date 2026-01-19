import itertools
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.modules.setdefault("serial", SimpleNamespace(Serial=object))

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robot_control.serial_controller import RobotArmSerial, RobotArmTimeoutError


class MockSerial:
    def __init__(self, read_lines=None):
        self.read_lines = list(read_lines or [])
        self.written = []
        self.is_open = True

    def write(self, data):
        self.written.append(data)

    def readline(self):
        if self.read_lines:
            return self.read_lines.pop(0)
        return b""

    def close(self):
        self.is_open = False


def test_command_formatting(monkeypatch):
    mock_serial = MockSerial()

    def fake_serial(*args, **kwargs):
        return mock_serial

    monkeypatch.setattr("robot_control.serial_controller.serial.Serial", fake_serial)
    controller = RobotArmSerial(port="COM1")
    controller.connect()

    controller.set_mode(True)
    controller.set_speed(75)
    controller.send_servo(3, 120)

    assert mock_serial.written == [
        b"MODE 1\n",
        b"SPD 75\n",
        b"3 120\n",
    ]


def test_send_frame_waits_for_done(monkeypatch):
    mock_serial = MockSerial(read_lines=[b"", b"DONE\n"])

    def fake_serial(*args, **kwargs):
        return mock_serial

    monkeypatch.setattr("robot_control.serial_controller.serial.Serial", fake_serial)
    controller = RobotArmSerial(port="COM1", timeout_s=1.0)
    controller.connect()

    elapsed = controller.send_frame([90, 90, 90, 90, 90, 90, 90])

    assert elapsed >= 0
    assert mock_serial.written[0] == b"MODE 1\n"
    assert mock_serial.written[-1] == b"7 90\n"


def test_send_frame_timeout(monkeypatch):
    mock_serial = MockSerial(read_lines=[b"", b""])

    def fake_serial(*args, **kwargs):
        return mock_serial

    monotonic_values = itertools.chain([0.0, 0.2, 0.4, 0.6, 1.2])

    def fake_monotonic():
        return next(monotonic_values)

    monkeypatch.setattr("robot_control.serial_controller.serial.Serial", fake_serial)
    monkeypatch.setattr("robot_control.serial_controller.time.monotonic", fake_monotonic)

    controller = RobotArmSerial(port="COM1", timeout_s=1.0)
    controller.connect()

    with pytest.raises(RobotArmTimeoutError):
        controller.send_frame([90, 90, 90, 90, 90, 90, 90])
