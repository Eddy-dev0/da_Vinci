import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from painterslicer.machines.robotarm_backend.kinematics_planar import PlanarCalibration, PlanarKinematics
from painterslicer.machines.robotarm_backend.paintcode_parser import PaintCodeParser


def test_paintcode_parser_basic():
    text = """
    # comment
    TOOL round_medium
    PRESSURE 0.5
    MOVE 10 20
    Z_UP
    """
    steps = PaintCodeParser(text).parse()
    assert [step.command for step in steps] == ["TOOL", "PRESSURE", "MOVE", "Z_UP"]


def test_planar_kinematics_bilinear_center():
    calibration = PlanarCalibration(
        canvas_width_mm=100,
        canvas_height_mm=100,
        corner_tl_hover=[0, 0, 0, 0, 0, 0, 0],
        corner_tr_hover=[10, 10, 10, 10, 10, 10, 10],
        corner_br_hover=[20, 20, 20, 20, 20, 20, 20],
        corner_bl_hover=[30, 30, 30, 30, 30, 30, 30],
        corner_tl_touch=[0, 0, 0, 0, 0, 0, 0],
        corner_tr_touch=[10, 10, 10, 10, 10, 10, 10],
        corner_br_touch=[20, 20, 20, 20, 20, 20, 20],
        corner_bl_touch=[30, 30, 30, 30, 30, 30, 30],
    )
    kin = PlanarKinematics(calibration)
    pose = kin.canvas_to_servo_pose(50, 50, z_state="hover")
    assert pose == [15, 15, 15, 15, 15, 15, 15]


def test_planar_kinematics_pressure_interpolation():
    calibration = PlanarCalibration(
        canvas_width_mm=100,
        canvas_height_mm=100,
        corner_tl_hover=[0, 0, 0, 0, 0, 0, 0],
        corner_tr_hover=[0, 0, 0, 0, 0, 0, 0],
        corner_br_hover=[0, 0, 0, 0, 0, 0, 0],
        corner_bl_hover=[0, 0, 0, 0, 0, 0, 0],
        corner_tl_touch=[100, 100, 100, 100, 100, 100, 100],
        corner_tr_touch=[100, 100, 100, 100, 100, 100, 100],
        corner_br_touch=[100, 100, 100, 100, 100, 100, 100],
        corner_bl_touch=[100, 100, 100, 100, 100, 100, 100],
    )
    kin = PlanarKinematics(calibration)
    pose = kin.canvas_to_servo_pose(0, 0, z_state="pressure", pressure=0.25)
    assert pose == [25, 25, 25, 25, 25, 25, 25]
