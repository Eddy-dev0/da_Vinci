"""Robot arm backend for executing PaintCode."""

__all__ = ["RobotArmBackend", "PlanarKinematics", "PaintCodeParser", "PaintStep"]


def __getattr__(name: str):
    if name == "RobotArmBackend":
        from painterslicer.machines.robotarm_backend.backend import RobotArmBackend

        return RobotArmBackend
    if name == "PlanarKinematics":
        from painterslicer.machines.robotarm_backend.kinematics_planar import PlanarKinematics

        return PlanarKinematics
    if name == "PaintCodeParser":
        from painterslicer.machines.robotarm_backend.paintcode_parser import PaintCodeParser

        return PaintCodeParser
    if name == "PaintStep":
        from painterslicer.machines.robotarm_backend.paintcode_parser import PaintStep

        return PaintStep
    raise AttributeError(name)
