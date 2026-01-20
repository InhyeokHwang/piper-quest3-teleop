# teleop/control/gripper_stepper.py
from dataclasses import dataclass
from .. import config

@dataclass
class GripperOutputs:
    joint7: float
    joint8: float
    grip_hw: int
    open_ratio: float
    gripper_pos: int

def step_gripper(gripper_ctl, teleoperator) -> GripperOutputs:
    gripper_pos = gripper_ctl.update(teleoperator)  # 0..1000
    t = gripper_pos / 1000.0
    open_ratio = 1.0 - t

    joint7 =  config.SIM_GRIPPER_RANGE * open_ratio
    joint8 = -config.SIM_GRIPPER_RANGE * open_ratio
    grip_hw = int(round(config.GRIPPER_MAX_UM * open_ratio))

    return GripperOutputs(joint7, joint8, grip_hw, open_ratio, gripper_pos)
