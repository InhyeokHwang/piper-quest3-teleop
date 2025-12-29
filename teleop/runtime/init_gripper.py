# teleop/runtime/init_gripper.py
from ..controllers.gripper import GripperController, GripperConfig

def init_gripper():
    return GripperController(GripperConfig(
        mode="analog",
        # 필요하면 스케일 조정
        out_min=0,
        out_max=1000,
        close_when_high=True,
        alpha=0.35,
    ))