# teleop/control/sender.py
import time
from ..utils.conversions import rad6_to_piper_int6

def piper_send(driver, dry_run: bool, last_q, grip_um: int, next_send: float, send_period: float, rad_to_piper: float):
    now = time.monotonic()
    if now < next_send:
        return next_send

    while next_send <= now:
        next_send += send_period

    joint_int = rad6_to_piper_int6(last_q, rad_to_piper)

    if dry_run:
        print(f"[DRY RUN] JointCtrl{tuple(joint_int)}")
    else:
        driver.send_joints(joint_int)
        driver.set_gripper(position=grip_um, effort=2000, enable=True)

    return next_send
