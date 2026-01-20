# teleop/control/sender.py
import time
from typing import Sequence, Tuple

def _vec6(x: Sequence[float]) -> Tuple[float, float, float, float, float, float]:
    if len(x) != 6:
        raise ValueError(f"Expected length 6, got {len(x)}")
    return (float(x[0]), float(x[1]), float(x[2]), float(x[3]), float(x[4]), float(x[5]))

def piper_send_jointctrl(
    driver,
    dry_run: bool,
    last_q,                 # (6,) rad
    grip_hw: int,
    next_send: float,
    send_period: float,
    rad_to_piper: float,    # rad -> piper int unit
):
    now = time.monotonic()
    if now < next_send:
        return next_send
    while next_send <= now:
        next_send += send_period

    q = _vec6(last_q)

    # rad -> piper internal int
    joint_int = [int(round(q[i] * rad_to_piper)) for i in range(6)]

    if dry_run:
        # print(f"[DRY RUN] JointCtrl joint_int={joint_int}")
        return next_send
    
    if driver is None:
        return next_send

    driver.send_joints(joint_int)

    prev_grip = getattr(driver, "_prev_grip_hw", None)
    if prev_grip is None or abs(int(grip_hw) - int(prev_grip)) > 50:
        driver.set_gripper(position=grip_hw, effort=2000, enable=True)
        driver._prev_grip_hw = int(grip_hw)

    return next_send
