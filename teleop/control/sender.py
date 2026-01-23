# teleop/control/sender.py
import time
from typing import Sequence, Tuple
from .. import config  

def _vec6(x: Sequence[float]) -> Tuple[float, float, float, float, float, float]:
    if len(x) != 6:
        raise ValueError(f"Expected length 6, got {len(x)}")
    return (float(x[0]), float(x[1]), float(x[2]), float(x[3]), float(x[4]), float(x[5]))

def _clamp_int(x: int, lo: int, hi: int) -> int:
    return lo if x < lo else hi if x > hi else x

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

    GRIP_MAX_UM = int(getattr(config, "GRIP_MAX_UM", 60000))          # 60mm
    GRIP_INVERT = bool(getattr(config, "GRIP_INVERT", True))        
    GRIP_DEADBAND_UM = int(getattr(config, "GRIP_DEADBAND_UM", 200))  # 0.2mm
    GRIP_EFFORT = int(getattr(config, "GRIP_EFFORT", 2000))           # 0..5000

    grip_hw = int(round(grip_hw))
    grip_hw = _clamp_int(grip_hw, 0, 1000)

    grip_um = int(round(grip_hw * float(GRIP_MAX_UM) / 1000.0))
    grip_um = _clamp_int(grip_um, 0, GRIP_MAX_UM)

    if GRIP_INVERT:
        grip_um = GRIP_MAX_UM - grip_um

    prev_um = getattr(driver, "_prev_grip_um", None)
    if prev_um is None or abs(grip_um - int(prev_um)) >= GRIP_DEADBAND_UM:
        driver.set_gripper(position=grip_um, effort=GRIP_EFFORT, enable=True)
        driver._prev_grip_um = int(grip_um)

    return next_send