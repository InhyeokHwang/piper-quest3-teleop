# teleop/control/sender.py
import time
from typing import Optional, Sequence, Tuple

def _vec6(x: Sequence[float]) -> Tuple[float, float, float, float, float, float]:
    if len(x) != 6:
        raise ValueError(f"Expected length 6, got {len(x)}")
    return (float(x[0]), float(x[1]), float(x[2]), float(x[3]), float(x[4]), float(x[5]))

def piper_send_jointctrl(
    driver,
    dry_run: bool,
    last_q,                 # (6,) rad
    grip_um: int,
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

    prev_grip = getattr(driver, "_prev_grip_um", None)
    if prev_grip is None or abs(int(grip_um) - int(prev_grip)) > 50:
        driver.set_gripper(position=grip_um, effort=2000, enable=True)
        driver._prev_grip_um = int(grip_um)

    return next_send


def piper_send_mit(
    driver,
    dry_run: bool,
    last_q,                 # (6,) rad
    grip_um: int,
    next_send: float,
    send_period: float,
    *,
    kp: float = 5.0,
    kd: float = 0.1,
    tau_ref: float = 0.0,
):
    now = time.monotonic()
    if now < next_send:
        return next_send
    while next_send <= now:
        next_send += send_period

    q_ref = _vec6(last_q)
    dq_ref = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    if driver is None:
        if dry_run:
            print(f"[DRY RUN] (no driver) MIT q={q_ref} kp={kp} kd={kd}")
        return next_send

    if dry_run:
        print(f"[DRY RUN] send_joints_mit q={q_ref} dq={dq_ref} kp={kp} kd={kd} tau={tau_ref}")
    else:
        driver.send_joints_mit(q_ref=q_ref, dq_ref=dq_ref, kp=kp, kd=kd, tau_ref=tau_ref)

        prev_grip = getattr(driver, "_prev_grip_um", None)
        if prev_grip is None or abs(int(grip_um) - int(prev_grip)) > 50:
            driver.set_gripper(position=grip_um, effort=2000, enable=True)
            driver._prev_grip_um = int(grip_um)

    return next_send