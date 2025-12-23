# piper/safety.py

from __future__ import annotations

import time
from typing import Sequence, Optional

from .driver import PiperDriver


def enable_and_wait(
    driver: PiperDriver,
    timeout_s: float = 5.0,
    poll_dt_s: float = 1.0,
    also_open_gripper: bool = True,
    fail_hard: bool = True,
) -> bool:
    """
    Enable motors and wait until all 6 motor drivers report enabled.

    Returns
    -------
    bool
        True if enabled within timeout, False otherwise.

    Notes
    -----
    - This corresponds to your previous enable_fun().
    - fail_hard=True will raise RuntimeError on timeout (instead of exit()).
    """
    if not driver.connected:
        raise RuntimeError("PiperDriver must be connected before enable_and_wait().")

    start = time.time()

    while True:
        # Try enabling each loop (SDK style)
        driver.enable()

        if also_open_gripper:
            # keep same behavior as your code
            try:
                driver.set_gripper(position=0, effort=2000, enable=True)
            except Exception as e:
                # gripper might fail depending on state; don't crash here
                print("[safety] Gripper open command failed:", e)

        enabled = driver.is_enabled()
        print("[safety] Enable status:", enabled)

        if enabled:
            return True

        if (time.time() - start) > timeout_s:
            msg = f"[safety] Enable timeout after {timeout_s:.1f}s."
            if fail_hard:
                raise RuntimeError(msg)
            print(msg)
            return False

        time.sleep(poll_dt_s)


def read_joint_radians(driver: PiperDriver, factor: float) -> Optional[list[float]]:
    """
    Read joint positions in radians using SDK message layout.
    Returns None if message structure is unknown.
    """
    msg = driver.get_joint_positions_raw()
    try:
        js = msg.joint_state
        return [
            js.joint_1 / factor,
            js.joint_2 / factor,
            js.joint_3 / factor,
            js.joint_4 / factor,
            js.joint_5 / factor,
            js.joint_6 / factor,
        ]
    except AttributeError:
        return None


def move_to_start_pose(
    driver: PiperDriver,
    start_position_rad: Sequence[float],
    factor: float,
    steps: int = 200,
    step_dt_s: float = 0.01,
    pos_tol_rad: float = 0.01,
    max_wait_s: float = 2.0,
    motion_speed: int = 20,
    check_reached: bool = True,
) -> bool:
    """
    Move arm to a safe/start pose (joint space interpolation), then optionally check reached.

    Parameters
    ----------
    driver : PiperDriver
        Connected driver
    start_position_rad : Sequence[float]
        length >= 6, first 6 are joint targets in rad
    factor : float
        rad -> piper int unit factor (your FACTOR)
    steps : int
        interpolation steps
    step_dt_s : float
        sleep per step
    pos_tol_rad : float
        tolerance to consider reached
    max_wait_s : float
        timeout for final reach check
    motion_speed : int
        MotionCtrl_2 speed used before moving (matches your code's intent)
    check_reached : bool
        If False, skip final readback loop.

    Returns
    -------
    bool
        True if reached (or check_reached=False), False otherwise.
    """
    if not driver.connected:
        raise RuntimeError("PiperDriver must be connected before move_to_start_pose().")

    if len(start_position_rad) < 6:
        raise ValueError("start_position_rad must have at least 6 joint values (rad).")

    target = list(start_position_rad[:6])

    print("[safety] Moving to START_POSITION...")

    # put into joint position control mode (same call pattern as your original)
    try:
        driver.set_motion_mode(ctrl_mode=0x01, move_mode=0x01, speed=motion_speed, acc=0x00)
    except Exception as e:
        print("[safety] MotionCtrl_2 failed (continuing):", e)

    current = read_joint_radians(driver, factor)
    if current is None:
        print("[safety][WARN] Joint message structure unknown; using zeros as current.")
        current = [0.0] * 6

    # Interpolate and send
    for i in range(steps):
        alpha = (i + 1) / steps
        interp = [current[j] + alpha * (target[j] - current[j]) for j in range(6)]
        joint_int = [round(interp[j] * factor) for j in range(6)]
        driver.send_joints(joint_int)
        time.sleep(step_dt_s)

    if not check_reached:
        return True

    # Reached check
    print("[safety] Checking if arm reached start pose...")
    t0 = time.time()

    while True:
        final_rad = read_joint_radians(driver, factor)
        if final_rad is None:
            print("[safety][WARN] Failed to read joint state for reach check.")
            return False

        errors = [abs(final_rad[i] - target[i]) for i in range(6)]
        max_err = max(errors)
        print(f"[safety] max_err(rad): {max_err:.5f}")

        if max_err < pos_tol_rad:
            print("[safety] Start pose reached (within tolerance).")
            return True

        if (time.time() - t0) > max_wait_s:
            print("[safety] Timeout waiting for start pose.")
            return False

        time.sleep(0.02)
