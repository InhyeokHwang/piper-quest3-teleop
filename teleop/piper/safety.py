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
    if not driver.connected:
        raise RuntimeError("PiperDriver must be connected before enable_and_wait().")

    start = time.time()
    tries = 0
    last_exc = None

    while True:
        tries += 1

        # enable 명령이 실제로 성공하는지/예외나는지 로그
        try:
            ret = driver.enable()
            print(f"[safety] enable() try={tries} ret={ret}")
            last_exc = None
        except Exception as e:
            last_exc = e
            print(f"[safety] enable() try={tries} exception={repr(e)}")

        # 그리퍼는 실패해도 계속
        if also_open_gripper:
            try:
                driver.set_gripper(position=0, effort=2000, enable=True)
            except Exception as e:
                print("[safety] Gripper open command failed:", repr(e))

        # is_enabled도 예외/원시값 로깅
        try:
            enabled = driver.is_enabled()
            print(f"[safety] Enable status: {enabled}")
        except Exception as e:
            print("[safety] is_enabled() exception:", repr(e))
            enabled = False

        if enabled:
            return True

        if (time.time() - start) > timeout_s:
            msg = f"[safety] Enable timeout after {timeout_s:.1f}s."
            if last_exc is not None:
                msg += f" last enable() exception={repr(last_exc)}"
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
        driver.set_motion_mode(ctrl_mode=0x01, move_mode=0x01, speed=motion_speed)
    except Exception as e:
        print("[safety] MotionCtrl_2 failed (continuing):", e)

    current = read_joint_radians(driver, factor)
    if current is None:
        print("[safety][ERROR] Cannot read joint state; abort move_to_start_pose for safety.")
        return False

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
