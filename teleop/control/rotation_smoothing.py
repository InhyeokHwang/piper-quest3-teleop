# rotation_smoothing.py
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class PoseSmoothingParams:
    alpha_pos: float = 0.20              # position EMA
    alpha_rot: float = 0.10              # rotation SLERP (usually smaller than pos)
    pos_deadband: float = 0.002          # meters
    rot_deadband_rad: float = np.deg2rad(1.0)  # radians (e.g. 1 degree)


def rotmat_to_quat(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> quaternion (w, x, y, z)."""
    R = np.asarray(R, dtype=float)
    t = np.trace(R)

    if t > 0.0:
        s = np.sqrt(t + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    else:
        if (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s

    q = np.array([w, x, y, z], dtype=float)
    q /= (np.linalg.norm(q) + 1e-12)
    return q


def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    """Quaternion (w, x, y, z) -> rotation matrix."""
    q = np.asarray(q, dtype=float)
    q = q / (np.linalg.norm(q) + 1e-12)
    w, x, y, z = q

    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz),       2.0 * (xz + wy)],
            [2.0 * (xy + wz),       1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy),       2.0 * (yz + wx),       1.0 - 2.0 * (xx + yy)],
        ],
        dtype=float,
    )


def slerp_quat(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation between quaternions (w,x,y,z)."""
    q0 = np.asarray(q0, dtype=float)
    q1 = np.asarray(q1, dtype=float)
    q0 = q0 / (np.linalg.norm(q0) + 1e-12)
    q1 = q1 / (np.linalg.norm(q1) + 1e-12)

    dot = float(np.dot(q0, q1))
    # shortest path
    if dot < 0.0:
        q1 = -q1
        dot = -dot

    dot = float(np.clip(dot, -1.0, 1.0))

    # very close: lerp
    if dot > 0.9995:
        q = q0 + t * (q1 - q0)
        q /= (np.linalg.norm(q) + 1e-12)
        return q

    theta0 = float(np.arccos(dot))
    sin_theta0 = float(np.sin(theta0))
    theta = theta0 * float(t)

    s0 = float(np.sin(theta0 - theta)) / (sin_theta0 + 1e-12)
    s1 = float(np.sin(theta)) / (sin_theta0 + 1e-12)

    q = s0 * q0 + s1 * q1
    q /= (np.linalg.norm(q) + 1e-12)
    return q


def relative_rotation_angle(R_prev: np.ndarray, R_new: np.ndarray) -> float:
    """Angle (rad) of relative rotation R_prev^T R_new."""
    R_rel = R_prev.T @ R_new
    c = (np.trace(R_rel) - 1.0) * 0.5
    c = float(np.clip(c, -1.0, 1.0))
    return float(np.arccos(c))


def smooth_target_T(
    T_filt: np.ndarray,
    target_T: np.ndarray,
    params: PoseSmoothingParams = PoseSmoothingParams(),
) -> np.ndarray:
    """
    Smooth 4x4 target pose:
    - position: EMA + deadband
    - rotation: SLERP + deadband
    Returns updated T_filt (also updates in-place if you pass same array).
    """
    if T_filt is None:
        return np.array(target_T, dtype=float, copy=True)

    T_filt = np.asarray(T_filt, dtype=float)
    target_T = np.asarray(target_T, dtype=float)

    # --- position ---
    p = target_T[:3, 3]
    p_prev = T_filt[:3, 3]
    dp = p - p_prev
    if float(np.linalg.norm(dp)) >= params.pos_deadband:
        T_filt[:3, 3] = (1.0 - params.alpha_pos) * p_prev + params.alpha_pos * p

    # --- rotation ---
    R_new = target_T[:3, :3]
    R_prev = T_filt[:3, :3]
    dtheta = relative_rotation_angle(R_prev, R_new)

    if dtheta >= params.rot_deadband_rad:
        q_prev = rotmat_to_quat(R_prev)
        q_new = rotmat_to_quat(R_new)
        q_filt = slerp_quat(q_prev, q_new, params.alpha_rot)
        T_filt[:3, :3] = quat_to_rotmat(q_filt)

    # keep homogeneous bottom row/col safe
    T_filt[3, :] = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
    return T_filt
