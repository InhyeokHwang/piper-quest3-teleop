import numpy as np
from pytransform3d import rotations
from typing import Optional

# x, y, z, qx, qy, qz, qw 입력을 4x4 행렬로 변환
def pose7_to_matrix(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=float).reshape(-1)
    if pose.size != 7:
        raise ValueError(f"pose must have 7 elements [x,y,z,qx,qy,qz,qw], got {pose.size}")

    pos = pose[:3]
    qx, qy, qz, qw = pose[3:]

    # pytransform3d expects quaternion in [w, x, y, z]
    q_wxyz = np.array([qw, qx, qy, qz], dtype=float)

    n = np.linalg.norm(q_wxyz)
    if n < 1e-8:
        raise ValueError("Quaternion norm is too small (near zero).")
    q_wxyz /= n

    R = rotations.matrix_from_quaternion(q_wxyz)

    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3, 3] = pos
    return T

def stabilize_quaternion_sign(
    q_wxyz: np.ndarray,
    q_prev_wxyz: Optional[np.ndarray],
) -> np.ndarray:
    """
    Stabilize quaternion sign to prevent sudden flips (q <-> -q).

    Parameters
    ----------
    q_wxyz : np.ndarray
        Current quaternion [w, x, y, z]
    q_prev_wxyz : np.ndarray or None
        Previous quaternion [w, x, y, z].
        If None, no stabilization is applied.

    Returns
    -------
    np.ndarray
        Stabilized quaternion [w, x, y, z]
    """
    q = np.asarray(q_wxyz, dtype=float).reshape(4)

    n = np.linalg.norm(q)
    if n < 1e-8:
        raise ValueError("Quaternion norm is too small (near zero).")
    q /= n

    if q_prev_wxyz is None:
        return q

    q_prev = np.asarray(q_prev_wxyz, dtype=float).reshape(4)
    q_prev /= max(np.linalg.norm(q_prev), 1e-8)

    # q and -q represent the same rotation
    # choose the one closer to previous quaternion
    if np.dot(q, q_prev) < 0.0:
        q = -q

    return q