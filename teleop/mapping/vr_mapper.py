# mapping/vr_mapper.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import numpy as np
from pytransform3d import rotations

from ..kinematics.pose import pose7_to_matrix, stabilize_quaternion_sign


@dataclass
class VRMapperConfig:
    """
    VR -> Robot EE mapping parameters.
    """
    position_scale: float = 1.0
    zero_pos_threshold: float = 1e-3  # vr_pos norm < threshold => treat as invalid
    quat_norm_eps: float = 1e-8

    # VR relative rotation -> Robot relative rotation axis remap matrix
    # Same as your P matrix
    P: np.ndarray = np.array(
        [
            [0.0, 0.0, -1.0],  # X_robot <-  Z_vr
            [0.0, 1.0,  0.0],  # Y_robot <-  Y_vr
            [1.0, 0.0,  0.0],  # Z_robot <-  X_vr (note: sign depends on your convention)
        ],
        dtype=float,
    )


class VRToRobotMapper:
    """
    Maps VR controller pose (right hand) to robot end-effector target pose (4x4).

    Input
    -----
    right_pose: [x, y, z, qx, qy, qz, qw]  (xyzw)

    Output
    ------
    target_T: 4x4 homogeneous transform (robot base frame)
    """

    def __init__(
        self,
        ee_start_pos: np.ndarray,   # (3,) robot EE position at q=0 (or your chosen reference)
        ee_start_rot: np.ndarray,   # (3,3) robot EE rotation at q=0
        config: VRMapperConfig | None = None,
        debug: bool = False,
    ):
        self.ee_start_pos = np.asarray(ee_start_pos, dtype=float).reshape(3)
        self.R_ee0 = np.asarray(ee_start_rot, dtype=float).reshape(3, 3)
        self.cfg = config if config is not None else VRMapperConfig()
        self.debug = debug

        # runtime state
        self.vr_neutral_pos: Optional[np.ndarray] = None  # (3,)
        self.R_vr0: Optional[np.ndarray] = None          # (3,3)

        self.prev_q_target_wxyz: Optional[np.ndarray] = None  # quaternion sign stabilization

    def reset_calibration(self):
        """
        Reset neutral position and orientation calibration.
        """
        self.vr_neutral_pos = None
        self.R_vr0 = None
        self.prev_q_target_wxyz = None

    def _is_vr_pos_valid(self, vr_pos: np.ndarray) -> bool:
        return np.linalg.norm(vr_pos) >= self.cfg.zero_pos_threshold

    def compute_target_T(
        self,
        right_pose: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        """
        Returns (target_T, info).

        - target_T is None when calibration is not ready or input is invalid.
        - info contains debug fields useful for logging/plotting.
        """
        pose = np.asarray(right_pose, dtype=float).reshape(-1)
        if pose.size != 7:
            raise ValueError(f"right_pose must be 7 elements (xyzw), got {pose.size}")

        vr_pos = pose[:3].copy()
        vr_quat_xyzw = pose[3:].copy()

        info: Dict[str, Any] = {
            "vr_pos": vr_pos,
            "vr_quat_xyzw": vr_quat_xyzw,
            "vr_neutral_pos": None,
            "rel_pos": None,
            "mapped_pos": None,
            "calibrated": False,
            "R_vr0_set": self.R_vr0 is not None,
        }

        # 1) validity check (Quest not yet streaming etc.)
        if not self._is_vr_pos_valid(vr_pos):
            if self.debug:
                print("[VRMapper] vr_pos ~ 0, ignoring frame.")
            return None, info

        # 2) set neutral position once
        if self.vr_neutral_pos is None:
            self.vr_neutral_pos = vr_pos.copy()
            if self.debug:
                print("[VRMapper] Set vr_neutral_pos:", self.vr_neutral_pos)

        info["vr_neutral_pos"] = self.vr_neutral_pos

        # 3) position mapping: rel motion + EE_START
        rel_pos = vr_pos - self.vr_neutral_pos
        mapped_pos = self.cfg.position_scale * rel_pos + self.ee_start_pos
        info["rel_pos"] = rel_pos
        info["mapped_pos"] = mapped_pos

        # 4) orientation mapping
        qx, qy, qz, qw = vr_quat_xyzw
        q_vr_wxyz = np.array([qw, qx, qy, qz], dtype=float)

        n = np.linalg.norm(q_vr_wxyz)
        if n < self.cfg.quat_norm_eps:
            if self.debug:
                print("[VRMapper] VR quaternion near zero, ignoring frame.")
            return None, info
        q_vr_wxyz /= n

        R_vr = rotations.matrix_from_quaternion(q_vr_wxyz)

        # 4-1) calibration: store R_vr0 once
        if self.R_vr0 is None:
            self.R_vr0 = R_vr.copy()
            info["calibrated"] = False
            if self.debug:
                print("[VRMapper] Calibrated orientation baseline R_vr0 set.")
            # 첫 프레임은 기준 잡는 용도라 target을 안 내보내는 게 안전
            return None, info

        # 4-2) compute delta rotation in VR frame
        dR_vr = self.R_vr0.T @ R_vr

        # 4-3) axis remap VR -> Robot
        P = np.asarray(self.cfg.P, dtype=float).reshape(3, 3)
        dR_robot = P @ dR_vr @ P.T

        # 4-4) robot target orientation: EE0 * dR_robot
        R_target = self.R_ee0 @ dR_robot

        # quaternion sign stabilization (wxyz)
        q_target_wxyz = rotations.quaternion_from_matrix(R_target)
        q_target_wxyz = stabilize_quaternion_sign(q_target_wxyz, self.prev_q_target_wxyz)
        self.prev_q_target_wxyz = q_target_wxyz.copy()

        # convert back to xyzw for pose7_to_matrix
        mapped_quat_xyzw = np.array(
            [q_target_wxyz[1], q_target_wxyz[2], q_target_wxyz[3], q_target_wxyz[0]],
            dtype=float,
        )

        mapped_pose7 = np.concatenate([mapped_pos, mapped_quat_xyzw])
        target_T = pose7_to_matrix(mapped_pose7)

        info["calibrated"] = True

        if self.debug:
            print("=== [VRMapper] VR -> Robot Target ===")
            print(" vr_pos:", vr_pos)
            print(" vr_quat_xyzw:", vr_quat_xyzw)
            print(" neutral:", self.vr_neutral_pos)
            print(" rel_pos:", rel_pos)
            print(" mapped_pos:", mapped_pos)
            print("====================================")

        return target_T, info
