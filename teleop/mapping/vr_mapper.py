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
        # vr_pos = [x, y, z]라고 하면 norm = sqrt(x^2 + y^2 + z^2)
        # 즉 원점으로부터 직선거리가 threshold보다 커지면 valid한 input으로 판단
        # (vr_pos가 스트리밍 초기에 [0, 0, 0]으로 들어오기 때문)
        return np.linalg.norm(vr_pos) >= self.cfg.zero_pos_threshold
    
    def _normalize_vr_quat(self, quat_xyzw: np.ndarray) -> Optional[np.ndarray]:
        qx, qy, qz, qw = quat_xyzw
        q = np.array([qw, qx, qy, qz], dtype=float)  # wxyz 형태로 만듦

        n = np.linalg.norm(q) # sqrt(qw^2 + qx^2 + qy^2 + qz^2)
        if n < self.cfg.quat_norm_eps: # 아직 값이 안 들어온 상태
            return None
        return q / n
    
    def set_neutral_from_pose7(self, right_pose: np.ndarray) -> None:
        """
        Force-update VR neutral baselines (position + orientation) from a pose7 input.
        right_pose: [x,y,z,qx,qy,qz,qw] (xyzw)
        """
        pose = np.asarray(right_pose, dtype=float).reshape(-1)
        if pose.size != 7:
            raise ValueError(f"right_pose must be 7 elements (xyzw), got {pose.size}")

        vr_pos = pose[:3].copy()
        vr_quat_xyzw = pose[3:].copy()

        # only update if valid (avoid zero frame)
        if not self._is_vr_pos_valid(vr_pos):
            return

        q_vr_wxyz = self._normalize_vr_quat(vr_quat_xyzw)
        if q_vr_wxyz is None:
            return

        R_vr = rotations.matrix_from_quaternion(q_vr_wxyz)

        # set new neutral baselines
        self.vr_neutral_pos = vr_pos
        self.R_vr0 = R_vr
        self.prev_q_target_wxyz = None  # reset sign stabilization history
        if self.debug:
            print("[VRMapper] Neutral reset from squeeze press.")

    def clear_neutral(self) -> None:
        """Next valid frame will re-initialize neutral baselines."""
        self.vr_neutral_pos = None
        self.R_vr0 = None
        self.prev_q_target_wxyz = None

    def compute_target_T(
        self,
        right_pose: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        """
        Returns (target_T, info).

        - target_T is None when calibration is not ready or input is invalid.
        """
        pose = np.asarray(right_pose, dtype=float).reshape(-1)
        if pose.size != 7:
            raise ValueError(f"right_pose must be 7 elements (xyzw), got {pose.size}")

        vr_pos = pose[:3].copy()
        vr_quat_xyzw = pose[3:].copy()

        # validity check (Quest not yet streaming etc.)
        if not self._is_vr_pos_valid(vr_pos):
            if self.debug:
                print("[VRMapper] vr_pos ~ 0, ignoring frame.")
            return None

        # set neutral position once
        if self.vr_neutral_pos is None:
            self.vr_neutral_pos = vr_pos.copy()
            if self.debug:
                print("[VRMapper] Set vr_neutral_pos:", self.vr_neutral_pos)

        ##### position mapping: rel motion + EE_START
        rel_pos = vr_pos - self.vr_neutral_pos
        mapped_pos = self.cfg.position_scale * rel_pos + self.ee_start_pos ## EE 시작 위치 + 컨트롤러 rel pos * scale 한 xyz

        ##### orientation mapping
        q_vr_wxyz = self._normalize_vr_quat(vr_quat_xyzw)

        if q_vr_wxyz is None:
            if self.debug:
                print("[VRMapper] VR quaternion near zero, ignoring frame.")
            return None
        
        R_vr = rotations.matrix_from_quaternion(q_vr_wxyz)

        # 4-1) calibration: store R_vr0 once
        if self.R_vr0 is None:
            self.R_vr0 = R_vr.copy()
            if self.debug:
                print("[VRMapper] Calibrated orientation baseline R_vr0 set.")
            # 첫 프레임은 기준 잡는 용도라 target을 안 내보내는 게 안전
            return None

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

        if self.debug:
            print("=== [VRMapper] VR -> Robot Target ===")
            print(" vr_pos:", vr_pos)
            print(" vr_quat_xyzw:", vr_quat_xyzw)
            print(" neutral:", self.vr_neutral_pos)
            print(" rel_pos:", rel_pos)
            print(" mapped_pos:", mapped_pos)
            print("====================================")

        return target_T
