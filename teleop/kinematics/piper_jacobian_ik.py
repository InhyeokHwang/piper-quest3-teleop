#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

사용 방법 개요
--------------
1) Piper용 Forward Kinematics 클래스를 하나 구현해서 넘겨줘야 한다.

   fk 객체는 최소한 다음 메서드를 가져야 한다:

   - compute_fk(joint_values: list[float] | np.ndarray) -> np.ndarray (4x4)
   - compute_single_transform(i: int, joint_val: float) -> np.ndarray (4x4)
   - get_dh_params() -> np.ndarray 혹은 list[list[float]]
       * C++의 fk_.getDHParams()와 동일한 구조를 가정
       * 각 row: [alpha, a, d, theta_offset] 등 (d==0이면 revolute joint로 취급)

2) PiperJacobianIK 에 fk 객체를 넣어서 생성:

   fk = PiperForwardKinematicsPython(...)
   ik = PiperJacobianIK(fk)

3) 목표 pose (4x4 homogeneous) 와 초기 joint guess 를 가지고 IK 계산:

   target_pose = np.eye(4, dtype=float)
   initial_guess = np.zeros(6, dtype=float)
   q_sol = ik.compute_ik(initial_guess, target_pose, verbose=True)

"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple, Optional
from teleop.kinematics.piper_forward_kinematics import PiperForwardKinematics

class PiperJacobianIK:
    def __init__(
        self,
        fk: PiperForwardKinematics,
        max_iterations: int = 100,
        position_tolerance: float = 1e-5,
        orientation_tolerance: float = 1e-3,
        damping_factor: float = 0.1,
        use_analytical_jacobian: bool = False,
    ):
        """
        Parameters
        ----------
        fk : object
            Piper forward kinematics 객체.
            다음 메서드를 제공해야 함:
                - compute_fk(joint_values) -> (4,4) np.ndarray
                - compute_single_transform(i, joint_val) -> (4,4) np.ndarray
                - get_dh_params() -> array-like, shape (6, 4) or similar
        """
        self.fk = fk
        self.max_iterations = max_iterations
        self.position_tolerance = position_tolerance
        self.orientation_tolerance = orientation_tolerance
        self.damping_factor = damping_factor
        self.use_analytical_jacobian = use_analytical_jacobian

        # 기본 joint limit: [-pi, pi]
        self.joint_limits: List[Tuple[float, float]] = [
            (-np.pi, np.pi) for _ in range(6)
        ]

        # 마지막 IK 성공 여부
        self.success: bool = False

    # -------------------------
    # 설정 메서드들
    # -------------------------
    def set_max_iterations(self, max_iter: int) -> None:
        self.max_iterations = max_iter

    def set_position_tolerance(self, tol: float) -> None:
        self.position_tolerance = tol

    def set_orientation_tolerance(self, tol: float) -> None:
        self.orientation_tolerance = tol

    def set_damping_factor(self, lam: float) -> None:
        self.damping_factor = lam

    def use_analytical(self, use: bool) -> None:
        self.use_analytical_jacobian = use

    def set_joint_limits(self, limits: List[Tuple[float, float]]) -> None:
        """
        limits : 길이 6 리스트, 각 요소는 (min, max) 라디안
        """
        if len(limits) != 6:
            raise ValueError("Joint limits must be specified for all 6 joints.")
        self.joint_limits = limits

    def set_tolerance(self, tol: float) -> None:
        """
        C++ 코드의 setTolerance 와 동일한 의미.
        위치/자세 tolerance를 함께 설정.
        """
        self.set_position_tolerance(tol)
        # Orientation은 보통 더 느슨하게
        self.set_orientation_tolerance(tol * 10.0)

    # -------------------------
    # 메인 IK 함수
    # -------------------------
    def compute_ik(
        self,
        initial_guess: np.ndarray,
        target_pose: np.ndarray,
        verbose: bool = False,
        return_final_error: bool = False,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        DLS (Damped Least Squares) Jacobian IK.

        Parameters
        ----------
        initial_guess : (6,) array-like
            초기 조인트 각 (rad).
        target_pose : (4,4) array-like
            원하는 EE pose (homogeneous transform).
        verbose : bool
            반복마다 error norm 출력 여부.
        return_final_error : bool
            True이면 (solution, final_error) 를 반환.

        Returns
        -------
        joint_values : (6,) np.ndarray
        final_error  : (6,) np.ndarray or None
        """
        initial_guess = np.asarray(initial_guess, dtype=float).reshape(-1) # 형태 정리하기, float로 통일하고 1차원 벡터로 펴기
        if initial_guess.size < 6:
            raise ValueError("Initial guess must have at least 6 joint values.")
        joint_values = initial_guess.copy() # 맨 처음에는 내가 세팅한 all 0 값일 것임. 이후에는 0.01초 간격으로 읽어옴(teleop_real_arm의 루프가 0.01초)

        # 원하는 EE pose (vr controller로부터 수신)
        target_pose = np.asarray(target_pose, dtype=float)
        if target_pose.shape != (4, 4):
            raise ValueError("target_pose must be a 4x4 homogeneous matrix.")

        local_success = False
        error = None

        ### 디폴트가 100회
        for it in range(self.max_iterations):
            current_pose = self.fk.compute_fk(joint_values) # fk로 현재 EE 위치 구하기
            error = self._compute_pose_error(current_pose, target_pose) # vr controller로부터 넘어온 EE위치(target_pose)와 현재 piper의 조인트 정보(current_pose) 에러 구하기
            
            ### 오리엔테이션 제거 용도 (테스트용) ###
            # error[3:] = 0.0  ### 이거 쓰면 아주 조금 개선이 되지만 그리퍼 방향이 엉망이라 안 쓰는게 좋음
            ##################################
            
            pos_err = np.linalg.norm(error[:3]) # 포지션 차이
            ori_err = np.linalg.norm(error[3:]) # 오리엔테이션 차이

            if verbose:
                print(
                    f"[IK] iter={it}, error_norm={np.linalg.norm(error):.6e}, "
                    f"(pos={pos_err:.3e}, ori={ori_err:.3e})"
                )

            # 수렴 체크
            if pos_err < self.position_tolerance and ori_err < self.orientation_tolerance:
                local_success = True
                break

            # Jacobian 계산 (디폴트가 numerical jacobian)
            if self.use_analytical_jacobian:
                J = self._compute_analytical_jacobian(joint_values, current_pose)
            else:
                J = self._compute_numerical_jacobian(joint_values)

            # Damped Least Squares 계산
            Jt = J.T                    # (6,6)
            JJt = J @ Jt                # (6,6)
            # JJt + lambda^2 I
            lam2 = self.damping_factor ** 2
            JJt_damped = JJt + lam2 * np.eye(JJt.shape[0], dtype=float)

            # Δθ = J^T (JJ^T + λ^2 I)^(-1) e
            delta_theta = Jt @ np.linalg.solve(JJt_damped, error)

            # 조인트 업데이트 + joint limit clamp
            for i in range(6):
                new_val = joint_values[i] + delta_theta[i]
                lo, hi = self.joint_limits[i]
                joint_values[i] = float(np.clip(new_val, lo, hi))

            # [-π, π] 정규화
            self._normalize_joint_angles(joint_values)

        self.success = local_success

        # IK 계산 실패
        if not local_success:
            # 마지막 포즈/오차 다시 계산
            final_pose = self.fk.compute_fk(joint_values)
            final_error = self._compute_pose_error(final_pose, target_pose)
            final_pos_err = np.linalg.norm(final_error[:3])
            final_ori_err = np.linalg.norm(final_error[3:])

            if verbose:
                print("============================================")
                print("[IK] DID NOT CONVERGE")
                print(f"[IK] final pos_err = {final_pos_err:.6e}")
                print(f"[IK] final ori_err = {final_ori_err:.6e}")
                print(f"[IK] final error vec = {final_error}")
                print("[IK] target pose:")
                print(target_pose)
                print("[IK] final  pose:")
                print(final_pose)
                print("============================================")

            raise RuntimeError("IK did not converge within maximum iterations.")

        # 최종 에러 계산 (옵션)
        final_error = None
        if return_final_error:
            current_pose = self.fk.compute_fk(joint_values)
            final_error = self._compute_pose_error(current_pose, target_pose)

        return joint_values, final_error

    # -------------------------
    # Solution 검증용 함수
    # -------------------------
    def verify_solution(
        self, joint_values: np.ndarray, target_pose: np.ndarray
    ) -> np.ndarray:
        """
        주어진 joint_values가 target_pose를 얼마나 만족하는지 6D error 반환.
        """
        joint_values = np.asarray(joint_values, dtype=float).reshape(-1)
        target_pose = np.asarray(target_pose, dtype=float)
        current_pose = self.fk.compute_fk(joint_values)
        return self._compute_pose_error(current_pose, target_pose)

    # -------------------------
    # 내부 유틸 함수들
    # -------------------------
    def _normalize_joint_angles(self, joints: np.ndarray) -> None:
        """
        각도를 [-π, π] 범위로 정규화.
        """
        for i in range(joints.size):
            angle = joints[i]
            # fmod 기반 정규화: (angle + pi) mod 2pi - pi
            joints[i] = (angle + np.pi) % (2.0 * np.pi) - np.pi

    def _compute_pose_error(
        self, current: np.ndarray, target: np.ndarray
    ) -> np.ndarray:
        """
        6D pose error: [position_error (3), orientation_error (3)]
        Orientation은 axis-angle 기반 error.
        """
        current = np.asarray(current, dtype=float)
        target = np.asarray(target, dtype=float)

        error = np.zeros(6, dtype=float)

        # 위치 오차
        p_cur = current[:3, 3]
        p_tar = target[:3, 3]
        error[:3] = p_tar - p_cur

        # 회전 오차: R_e = R_target * R_current^T
        R_cur = current[:3, :3]
        R_tar = target[:3, :3]
        R_e = R_tar @ R_cur.T

        axis, angle = self._rotation_matrix_to_axis_angle(R_e)
        error[3:] = axis * angle

        return error

    @staticmethod
    def _rotation_matrix_to_axis_angle(R: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        3x3 회전행렬을 axis-angle 로 변환.
        axis: (3,) 단위벡터
        angle: float (rad)
        """
        # trace 기반으로 angle 계산
        eps = 1e-9
        trace = np.trace(R)
        # 수치오차 보정
        cos_angle = (trace - 1.0) / 2.0
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)

        if abs(angle) < eps:
            # 거의 회전이 없는 경우: axis는 의미 없지만, (1,0,0) 같은 기본축 사용
            return np.array([1.0, 0.0, 0.0], dtype=float), 0.0

        # 일반적인 경우
        # axis = 1 / (2 sinθ) * [R32 - R23, R13 - R31, R21 - R12]
        denom = 2.0 * np.sin(angle)
        if abs(denom) < eps:
            # 거의 π 회전인 경우 등 특이 케이스 간단 처리
            # 대략적인 축만 복원
            axis = np.empty(3, dtype=float)
            axis[0] = np.sqrt(max(R[0, 0] - R[1, 1] - R[2, 2] + 1.0, 0.0)) / 2.0
            axis[1] = np.sqrt(max(-R[0, 0] + R[1, 1] - R[2, 2] + 1.0, 0.0)) / 2.0
            axis[2] = np.sqrt(max(-R[0, 0] - R[1, 1] + R[2, 2] + 1.0, 0.0)) / 2.0
            # 방향 부호는 행렬의 비대칭 부분으로 보정 가능하지만, 여기서는 생략
            norm = np.linalg.norm(axis)
            if norm < eps:
                axis = np.array([1.0, 0.0, 0.0], dtype=float)
            else:
                axis /= norm
            return axis, angle

        axis = np.array(
            [
                (R[2, 1] - R[1, 2]) / denom,
                (R[0, 2] - R[2, 0]) / denom,
                (R[1, 0] - R[0, 1]) / denom,
            ],
            dtype=float,
        )

        # 정규화
        axis_norm = np.linalg.norm(axis)
        if axis_norm < eps:
            axis = np.array([1.0, 0.0, 0.0], dtype=float)
        else:
            axis /= axis_norm

        return axis, angle

    def _compute_numerical_jacobian(self, joint_values: np.ndarray) -> np.ndarray:
        """
        수치 미분 기반 Jacobian 계산.
        """
        delta = 1e-6
        joint_values = joint_values.copy()

        J = np.zeros((6, 6), dtype=float)

        T0 = self.fk.compute_fk(joint_values)
        p0 = T0[:3, 3]
        R0 = T0[:3, :3]

        for i in range(6):
            perturbed = joint_values.copy()
            perturbed[i] += delta

            Ti = self.fk.compute_fk(perturbed)
            pi = Ti[:3, 3]
            Ri = Ti[:3, :3]

            # 위치 부분 (linear velocity)
            J[:3, i] = (pi - p0) / delta

            # 회전 부분 (angular velocity): dR = Ri * R0^T
            dR = Ri @ R0.T
            axis, angle = self._rotation_matrix_to_axis_angle(dR)
            J[3:, i] = axis * angle / delta

        return J

    def _compute_analytical_jacobian(
        self, joint_values: np.ndarray, current_pose: np.ndarray
    ) -> np.ndarray:
        """
        해석 Jacobian. PiperForwardKinematics 의 convention 에 따라 달라질 수 있음.
        """
        joint_values = np.asarray(joint_values, dtype=float).reshape(-1)
        J = np.zeros((6, 6), dtype=float)

        T = np.eye(4, dtype=float)
        z_axes = []
        p_origins = []

        # DH 파라미터에서 d 값이 0이면 revolute joint로 간주한다.
        dh_params = np.asarray(self.fk.get_dh_params(), dtype=float)

        for i in range(6):
            # compute_single_transform(i, joint_values[i]) 가
            # 각 조인트 i 의 변환을 반환해야 한다.
            T = T @ self.fk.compute_single_transform(i, joint_values[i])
            z_axes.append(T[:3, 2].copy())   # frame i 의 z축
            p_origins.append(T[:3, 3].copy())  # frame i 의 origin

        p_end = p_origins[-1]

        for i in range(6):
            z = z_axes[i]
            p_i = p_origins[i]

            # revolute vs prismatic 구분
            d_i = dh_params[i][2] if dh_params.ndim >= 2 else 0.0

            if np.isclose(d_i, 0.0):
                # Revolute joint
                J[:3, i] = np.cross(z, (p_end - p_i))
                J[3:, i] = z
            else:
                # Prismatic joint (Piper에는 없지만 형식상 포함)
                J[:3, i] = z
                J[3:, i] = 0.0

        return J
