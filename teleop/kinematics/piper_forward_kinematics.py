#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
piper_forward_kinematics.py

C++ PiperForwardKinematics (Agilex-College/piper_kinematics)를
파이썬/NumPy 버전으로 옮긴 코드.

- STANDARD / MODIFIED 두 가지 DH 타입 지원
- compute_fk(joint_values): 6 DoF Piper arm의 EE pose (4x4) 리턴
- compute_single_transform(i, joint_val): i번째 조인트에 대한 4x4 변환
- get_dh_params(): DH 파라미터 테이블 리턴

"""

from __future__ import annotations

import numpy as np
from enum import Enum
from typing import List


class DHType(str, Enum):
    STANDARD = "STANDARD"
    MODIFIED = "MODIFIED"


class PiperForwardKinematics:
    def __init__(self, dh_type: DHType | str = DHType.STANDARD):
        """
        Parameters
        ----------
        dh_type : DHType or str
            'STANDARD' 또는 'MODIFIED'
        """
        if isinstance(dh_type, str):
            dh_type = DHType(dh_type.upper())
        self.dh_type: DHType = dh_type

        # DH 파라미터 테이블: 각 row = [alpha, a, d, theta_offset]
        self.dh_params: np.ndarray = np.zeros((6, 4), dtype=float)
        self._setup_dh_parameters()

    # ------------------------------------------------------------------
    # Public API (IK 쪽에서 사용할 것들)
    # ------------------------------------------------------------------
    def compute_fk(self, joint_values: List[float] | np.ndarray) -> np.ndarray:
        """
        6개 조인트 각(joint_values) 에 대해 EE pose (4x4 homogeneous) 계산.

        Parameters
        ----------
        joint_values : 길이 >=6 의 array-like (rad)

        Returns
        -------
        T : (4,4) np.ndarray
        """
        joint_values = np.asarray(joint_values, dtype=float).reshape(-1)
        if joint_values.size < 6:
            raise ValueError("Piper arm requires at least 6 joint values for FK.")

        T = np.eye(4, dtype=float)

        for i in range(6):
            alpha, a, d, theta_offset = self.dh_params[i]
            theta = joint_values[i] + theta_offset
            T = T @ self._compute_transform(alpha, a, d, theta)

        return T
    
    # ------------------------------------------------------------------
    # 추가: base + 각 조인트 프레임까지 모두 반환
    # ------------------------------------------------------------------
    def compute_fk_all(self, joint_values: List[float] | np.ndarray) -> List[np.ndarray]:
        """
        6개 조인트 각(joint_values)에 대해
        base 포함 각 단계의 pose (4x4 homogeneous)들을 반환.

        Returns
        -------
        Ts : List[np.ndarray]
            길이 7 리스트.
            Ts[0] = base frame (I)
            Ts[1] = joint1까지 누적변환
            ...
            Ts[6] = joint6(=EE)까지 누적변환 (compute_fk와 동일)
        """
        joint_values = np.asarray(joint_values, dtype=float).reshape(-1)
        if joint_values.size < 6:
            raise ValueError("Piper arm requires at least 6 joint values for FK.")

        T = np.eye(4, dtype=float)
        Ts: List[np.ndarray] = [T.copy()]  # base

        for i in range(6):
            alpha, a, d, theta_offset = self.dh_params[i]
            theta = joint_values[i] + theta_offset
            T = T @ self._compute_transform(alpha, a, d, theta)
            Ts.append(T.copy())

        return Ts
        
    def fk_all_joint_positions(self, joint_values: List[float] | np.ndarray) -> np.ndarray:
        """
        base 포함 각 조인트 프레임 원점들의 위치를 (N,3)으로 반환.

        Returns
        -------
        joints_xyz : (7,3) np.ndarray
            joints_xyz[0] = base (0,0,0)
            joints_xyz[1] = joint1 frame origin
            ...
            joints_xyz[6] = joint6(=EE) frame origin
        """
        Ts = self.compute_fk_all(joint_values)          # List[7] of (4,4)
        joints_xyz = np.stack([T[:3, 3] for T in Ts], axis=0)
        return joints_xyz

    def compute_single_transform(self, joint_index: int, joint_value: float) -> np.ndarray:
        """
        특정 조인트 하나에 대한 4x4 변환.
        """
        if not (0 <= joint_index < self.dh_params.shape[0]):
            raise ValueError("Joint index out of range.")

        alpha, a, d, theta_offset = self.dh_params[joint_index]
        theta = joint_value + theta_offset
        return self._compute_transform(alpha, a, d, theta)

    def get_dh_params(self) -> np.ndarray:
        return self.dh_params.copy()

    def get_dh_type(self) -> DHType:
        return self.dh_type


    # DH 파라미터로 Transformation Matrix를 만들 때, 기존에는 순서가 다음과 같음 Ti−1,i​ = Rotz​(θi​)Transz​(di​)Transx​(ai​)Rotx​(αi​)
    # Modified는 이 행렬 곱셈의 순서를 바꾼 것임. Rot_x(α), Trans_x(a), Rot_z(θ), Trans_z(d)
    # 두 가지가 정의되는 이유는 프레임을 관절 뒤에 두는 경우 Standard이고 프레임을 관절 앞에 두는 경우 Modified이기 떄문.
    def _setup_dh_parameters(self) -> None:
        """
        STANDARD:
            // [alpha, a, d, theta_offset]
            {-M_PI/2,   0,          0.123,      0},                     // Joint 1
            {0,         0.28503,    0,          -172.22/180*M_PI},      // Joint 2 
            {M_PI/2,    -0.021984,  0,          -102.78/180*M_PI},      // Joint 3
            {-M_PI/2,   0,          0.25075,    0},                     // Joint 4
            {M_PI/2,    0,          0,          0},                     // Joint 5
            {0,         0,          0.211,      0}                      // Joint 6

        MODIFIED:
            // [alpha, a, d, theta_offset]
            {0,         0,          0.123,      0},                     // Joint 1
            {-M_PI/2,   0,          0,          -172.22/180*M_PI},      // Joint 2 
            {0,         0.28503,    0,          -102.78/180*M_PI},      // Joint 3
            {M_PI/2,    -0.021984,  0.25075,    0},                     // Joint 4
            {-M_PI/2,   0,          0,          0},                     // Joint 5
            {M_PI/2,    0,          0.211,      0}                      // Joint 6
        """
        if self.dh_type == DHType.STANDARD:
            self.dh_params = np.array(
                [
                    [-np.pi / 2.0, 0.0,       0.123,    0.0],  # joint 1의 DH 파라미터
                    [0.0,          0.28503,   0.0,      np.deg2rad(-172.22)], # joint 2의 DH 파라미터
                    [np.pi / 2.0, -0.021984,  0.0,      np.deg2rad(-102.78)], # joint 3의 DH 파라미터
                    [-np.pi / 2.0, 0.0,       0.25075,  0.0], # joint 4의 DH 파라미터
                    [np.pi / 2.0,  0.0,       0.0,      0.0], # joint 5의 DH 파라미터
                    [0.0,          0.0,       0.211,    0.0], # joint 6의 DH 파라미터
                ],
                dtype=float,
            )
        else:
            # MODIFIED
            self.dh_params = np.array(
                [
                    [0.0,          0.0,       0.123,    0.0], # joint 1의 DH 파라미터
                    [-np.pi / 2.0, 0.0,       0.0,      np.deg2rad(-172.22)], # joint 2의 DH 파라미터
                    [0.0,          0.28503,   0.0,      np.deg2rad(-102.78)], # joint 3의 DH 파라미터
                    [np.pi / 2.0, -0.021984,  0.25075,  0.0], # joint 4의 DH 파라미터
                    [-np.pi / 2.0, 0.0,       0.0,      0.0], # joint 5의 DH 파라미터
                    [np.pi / 2.0,  0.0,       0.211,    0.0], # joint 6의 DH 파라미터
                ],
                dtype=float,
            )

    def _compute_transform(self, alpha: float, a: float, d: float, theta: float) -> np.ndarray:
        """
        STANDARD / MODIFIED 두 가지 케이스.

        Returns
        -------
        T : (4,4) np.ndarray
        """
        ca = np.cos(alpha)
        sa = np.sin(alpha)
        ct = np.cos(theta)
        st = np.sin(theta)

        T = np.zeros((4, 4), dtype=float)

        if self.dh_type == DHType.STANDARD:
            # Standard DH
            # T << cos(theta), -sin(theta)*cos(alpha),  sin(theta)*sin(alpha), a*cos(theta),
            #      sin(theta),  cos(theta)*cos(alpha), -cos(theta)*sin(alpha), a*sin(theta),
            #      0,           sin(alpha),             cos(alpha),            d,
            #      0,           0,                      0,                     1;
            T[0, 0] = ct
            T[0, 1] = -st * ca
            T[0, 2] =  st * sa
            T[0, 3] =  a * ct

            T[1, 0] = st
            T[1, 1] =  ct * ca
            T[1, 2] = -ct * sa
            T[1, 3] =  a * st

            T[2, 0] = 0.0
            T[2, 1] = sa
            T[2, 2] = ca
            T[2, 3] = d

            T[3, 0] = 0.0
            T[3, 1] = 0.0
            T[3, 2] = 0.0
            T[3, 3] = 1.0

        else:
            # Modified DH
            # T << cos(theta),            -sin(theta),            0,             a,
            #      sin(theta)*cos(alpha), cos(theta)*cos(alpha), -sin(alpha), -sin(alpha)*d,
            #      sin(theta)*sin(alpha), cos(theta)*sin(alpha),  cos(alpha),  cos(alpha)*d,
            #      0,                     0,                      0,             1;
            T[0, 0] = ct
            T[0, 1] = -st
            T[0, 2] = 0.0
            T[0, 3] = a

            T[1, 0] = st * ca
            T[1, 1] = ct * ca
            T[1, 2] = -sa
            T[1, 3] = -sa * d

            T[2, 0] = st * sa
            T[2, 1] = ct * sa
            T[2, 2] = ca
            T[2, 3] = ca * d

            T[3, 0] = 0.0
            T[3, 1] = 0.0
            T[3, 2] = 0.0
            T[3, 3] = 1.0

        return T
