# piper/driver.py

from typing import Sequence, Optional, Union
import time

from piper_sdk import C_PiperInterface_V2


class PiperDriver:
    """
    Low-level Piper hardware driver.
    This class is the ONLY place that directly talks to Piper SDK / CAN.
    """

    def __init__(self, can_port: str = "can0"):
        self.can_port = can_port
        self._piper: Optional[C_PiperInterface_V2] = None
        self.connected = False

    # =============================
    # Connection / lifecycle
    # =============================
    def connect(self):
        if self.connected:
            return

        self._piper = C_PiperInterface_V2(self.can_port)
        self._piper.ConnectPort()
        self.connected = True

    def enable(self):
        if not self.connected:
            raise RuntimeError("Piper is not connected.")

        self._piper.EnableArm(7)

    def disable(self):
        if not self.connected:
            return

        try:
            self._piper.DisableArm(7)
        except Exception as e:
            print("[PiperDriver] DisableArm failed:", e)

    # =============================
    # Motion control
    # =============================
    def set_motion_mode(
        self,
        ctrl_mode: int = 0x01,
        move_mode: int = 0x04,
        speed: int = 50,
        is_mit_mode: int = 0xAD,
        residence_time: int = 0,
        installation_pos: int = 0x00,
    ):
        if not self.connected:
            raise RuntimeError("Piper is not connected.")

        self._piper.MotionCtrl_2(
            ctrl_mode=ctrl_mode,
            move_mode=move_mode,
            move_spd_rate_ctrl=speed,
            is_mit_mode=is_mit_mode,
            residence_time=residence_time,
            installation_pos=installation_pos,
        )

    def send_joints(self, joint_values: Sequence[int]):
        """
        joint_values: iterable of 6 integers (Piper internal unit)
        """
        if not self.connected:
            raise RuntimeError("Piper is not connected.")

        if len(joint_values) != 6:
            raise ValueError(f"Expected 6 joint values, got {len(joint_values)}")

        self._piper.JointCtrl(*joint_values)

    def send_joints_mit(
        self,
        q_ref: Sequence[float],      # rad
        dq_ref: Sequence[float],     # rad/s (일단 0으로 둬도 됨)
        kp: float = 3.0,
        kd: float = 2.0,
        tau_ref: Union[Sequence[float], float] = 0.0,
    ):
        if not self.connected:
            raise RuntimeError("Piper is not connected.")
        if len(q_ref) != 6 or len(dq_ref) != 6:
            raise ValueError("q_ref and dq_ref must be length 6")

        if isinstance(tau_ref, (int, float)):
            tau_ref = [float(tau_ref)] * 6
        if len(tau_ref) != 6:
            raise ValueError("tau_ref must be length 6 or scalar")

        for i in range(6):
            motor_num = i + 1
            self._piper.JointMitCtrl(
                motor_num=motor_num,
                pos_ref=float(q_ref[i]),
                vel_ref=float(dq_ref[i]),
                kp=float(kp),
                kd=float(kd),
                t_ref=float(tau_ref[i]),
            )

    def set_gripper(
        self,
        position: int,
        effort: int = 2000,   # 0~5000
        enable: bool = True,
        clear_error: bool = False,
    ):
        if not self.connected:
            raise RuntimeError("Piper is not connected.")

        if clear_error:
            code = 0x03 if enable else 0x02  # enable/disable + clear error
        else:
            code = 0x01 if enable else 0x00  # enable/disable

        pos = abs(int(position))
        eff = int(max(0, min(5000, effort)))

        self._piper.GripperCtrl(pos, eff, code, 0x00)

    # =============================
    # State feedback
    # =============================
    def get_joint_positions_raw(self):
        """
        Returns raw joint state message from SDK.
        Used only by safety / calibration code.
        """
        if not self.connected:
            raise RuntimeError("Piper is not connected.")

        return self._piper.GetArmJointMsgs()

    def get_low_speed_info(self):
        if not self.connected:
            raise RuntimeError("Piper is not connected.")

        return self._piper.GetArmLowSpdInfoMsgs()

    # =============================
    # Utility
    # =============================
    def is_enabled(self) -> bool:
        """
        Check motor enable status via low speed info.
        """
        if not self.connected:
            return False

        info = self.get_low_speed_info()
        try:
            return (
                info.motor_1.foc_status.driver_enable_status
                and info.motor_2.foc_status.driver_enable_status
                and info.motor_3.foc_status.driver_enable_status
                and info.motor_4.foc_status.driver_enable_status
                and info.motor_5.foc_status.driver_enable_status
                and info.motor_6.foc_status.driver_enable_status
            )
        except AttributeError:
            # SDK 구조 변경 대비
            return False
