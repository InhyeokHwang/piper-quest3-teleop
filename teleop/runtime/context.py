from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Any
import numpy as np

from ..VuerTeleop import VuerTeleop
from teleop.runtime.init_left_controller import LeftController
from teleop.runtime.init_right_controller import RightController


@dataclass
class RuntimeContext:
    # core handles 
    teleoperator: VuerTeleop
    fk: Any
    mapper: Any

    model: Any
    data: Any
    configuration: Any
    tasks: list
    limits: list
    solver: Any
    rate: Any

    # optional IO / UI
    cam: Optional[Any]
    viewer: Optional[Any]

    # controllers
    left: LeftController
    right: RightController

    # gripper qpos indices in MuJoCo
    q_idx7: int
    q_idx8: int

    # constant target
    T_zero: np.ndarray

    # loop state
    last_q: np.ndarray
    T_filt: Optional[np.ndarray]
    vel_filt: Optional[np.ndarray]

    # ros2 bridge (Vision60)
    vision_rclpy: Optional[Any]
    vision_node: Optional[Any]

    cmd_shared: Optional[Any] = None   # multiprocessing.Array('d', 7, lock=True)
    stop_event: Optional[Any] = None   # multiprocessing.Event()
    sender_proc: Optional[Any] = None  # multiprocessing.Process

    startup_sent_zero: bool = False
    sent_joint_zero: bool = False
    hold_target: Optional[np.ndarray] = None
    mode: str = "RETURNING"

    _closed: bool = False

    def close(self) -> None:
        """Release resources safely (idempotent)."""
        if self._closed:
            return
        self._closed = True

        # --- Stop sender process first ---
        try:
            if self.stop_event is not None:
                self.stop_event.set()
            if self.sender_proc is not None:
                self.sender_proc.join(timeout=2.0)
                if self.sender_proc.is_alive():
                    self.sender_proc.terminate()
                    self.sender_proc.join(timeout=1.0)
        except Exception:
            pass
        self.sender_proc = None
        self.stop_event = None
        self.cmd_shared = None

        # Viewer
        v = self.viewer
        if v is not None:
            try:
                if hasattr(v, "close") and callable(v.close):
                    v.close()
            except Exception:
                pass
            self.viewer = None

        # Camera
        c = self.cam
        if c is not None:
            try:
                if hasattr(c, "close") and callable(c.close):
                    c.close()
            except Exception:
                pass
            self.cam = None

        # Teleoperator
        t = self.teleoperator
        if t is not None:
            try:
                if hasattr(t, "close") and callable(t.close):
                    t.close()
            except Exception:
                pass