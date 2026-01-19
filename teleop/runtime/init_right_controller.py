from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Any

def _clamp(x: float, lo: float, hi: float) -> float:
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x

def _get_float(seq: Sequence[float], idx: int, default: float = 0.0) -> float:
    try:
        return float(seq[idx])
    except Exception:
        return default


@dataclass
class RightControllerConfig:
    idx_trigger_pressed: int = 0
    idx_squeeze_pressed: int = 1
    idx_return_pressed: int = 4
    idx_trigger_value: int = 6

    hold_to_teleop: bool = True
    use_relative_controller_delta: bool = True
    threshold: float = 0.5

    # gripper output
    gripper_out_min: int = 0
    gripper_out_max: int = 1000
    gripper_close_when_high: bool = True

    gripper_deadzone_low: float = 0.05
    gripper_deadzone_high: float = 0.95
    gripper_alpha: float = 0.35 # EMA alpha
    gripper_mode: str = "analog" # "analog" or "toggle"
    toggle_open_vg: float = 0.0
    toggle_closed_vg: float = 1.0

@dataclass
class RightControllerState:
    # squeeze
    holding: bool
    just_pressed: bool
    just_released: bool

    # return-to-zero
    go_to_zero: bool
    return_holding: bool
    return_just_pressed: bool

    # gripper
    grip_out: int
    grip_vg: float  # 0..1 

    # optional pose mapping (relative delta)
    target_T: Optional[Any] = None

class RightController:

    def __init__(self, config: RightControllerConfig | None = None):
        self.cfg = config or RightControllerConfig()

        # --- squeeze state ---
        self._last_squeeze: float = 0.0

        # captured references for relative delta (optional)
        self._neutral_target_T = None
        self._ref_controller_T = None

        # --- return-to-zero state ---
        self._last_return: float = 0.0

        # --- gripper state ---
        self._vg_smoothed: float = 0.0
        self._toggle_closed: bool = False
        self._last_trigger_pressed: float = 0.0

    # ---------- optional pose helpers ----------
    @staticmethod
    def _to_T44(x):
        """
        Convert controller pose to (4,4) np.ndarray.
        Accepts:
          - np.ndarray shape (4,4)
          - flat list/tuple/np.ndarray length 16
        """
        import numpy as np

        if x is None:
            return None
        arr = np.asarray(x, dtype=float)
        if arr.shape == (4, 4):
            return arr
        if arr.ndim == 1 and arr.size == 16:
            return arr.reshape(4, 4)
        raise ValueError(f"T must be (4,4) or flat len16, got shape {arr.shape}")

    @staticmethod
    def _apply_relative_delta(*, neutral_target_T, ref_controller_T, controller_T):
        import numpy as np
        delta = np.linalg.inv(ref_controller_T) @ controller_T
        return neutral_target_T @ delta

    # ---------- lifecycle ----------
    def reset(self, *, open_gripper: bool = True) -> None:
        self._last_squeeze = 0.0
        self._neutral_target_T = None
        self._ref_controller_T = None

        self._last_return = 0.0

        self._vg_smoothed = 0.0 if open_gripper else 1.0
        self._toggle_closed = (not open_gripper)
        self._last_trigger_pressed = 0.0

    # ---------- main ----------
    def update(
        self,
        teleoperator,
        *,
        controller_T: Optional[Any] = None,
        neutral_target_T: Optional[Any] = None,
        enable_relative_target: bool = False,
    ) -> RightControllerState:
        """
        controller_T: current right controller pose (4x4) if you want relative target
        neutral_target_T: 4x4 baseline target pose
        enable_relative_target:
            If True and cfg.use_relative_controller_delta is True:
              - on squeeze just_pressed: capture references
              - while holding: compute target_T by applying relative delta
        """
        rs = getattr(teleoperator, "right_state", None) or []

        # -----------------
        # squeeze edge/hold
        # -----------------
        squeeze = _get_float(rs, self.cfg.idx_squeeze_pressed, 0.0)
        holding = squeeze > self.cfg.threshold
        just_pressed = holding and (self._last_squeeze <= self.cfg.threshold)
        just_released = (not holding) and (self._last_squeeze > self.cfg.threshold)
        self._last_squeeze = squeeze

        # -----------------
        # return-to-zero (A) edge
        # -----------------
        ret = _get_float(rs, self.cfg.idx_return_pressed, 0.0)
        return_holding = ret > self.cfg.threshold
        return_just_pressed = return_holding and (self._last_return <= self.cfg.threshold)
        self._last_return = ret
        go_to_zero = return_just_pressed  # edge trigger

        # -----------------
        # gripper (trigger)
        # -----------------
        vg_raw = self._read_gripper_vg(rs)          # 0..1
        vg = self._post_process_gripper(vg_raw)     # clamp + deadzone + smoothing
        grip_out = self._vg_to_output(vg)

        # -----------------
        # optional relative target
        # -----------------
        target_T = None
        if enable_relative_target and self.cfg.use_relative_controller_delta:
            cT = self._to_T44(controller_T) if controller_T is not None else None
            nT = self._to_T44(neutral_target_T) if neutral_target_T is not None else None

            if just_pressed and cT is not None and nT is not None:
                # capture references when squeeze is pressed
                self._ref_controller_T = cT.copy()
                self._neutral_target_T = nT.copy()

            if holding and (self._ref_controller_T is not None) and (self._neutral_target_T is not None) and (cT is not None):
                target_T = self._apply_relative_delta(
                    neutral_target_T=self._neutral_target_T,
                    ref_controller_T=self._ref_controller_T,
                    controller_T=cT,
                )

        return RightControllerState(
            holding=holding,
            just_pressed=just_pressed,
            just_released=just_released,
            go_to_zero=go_to_zero,
            return_holding=return_holding,
            return_just_pressed=return_just_pressed,
            grip_out=grip_out,
            grip_vg=vg,
            target_T=target_T,
        )

    # ---------- gripper internals ----------
    def _read_gripper_vg(self, rs: Sequence[float]) -> float:
        mode = (self.cfg.gripper_mode or "analog").lower().strip()

        if mode == "toggle":
            pressed = _get_float(rs, self.cfg.idx_trigger_pressed, 0.0)
            # rising edge
            if pressed > self.cfg.threshold and self._last_trigger_pressed <= self.cfg.threshold:
                self._toggle_closed = not self._toggle_closed
            self._last_trigger_pressed = pressed
            return self.cfg.toggle_closed_vg if self._toggle_closed else self.cfg.toggle_open_vg

        # default: analog
        v = _get_float(rs, self.cfg.idx_trigger_value, 0.0)
        return _clamp(v, 0.0, 1.0)

    def _post_process_gripper(self, vg: float) -> float:
        vg = _clamp(vg, 0.0, 1.0)

        if vg < self.cfg.gripper_deadzone_low:
            vg = 0.0
        if vg > self.cfg.gripper_deadzone_high:
            vg = 1.0

        a = _clamp(self.cfg.gripper_alpha, 0.0, 1.0)
        self._vg_smoothed = (1.0 - a) * self._vg_smoothed + a * vg
        return _clamp(self._vg_smoothed, 0.0, 1.0)

    def _vg_to_output(self, vg: float) -> int:
        if not self.cfg.gripper_close_when_high:
            vg = 1.0 - vg

        out = self.cfg.gripper_out_min + vg * (self.cfg.gripper_out_max - self.cfg.gripper_out_min)
        out_i = int(round(out))

        lo = min(self.cfg.gripper_out_min, self.cfg.gripper_out_max)
        hi = max(self.cfg.gripper_out_min, self.cfg.gripper_out_max)
        if out_i < lo:
            out_i = lo
        if out_i > hi:
            out_i = hi
        return out_i