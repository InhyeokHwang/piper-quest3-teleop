# teleop/controllers/gripper.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


def _clamp(x: float, lo: float, hi: float) -> float:
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


@dataclass
class GripperConfig:
    """
    Gripper control config (trigger-only).

    Input (teleoperator.right_state):
      - trigger pressed (0/1): idx_trigger_pressed (default 0)
      - triggerValue  (0..1): idx_trigger_value   (default 6)

    Output:
      - int in [out_min, out_max] (typically 0..1000 for Piper)

    Modes:
      - "analog": triggerValue continuously controls open..close
      - "toggle": trigger pressed rising-edge toggles open/close
    """
    # output scale (robot API units)
    out_min: int = 0
    out_max: int = 1000

    # direction:
    #   True : vg=0 -> open(out_min), vg=1 -> close(out_max)
    #   False: inverted
    close_when_high: bool = True

    # deadzone in vg (0..1)
    deadzone_low: float = 0.05
    deadzone_high: float = 0.95

    # smoothing alpha (EMA): 1.0=no smoothing, smaller=more smooth
    alpha: float = 0.35

    # control mode: "analog" | "toggle"
    mode: str = "analog"

    # right_state indices (your convention)
    idx_trigger_pressed: int = 0   # trigger (0/1)
    idx_trigger_value: int = 6     # triggerValue (0..1)

    # toggle outputs in vg-space
    toggle_open_vg: float = 0.0
    toggle_closed_vg: float = 1.0


class GripperController:
    """
    Trigger-only gripper controller.

    - analog mode:
        vg = triggerValue (0..1)
    - toggle mode:
        rising edge of trigger pressed toggles open/close

    Returns:
        int position in [out_min, out_max]
    """

    def __init__(self, config: GripperConfig | None = None):
        self.cfg = config or GripperConfig()

        # internal state
        self._vg_smoothed: float = 0.0
        self._toggle_closed: bool = False
        self._last_trigger_pressed: float = 0.0

    def reset(self, *, open_gripper: bool = True) -> None:
        self._vg_smoothed = 0.0 if open_gripper else 1.0
        self._toggle_closed = False if open_gripper else True
        self._last_trigger_pressed = 0.0

    def update(self, teleoperator) -> int:
        vg = self._read_vg(teleoperator)   # 0..1
        vg = self._post_process(vg)        # clamp + deadzone + smoothing
        return self._vg_to_output(vg)

    # ---------- internal ----------

    def _read_vg(self, teleoperator) -> float:
        rs = getattr(teleoperator, "right_state", None)
        if rs is None:
            return 0.0
        try:
            return self._vg_from_right_state(rs)
        except Exception:
            return 0.0

    def _vg_from_right_state(self, rs: Sequence[float]) -> float:
        mode = (self.cfg.mode or "analog").lower().strip()
        if mode == "toggle":
            pressed = self._get_float(rs, self.cfg.idx_trigger_pressed, 0.0)
            # rising edge
            if pressed > 0.5 and self._last_trigger_pressed <= 0.5:
                self._toggle_closed = not self._toggle_closed
            self._last_trigger_pressed = pressed
            return self.cfg.toggle_closed_vg if self._toggle_closed else self.cfg.toggle_open_vg

        # default: analog
        v = self._get_float(rs, self.cfg.idx_trigger_value, 0.0)
        return _clamp(v, 0.0, 1.0)

    @staticmethod
    def _get_float(seq: Sequence[float], idx: int, default: float) -> float:
        try:
            return float(seq[idx])
        except Exception:
            return default

    def _post_process(self, vg: float) -> float:
        vg = _clamp(vg, 0.0, 1.0)

        # deadzone snap (useful for analog)
        if vg < self.cfg.deadzone_low:
            vg = 0.0
        if vg > self.cfg.deadzone_high:
            vg = 1.0

        # smoothing (EMA)
        a = _clamp(self.cfg.alpha, 0.0, 1.0)
        self._vg_smoothed = (1.0 - a) * self._vg_smoothed + a * vg
        return _clamp(self._vg_smoothed, 0.0, 1.0)

    def _vg_to_output(self, vg: float) -> int:
        if not self.cfg.close_when_high:
            vg = 1.0 - vg

        out = self.cfg.out_min + vg * (self.cfg.out_max - self.cfg.out_min)
        out_i = int(round(out))

        lo = min(self.cfg.out_min, self.cfg.out_max)
        hi = max(self.cfg.out_min, self.cfg.out_max)
        if out_i < lo:
            out_i = lo
        if out_i > hi:
            out_i = hi
        return out_i
