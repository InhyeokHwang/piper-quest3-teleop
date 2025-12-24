# teleop/controllers/gripper.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple


def _clamp(x: float, lo: float, hi: float) -> float:
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


@dataclass
class GripperConfig:
    """
    Gripper control config.

    output:
      - value in [out_min, out_max] (typically 0..1000 for Piper)
    input:
      - vg in [0,1] from:
          teleoperator.right_state    

    modes:
      - "toggle": A button toggles open/close
      - "analog": triggerValue directly controls open..close
      - "toggle_then_analog": A toggles enable_analog, then trigger controls (optional)
    """
    # output scale (robot API units)
    # value in [out_min, out_max] (typically 0..1000 for Piper)
    out_min: int = 0 
    out_max: int = 1000

    # mapping direction
    # If True: vg=0 -> open(out_min), vg=1 -> close(out_max)
    # If False: inverted
    close_when_high: bool = True

    # deadzone in vg (0..1) -> deadzone은 아날로그 입력에서 너무 작은 변화는 무시해버리는 것
    deadzone_low: float = 0.05 # vg < 0.05면 0.0으로 강제
    deadzone_high: float = 0.95 # vg > 0.95면 1.0으로 강제

    # smoothing
    # alpha=1.0 => no smoothing, immediate
    # alpha small => smoother, slower
    alpha: float = 0.35

    # control mode: "toggle" | "analog" | "toggle_then_analog"
    mode: str = "toggle"

    # right_state indices (expected convention from our earlier mapping)
    idx_a_button: int = 4         # aButton pressed (0/1)
    idx_trigger_value: int = 6    # triggerValue (0..1)
    idx_squeeze_value: int = 7    # squeezeValue (0..1) (optional alternative)

    # which analog channel to use in fallback (trigger or squeeze)
    analog_source: str = "trigger"  # "trigger" or "squeeze"

    # in toggle mode, what vg means for "closed"
    toggle_closed_vg: float = 1.0
    toggle_open_vg: float = 0.0


class GripperController:
    """
    Stateful gripper controller.

    Usage:
        gripper = GripperController(GripperConfig(mode="toggle"))
        pos = gripper.update(teleoperator)   # returns int in [out_min, out_max]
        driver.set_gripper(position=pos, speed=..., enable=1)

    Input priority:
      teleoperator.right_state   (Sequence of floats)
         - in toggle mode uses aButton edge
         - in analog mode uses triggerValue/squeezeValue
    """

    def __init__(self, config: GripperConfig | None = None):
        self.cfg = config or GripperConfig()

        # internal state
        self._vg_smoothed: float = 0.0
        self._last_a_pressed: float = 0.0
        self._toggle_closed: bool = False
        self._analog_enabled: bool = False  # used in toggle_then_analog

    def reset(self, *, open_gripper: bool = True) -> None:
        self._last_a_pressed = 0.0
        self._toggle_closed = False
        self._analog_enabled = False
        self._vg_smoothed = 0.0 if open_gripper else 1.0

    # ---------- public ----------

    def update(self, teleoperator) -> int:
        """
        Update internal state from teleoperator and return gripper position in robot units.
        """
        vg = self._read_vg(teleoperator)  # 0..1 vg는 virtual gripper value의 약자
        vg = self._post_process(vg)       # deadzone + clamp + smoothing
        return self._vg_to_output(vg)

    # ---------- internal ----------

    def _read_vg(self, teleoperator) -> float:
        if hasattr(teleoperator, "right_state"):
            try:
                rs = getattr(teleoperator, "right_state")
                return self._vg_from_right_state(rs)
            except Exception:
                return 0.0

        return 0.0

    def _vg_from_right_state(self, rs: Sequence[float]) -> float:
        # defensive: convert to list of floats without numpy dependency
        def get_idx(i: int, default: float = 0.0) -> float:
            try:
                return float(rs[i])
            except Exception:
                return default

        a_pressed = get_idx(self.cfg.idx_a_button, 0.0)

        mode = self.cfg.mode.lower().strip()
        if mode == "toggle":
            # rising edge on A
            if a_pressed > 0.5 and self._last_a_pressed <= 0.5:
                self._toggle_closed = not self._toggle_closed
            self._last_a_pressed = a_pressed
            return self.cfg.toggle_closed_vg if self._toggle_closed else self.cfg.toggle_open_vg

        if mode == "analog":
            return self._analog_value(rs)

        if mode == "toggle_then_analog":
            # A toggles whether analog is active, otherwise open/close toggled state
            if a_pressed > 0.5 and self._last_a_pressed <= 0.5:
                self._analog_enabled = not self._analog_enabled
            self._last_a_pressed = a_pressed

            if self._analog_enabled:
                return self._analog_value(rs)

            # not analog: behave like simple toggle open/close using last toggle state
            # (you can change this behavior if you want)
            if a_pressed > 0.5 and self._last_a_pressed <= 0.5:
                self._toggle_closed = not self._toggle_closed
            return self.cfg.toggle_closed_vg if self._toggle_closed else self.cfg.toggle_open_vg

        # unknown mode -> safe open
        return 0.0

    def _analog_value(self, rs: Sequence[float]) -> float:
        if self.cfg.analog_source == "squeeze":
            v = self._get_float(rs, self.cfg.idx_squeeze_value, 0.0)
        else:
            v = self._get_float(rs, self.cfg.idx_trigger_value, 0.0)
        return _clamp(v, 0.0, 1.0)

    @staticmethod
    def _get_float(seq: Sequence[float], idx: int, default: float) -> float:
        try:
            return float(seq[idx])
        except Exception:
            return default

    def _post_process(self, vg: float) -> float:
        # clamp first
        vg = _clamp(vg, 0.0, 1.0)

        # deadzone snap
        if vg < self.cfg.deadzone_low:
            vg = 0.0
        if vg > self.cfg.deadzone_high:
            vg = 1.0

        # smoothing
        a = _clamp(self.cfg.alpha, 0.0, 1.0)
        self._vg_smoothed = (1.0 - a) * self._vg_smoothed + a * vg
        return _clamp(self._vg_smoothed, 0.0, 1.0)

    def _vg_to_output(self, vg: float) -> int:
        # apply direction
        if not self.cfg.close_when_high:
            vg = 1.0 - vg

        out = self.cfg.out_min + vg * (self.cfg.out_max - self.cfg.out_min)
        # round and clamp to int
        out_i = int(round(out))
        if out_i < min(self.cfg.out_min, self.cfg.out_max):
            out_i = min(self.cfg.out_min, self.cfg.out_max)
        if out_i > max(self.cfg.out_min, self.cfg.out_max):
            out_i = max(self.cfg.out_min, self.cfg.out_max)
        return out_i
