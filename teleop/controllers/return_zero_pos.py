# teleop/controllers/return_zero_pos.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence
import numpy as np


@dataclass
class ReturnZeroPosConfig:
    """
    OpenTeleVision teleoperator.right_state index mapping:
      4: A Button
      5: B Button
    """
    idx_return_pressed: int = 4  # default: A Button
    threshold: float = 0.5


@dataclass
class ReturnZeroPosOutput:
    holding: bool
    just_pressed: bool
    go_to_zero: bool = False
    target_T: Optional[object] = None


class ReturnZeroPosController:
    def __init__(self, config: ReturnZeroPosConfig | None = None):
        self.cfg = config or ReturnZeroPosConfig()
        self._last_pressed: float = 0.0
        self._holding: bool = False

    @staticmethod
    def _get_float(seq: Sequence[float], idx: int, default: float) -> float:
        try:
            return float(seq[idx])
        except Exception:
            return default

    def reset(self) -> None:
        self._last_pressed = 0.0
        self._holding = False

    def update(self, teleoperator) -> ReturnZeroPosOutput:
        rs = getattr(teleoperator, "right_state", None)
        pressed = self._get_float(rs, self.cfg.idx_return_pressed, 0.0) if rs is not None else 0.0

        holding = pressed > self.cfg.threshold
        just_pressed = holding and (self._last_pressed <= self.cfg.threshold)
        self._last_pressed = pressed

        return ReturnZeroPosOutput(
            holding=holding,
            just_pressed=just_pressed,
            go_to_zero=just_pressed,  # edge trigger
        )