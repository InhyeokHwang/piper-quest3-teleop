# teleop/controllers/squeeze_reference.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence


@dataclass
class SqueezeReferenceConfig:
    # your right_state mapping: squeeze pressed is index 1
    idx_squeeze_pressed: int = 1

    # If True: only update target while squeeze is held (hold-to-teleop style)
    hold_to_teleop: bool = True

    # If True: while holding, map controller delta onto target pose
    # neutral_target_T' = neutral_target_T @ inv(ref_controller_T) @ controller_T
    use_relative_controller_delta: bool = True


@dataclass
class SqueezeOutput:
    holding: bool
    just_pressed: bool
    just_released: bool
    go_to_zero: bool = False
    target_T: Optional[object] = None


class SqueezeReferenceController:
    def __init__(self, config: SqueezeReferenceConfig | None = None):
        self.cfg = config or SqueezeReferenceConfig()
        self._last_pressed: float = 0.0
        self._holding: bool = False

        # captured references
        self._neutral_target_T = None
        self._ref_controller_T = None

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

        # unknown format
        raise ValueError(f"controller_T must be (4,4) or flat len16, got shape {arr.shape}")

    def reset(self) -> None:
        self._last_pressed = 0.0
        self._holding = False
        self._neutral_target_T = None
        self._ref_controller_T = None

    def update(self, teleoperator) -> SqueezeOutput:
        rs = getattr(teleoperator, "right_state", None)
        pressed = self._get_float(rs, self.cfg.idx_squeeze_pressed, 0.0) if rs is not None else 0.0

        just_pressed  = pressed > 0.5 and self._last_pressed <= 0.5
        just_released = pressed <= 0.5 and self._last_pressed > 0.5
        self._last_pressed = pressed

        holding = pressed > 0.5

        return SqueezeOutput(
            holding=holding,
            just_pressed=just_pressed,
            just_released=just_released,
        )
    
    @staticmethod
    def _get_float(seq: Sequence[float], idx: int, default: float) -> float:
        try:
            return float(seq[idx])
        except Exception:
            return default

    @staticmethod
    def _apply_relative_delta(*, neutral_target_T, ref_controller_T, controller_T):
        import numpy as np
        delta = np.linalg.inv(ref_controller_T) @ controller_T
        return neutral_target_T @ delta
