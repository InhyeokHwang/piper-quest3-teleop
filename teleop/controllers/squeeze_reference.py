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
    go_to_zero: bool
    target_T: Optional[object]  # (4,4) numpy array if provided


class SqueezeReferenceController:
    def __init__(self, config: SqueezeReferenceConfig | None = None):
        self.cfg = config or SqueezeReferenceConfig()
        self._last_pressed: float = 0.0
        self._holding: bool = False

        # captured references
        self._neutral_target_T = None
        self._ref_controller_T = None

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

    # squeeze_reference.py (핵심만)
    def update(
        self,
        teleoperator,
        *,
        controller_T=None,
        current_target_T=None,
    ) -> SqueezeOutput:
        rs = getattr(teleoperator, "right_state", None)
        pressed = self._get_float(rs, self.cfg.idx_squeeze_pressed, 0.0) if rs is not None else 0.0

        just_pressed  = pressed > 0.5 and self._last_pressed <= 0.5
        just_released = pressed <= 0.5 and self._last_pressed > 0.5
        self._last_pressed = pressed

        # ⭐ holding은 pressed로만 결정
        self._holding = pressed > 0.5

        target_T = None

        # ⭐ go_to_zero는 falling edge로만 결정 (holding 조건 제거)
        go_to_zero = bool(just_released)

        # (선택) neutral 캡처는 press 때만
        if just_pressed:
            # 여기서 neutral을 갱신/리셋하는 트리거만 쓰는 게 가장 깔끔
            if current_target_T is not None:
                self._neutral_target_T = current_target_T
            if controller_T is not None:
                self._ref_controller_T = controller_T

        # (선택) 상대변위 쓰는 경우에만 계산
        if self._holding and self.cfg.use_relative_controller_delta and self._neutral_target_T is not None:
            if controller_T is not None and self._ref_controller_T is not None:
                target_T = self._apply_relative_delta(
                    neutral_target_T=self._neutral_target_T,
                    ref_controller_T=self._ref_controller_T,
                    controller_T=controller_T,
                )

        return SqueezeOutput(
            holding=self._holding,
            just_pressed=just_pressed,
            just_released=just_released,
            go_to_zero=go_to_zero,
            target_T=target_T,
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
