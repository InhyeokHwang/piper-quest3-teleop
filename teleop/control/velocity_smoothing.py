# velocity_smoothing.py
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class VelocitySmoothingParams:
    ema_beta: float = 0.30        # 0..1 (bigger = less smoothing)
    accel_max: float = 25.0       # rad/s^2 (or m/s^2 for prismatic dof)
    vel_max: float | None = None  # optional extra clamp in rad/s; None = no clamp


def clamp(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.minimum(np.maximum(x, lo), hi)


def smooth_velocity(
    vel_raw: np.ndarray,
    vel_prev_filt: np.ndarray | None,
    dt: float,
    params: VelocitySmoothingParams = VelocitySmoothingParams(),
) -> np.ndarray:
    """
    Smooth IK output velocity in nv-space.
    Steps:
      1) EMA low-pass
      2) accel clamp (limits frame-to-frame change)
      3) optional vel clamp
    """
    v = np.asarray(vel_raw, dtype=float)

    if vel_prev_filt is None:
        v_f = v.copy()
    else:
        v_prev = np.asarray(vel_prev_filt, dtype=float)
        beta = float(np.clip(params.ema_beta, 0.0, 1.0))
        v_f = (1.0 - beta) * v_prev + beta * v

        # accel clamp
        dt_safe = max(float(dt), 1e-6)
        dv = v_f - v_prev
        dv_lim = params.accel_max * dt_safe
        v_f = v_prev + clamp(dv, -dv_lim, dv_lim)

    # optional vel clamp
    if params.vel_max is not None:
        vmax = float(params.vel_max)
        v_f = clamp(v_f, -vmax, vmax)

    return v_f
