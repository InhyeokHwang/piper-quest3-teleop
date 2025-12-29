# teleop/runtime/init_squeeze.py
from __future__ import annotations

from ..controllers.squeeze_reference import SqueezeReferenceController, SqueezeReferenceConfig

def init_squeeze():
    # squeeze pressed는 index 1
    cfg = SqueezeReferenceConfig(
        idx_squeeze_pressed=1,
        hold_to_teleop=True,
        use_relative_controller_delta=True,
    )
    ctl = SqueezeReferenceController(cfg)
    ctl.reset()
    return ctl
