# teleop/runtime/return_zero_pos_init.py
from __future__ import annotations

from ..controllers.return_zero_pos import ReturnZeroPosController, ReturnZeroPosConfig

def init_return_zero_pos():
    cfg = ReturnZeroPosConfig(
        idx_return_pressed=4,  
        threshold=0.5,
    )
    ctl = ReturnZeroPosController(cfg)
    ctl.reset()
    return ctl
