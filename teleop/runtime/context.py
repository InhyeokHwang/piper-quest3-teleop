# teleop/runtime/context.py
from dataclasses import dataclass
from typing import Optional
import numpy as np

@dataclass
class RuntimeContext:
    teleoperator: object
    cam: Optional[object]

    fk: object
    mapper: object

    model: object
    data: object
    configuration: object
    tasks: list
    limits: list
    solver: str
    rate: object

    viewer: Optional[object]

    gripper_ctl: object
    squeeze_ctl: object          # (추가한 거면 여기도 default 없이)

    # loop state (default 없는 것들은 여기서 끝까지 계속)
    last_q: np.ndarray
    T_filt: Optional[np.ndarray]
    vel_filt: Optional[np.ndarray]

    # indices for gripper joints
    q_idx7: int
    q_idx8: int

    # zero pose / target cache (default 없는 것부터, default 있는 건 아래로)
    T_zero: np.ndarray
    target_T: Optional[np.ndarray]  # <- 여기서도 default 주지 말거나, 아래로 내려

    # ----- default fields MUST be last -----
    driver: Optional[object] = None
    _closed: bool = False

    def close(self):
        """Release resources safely (idempotent)."""
        if self._closed:
            return
        self._closed = True

        # Viewer 먼저 (GLX 관련 에러 줄이기)
        v = getattr(self, "viewer", None)
        if v is not None:
            try:
                # 일부 viewer는 is_running()이 있음
                if hasattr(v, "is_running") and callable(v.is_running):
                    if v.is_running() and hasattr(v, "close") and callable(v.close):
                        v.close()
                elif hasattr(v, "close") and callable(v.close):
                    v.close()
            except Exception:
                pass
            self.viewer = None  # 중복 close 방지

        # Camera
        c = getattr(self, "cam", None)
        if c is not None and hasattr(c, "close") and callable(c.close):
            try:
                c.close()
            except Exception:
                pass
            self.cam = None

        # Teleoperator (여기서 tv 프로세스 + shm 정리되어야 함)
        t = getattr(self, "teleoperator", None)
        if t is not None and hasattr(t, "close") and callable(t.close):
            try:
                t.close()
            except Exception:
                pass

        # Driver (실로봇일 때)
        d = getattr(self, "driver", None)
        if d is not None:
            # driver가 disconnect/close 중 뭘 제공하는지 몰라서 둘 다 시도
            for fn in ("disconnect", "close"):
                f = getattr(d, fn, None)
                if callable(f):
                    try:
                        f()
                    except Exception:
                        pass
                    break
            self.driver = None
