import time
import numpy as np

from .init_driver import init_driver
from ..control.sender import piper_send_jointctrl
from .. import config

def piper_sender(args, cmd_shared, stop_event, send_hz: float):
    driver = None
    try:
        driver = init_driver(args)
        send_period = 1.0 / max(float(send_hz), 1e-6)
        next_send = time.monotonic()

        while not stop_event.is_set():
            now = time.monotonic()
            if now < next_send:
                time.sleep(min(next_send - now, 0.005))
                continue

            with cmd_shared.get_lock():
                q = np.array(cmd_shared[0:6], dtype=float)
                grip_hw = int(cmd_shared[6])

            next_send = piper_send_jointctrl(
                driver, args.dry_run, q, grip_hw,
                next_send, send_period, config.RAD_TO_PIPER
            )

    finally:
        try:
            if driver is not None:
                driver.close()
        except Exception:
            pass
