from dataclasses import dataclass
import time

@dataclass
class LeftControllerConfig:
    idx_trigger_pressed: int = 0
    idx_squeeze_pressed: int = 1

    idx_x_button: int = 4
    idx_y_button: int = 5

    idx_trigger_value: int = 6
    idx_squeeze_value: int = 7

    idx_thumbstick_x: int = 10   # +right
    idx_thumbstick_y: int = 11   # +down 

    deadband: float = 0.15
    max_v: float = 0.6
    max_w: float = 1.2
    pub_hz: float = 20.0

@dataclass
class LeftControllerState:
    y_was_pressed: bool = False
    last_pub_t: float = 0.0
    vision_action: int = 2

class LeftController:
    def __init__(self, cfg: LeftControllerConfig, st: LeftControllerState | None = None):
        self.cfg = cfg
        self.st = st if st is not None else LeftControllerState()

    def _deadband(self, x: float) -> float:
        return 0.0 if abs(x) < self.cfg.deadband else x

    def update(self, left_state, vision_node=None, vision_rclpy=None):
        ls = left_state
        
        if ls is None or len(ls) <= self.cfg.idx_thumbstick_y:
            return 0.0, 0.0

        # thumbstick
        lx_raw = float(ls[self.cfg.idx_thumbstick_x])
        ly_raw = float(ls[self.cfg.idx_thumbstick_y])

        # trigger / squeeze
        tr_btn = bool(ls[self.cfg.idx_trigger_pressed])
        sq_btn = bool(ls[self.cfg.idx_squeeze_pressed])
        tr_val = float(ls[self.cfg.idx_trigger_value])
        sq_val = float(ls[self.cfg.idx_squeeze_value])

        # x / y
        x_btn = bool(ls[self.cfg.idx_x_button])
        y_btn = bool(ls[self.cfg.idx_y_button])

        lx = self._deadband(lx_raw)

        fwd = tr_val if tr_val > 0.05 else (1.0 if tr_btn else 0.0)
        bwd = sq_val if sq_val > 0.05 else (1.0 if sq_btn else 0.0)

        v_axis = fwd - bwd
        w_axis = -lx

        cmd_v = self.cfg.max_v * v_axis
        cmd_w = self.cfg.max_w * w_axis

        # y pressed
        if vision_node is not None and y_btn:
            if not self.st.y_was_pressed:
                new_action = 0 if self.st.vision_action == 2 else 2
                print(f"[VISION60] Y pressed: action {self.st.vision_action} -> {new_action}")
                try:
                    vision_node.set_action(new_action)
                    self.st.vision_action = new_action
                except Exception as e:
                    print("[VISION60] set_action failed:", e)
            self.st.y_was_pressed = True
        else:
            self.st.y_was_pressed = False

        # ros2 publish
        if vision_node is not None:
            period = 1.0 / max(self.cfg.pub_hz, 1e-6)
            now = time.monotonic()

            if (now - self.st.last_pub_t) >= period:
                vision_node.publish_twist(cmd_v, cmd_w)
                if vision_rclpy is not None:
                    try:
                        vision_rclpy.spin_once(vision_node, timeout_sec=0.0)
                    except Exception:
                        pass
                self.st.last_pub_t = now
        return cmd_v, cmd_w