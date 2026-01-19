from dataclasses import dataclass
from ghost_manager_interfaces.srv import EnsureMode

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

@dataclass
class Vision60Command:
    v: float # m/s
    w: float # rad/s

class Vision60Bridge(Node):
    def __init__(self):
        super().__init__("vision60_teleop_bridge")
        self.pub_twist = self.create_publisher(Twist, "/mcu/command/manual_twist", 10)

        self.ensure_client = self.create_client(EnsureMode, "/ensure_mode")
        while not self.ensure_client.wait_for_service(timeout_sec=0.1):
            self.get_logger().info("Waiting for /ensure_mode...")

        self.current_action = 0

    def publish_twist(self, v: float, w: float):
        msg = Twist()
        msg.linear.x = float(v)
        msg.linear.y = 0.0
        msg.linear.z = 0.0
        msg.angular.x = 0.0
        msg.angular.y = 0.0
        msg.angular.z = float(w)
        self.pub_twist.publish(msg)

    def set_action(self, action: int):
        req = EnsureMode.Request()
        req.field = "action"
        req.valdes = action

        future = self.ensure_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=0.5)
        
        if future.done():
            self.current_action = action
            self.get_logger().info(f"[Vision60] action set to {action}")
        else:
            self.get_logger().warn("[Vision60] service call timeout!")

def init_ros2_bridge(vision60: bool):
    if vision60:
        rclpy.init(args=None)
        node = Vision60Bridge()
        return rclpy, node
    else:
        return None, None