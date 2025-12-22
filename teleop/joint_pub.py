#!/usr/bin/env python3
import socket
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

JOINT_NAMES = [
    "joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7", "joint8"
]  

UDP_IP = "127.0.0.1"
UDP_PORT = 15000

class JointStateUdpPublisher(Node):
    def __init__(self):
        super().__init__("joint_state_udp_pub")
        self.pub = self.create_publisher(JointState, "/joint_states", 10)

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((UDP_IP, UDP_PORT))
        self.sock.setblocking(False)

        self.timer = self.create_timer(0.01, self.tick)  # 100Hz

    def tick(self):
        try:
            data, _ = self.sock.recvfrom(4096)
        except BlockingIOError:
            return

        try:
            q = np.fromstring(data.decode("utf-8"), sep=" ").astype(float)
            if q.size < 6:
                return

            arm = q[:6]
            if q.size >= 8:
                gripper = q[6:8]
            else:
                gripper = np.array([0.0, 0.0])  # fallback
        except Exception:
            return

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = JOINT_NAMES
        msg.position = list(arm) + list(gripper)
        self.pub.publish(msg)

def main():
    rclpy.init()
    node = JointStateUdpPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
