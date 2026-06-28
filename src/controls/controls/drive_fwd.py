#!/usr/bin/env python3
"""Drive the car straight forward at a fixed acceleration (no YAML quoting pain).
Usage: python3 drive_fwd.py            # accel 2.0, steer 0
       python3 drive_fwd.py 3.0 0.0    # accel 3.0, steer 0.0
"""
import sys
import rclpy
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDriveStamped

accel = float(sys.argv[1]) if len(sys.argv) > 1 else 2.0
steer = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0


class Drv(Node):
    def __init__(self):
        super().__init__("drive_fwd")
        self.pub = self.create_publisher(AckermannDriveStamped, "/cmd", 10)
        self.create_timer(0.05, self.tick)   # 20 Hz
        print(f"driving: accel={accel} steer={steer}  (Ctrl+C to stop)", flush=True)

    def tick(self):
        m = AckermannDriveStamped()
        m.drive.acceleration = accel
        m.drive.steering_angle = steer
        self.pub.publish(m)


rclpy.init()
rclpy.spin(Drv())
