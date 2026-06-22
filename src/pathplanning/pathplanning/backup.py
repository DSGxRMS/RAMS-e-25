#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped
import pandas as pd
import numpy as np
import math
import os
import sys

# We will implement necessary math inline to ensure this script is self-contained
# and meets the specific sequential requirement without external dependency behavior.

class PathPublisher(Node):
    def __init__(self):
        super().__init__('pp_publisher')

        # Parameters
        self.declare_parameter('path_file', 'pathpoints_shifted.csv')
        self.declare_parameter('num_points', 7) # Increased default for better perception viz
        self.declare_parameter('interval_m', 0.5) 
        self.declare_parameter('publish_rate', 1.0) # Faster rate for smooth updates
        self.declare_parameter('loop', False)

        path_file = self.get_parameter('path_file').get_parameter_value().string_value
        self.num_points = self.get_parameter('num_points').get_parameter_value().integer_value
        self.interval_m = self.get_parameter('interval_m').get_parameter_value().double_value
        publish_rate = self.get_parameter('publish_rate').get_parameter_value().double_value
        self.is_loop = self.get_parameter('loop').get_parameter_value().bool_value

        # Resolve path file path
        if not os.path.isabs(path_file):
            base_dir = os.path.dirname(os.path.abspath(__file__))
            path_file = os.path.join(base_dir, path_file)

        self.get_logger().info(f"Loading path from {path_file}")
        
        # Load and preprocess path
        try:
            df = pd.read_csv(path_file)
            self.global_rx, self.global_ry = df["x"].to_numpy(), df["y"].to_numpy()
            self.get_logger().info(f"Path loaded: {len(self.global_rx)} points")
        except Exception as e:
            self.get_logger().error(f"Failed to load path: {e}")
            self.global_rx = []
            self.global_ry = []

        # State
        self.cx = 0.0
        self.cy = 0.0
        self.yaw = 0.0
        self.have_odom = False
        
        # INDEX TRACKING
        self.cur_idx = 0 
        self.path_initialized = False

        # ROS Interfaces
        self.create_subscription(Odometry, '/ground_truth/odom', self.odom_cb, 10)
        self.path_pub = self.create_publisher(Path, '/path_points', 10)
        self.timer = self.create_timer(1.0/publish_rate, self.timer_cb)

        # Visualization
        self.visualize = True
        if self.visualize:
            try:
                import matplotlib.pyplot as plt
                self.plt = plt
                self.plt.ion()
                self.fig, self.ax = self.plt.subplots()
                self.ax.set_title("Perception Simulation (Local Frame)")
                self.ax.set_xlabel("x (m)")
                self.ax.set_ylabel("y (m)")
                self.ax.grid(True)
                self.ln, = self.ax.plot([], [], 'ro', label='Planned Path')
                # Car fixed at origin looking up X
                self.ax.plot(0, 0, 'k^', markersize=10, label='Ego Car')
                self.ax.set_xlim(-5, 5)
                self.ax.set_ylim(-5, 5)
                self.ax.legend()
            except ImportError:
                self.get_logger().warning("Matplotlib not found, visualization disabled")
                self.visualize = False

    def odom_cb(self, msg):
        self.cx = msg.pose.pose.position.x
        self.cy = msg.pose.pose.position.y
        
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.yaw = math.atan2(siny_cosp, cosy_cosp)
        
        self.have_odom = True

    def get_dist_sq(self, idx):
        """Helper to get squared distance from car to a path index"""
        dx = self.cx - self.global_rx[idx]
        dy = self.cy - self.global_ry[idx]
        return dx**2 + dy**2

    def timer_cb(self):
        if not self.have_odom or len(self.global_rx) == 0:
            return

        # --- INITIALIZATION ---
        # If this is the first run, we do a one-time global search to find 
        # where the car started.
        if not self.path_initialized:
            min_dist = float('inf')
            best_idx = 0
            # Search entire array once
            for i in range(len(self.global_rx)):
                d = self.get_dist_sq(i)
                if d < min_dist:
                    min_dist = d
                    best_idx = i
            self.cur_idx = best_idx
            self.path_initialized = True

        # --- SEQUENTIAL TRACKING LOGIC ---
        # Instead of searching the whole array (which causes jumps), 
        # we check if the NEXT point is closer than the CURRENT point.
        # We loop this a few times (search_window) to allow the index to "catch up" 
        # if the car is moving fast, but we never look backwards.
        
        search_window = 50 # How many points ahead can we skip in one frame?
        
        for _ in range(search_window):
            current_dist = self.get_dist_sq(self.cur_idx)
            
            # Calculate next index
            next_idx = self.cur_idx + 1
            
            # Handle Loop / End of Track
            if self.is_loop:
                next_idx = next_idx % len(self.global_rx)
            elif next_idx >= len(self.global_rx):
                # End of track reached, stay at last point
                break 

            next_dist = self.get_dist_sq(next_idx)

            # If the next point is closer than the current one, advance!
            # This effectively "slides" the window forward relative to car motion.
            if next_dist < current_dist:
                self.cur_idx = next_idx
            else:
                # We are at the local minimum (closest point), stop searching.
                break

        # --- PUBLISH PATH ---
        path_msg = Path()
        path_msg.header.frame_id = 'map' 
        path_msg.header.stamp = self.get_clock().now().to_msg()

        local_xs = []
        local_ys = []
        
        cos_yaw = math.cos(self.yaw)
        sin_yaw = math.sin(self.yaw)

        # Extract strictly sequential points starting from our updated cur_idx
        for i in range(self.num_points):
            idx = self.cur_idx + i
            
            if self.is_loop:
                idx = idx % len(self.global_rx)
            else:
                if idx >= len(self.global_rx):
                    break # Don't publish past the end of the line
            
            gx = self.global_rx[idx]
            gy = self.global_ry[idx]

            # Transform to local frame (Simulating Perception)
            # (Local X is forward, Local Y is Left)
            dx = gx - self.cx
            dy = gy - self.cy
            
            # Rotation matrix for Global to Local
            lx = cos_yaw * dx + sin_yaw * dy
            ly = -sin_yaw * dx + cos_yaw * dy
            
            local_xs.append(lx)
            local_ys.append(ly)
            
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = gx
            pose.pose.position.y = gy
            pose.pose.position.z = 0.0
            
            # Simple orientation calculation (point towards next point)
            # We calculate heading based on i and i+1
            idx_next = idx + 1
            if self.is_loop: idx_next %= len(self.global_rx)
            
            if idx_next < len(self.global_rx):
                nx = self.global_rx[idx_next]
                ny = self.global_ry[idx_next]
                heading = math.atan2(ny - gy, nx - gx)
            else:
                heading = 0.0 # End of path default

            # Convert yaw to quaternion
            cy = math.cos(heading * 0.5)
            sy = math.sin(heading * 0.5)
            pose.pose.orientation.w = cy
            pose.pose.orientation.z = sy
            
            path_msg.poses.append(pose)

        self.path_pub.publish(path_msg)

        if self.visualize:
            self.ln.set_data(local_xs, local_ys)
            self.plt.pause(0.001)

def main(args=None):
    rclpy.init(args=args)
    node = PathPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()