#!/usr/bin/env python3
import math
import numpy as np
import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry, Path
from ackermann_msgs.msg import AckermannDriveStamped

class LongitudinalController(Node):
    def __init__(self):
        super().__init__("longitudinal_controller")

        # ---- Parameters ----
        # PID Gains
        self.declare_parameter("kp", 1.0)
        self.declare_parameter("ki", 0.1)
        self.declare_parameter("kd", 0.05)
        
        # Vehicle Model Parameters for Feedforward
        # Model: acc = k * tau + A * (v^2)
        # Therefore: tau = (acc_desired - A*v^2) / k
        self.declare_parameter("model_k", 5.0)   # Gain (Engine power/mass factor)
        self.declare_parameter("model_A", -0.01) # Drag coeff (Negative if drag opposes motion)

        # Velocity Profiling
        self.declare_parameter("max_velocity", 10.0)    # m/s (Straight line max)
        self.declare_parameter("max_lat_accel", 5.0)    # m/s^2 (Tire grip limit)
        self.declare_parameter("lookahead_dist", 3.0)   # Meters ahead to check curvature

        # Topics
        self.declare_parameter("topics.odom", "/slam/odom")
        self.declare_parameter("topics.path", "/planner/path")
        self.declare_parameter("topics.cmd", "/cmd_vel_out") # Adjust for EUFS (e.g. /eufs/set_vel)

        # Get params
        self.kp = self.get_parameter("kp").value
        self.ki = self.get_parameter("ki").value
        self.kd = self.get_parameter("kd").value
        
        self.k_model = self.get_parameter("model_k").value
        self.A_model = self.get_parameter("model_A").value
        
        self.v_max = self.get_parameter("max_velocity").value
        self.lat_acc_max = self.get_parameter("max_lat_accel").value
        self.lookahead = self.get_parameter("lookahead_dist").value

        # State
        self.current_v = 0.0
        self.current_path = [] # List of (x, y)
        self.integral_error = 0.0
        self.prev_error = 0.0
        self.last_time = self.get_clock().now()

        # Subscribers
        odom_topic = self.get_parameter("topics.odom").value
        path_topic = self.get_parameter("topics.path").value
        
        self.create_subscription(Odometry, odom_topic, self.cb_odom, 10)
        self.create_subscription(Path, path_topic, self.cb_path, 10)

        # Publisher (Command)
        self.cmd_pub = self.create_publisher(
            AckermannDriveStamped, 
            self.get_parameter("topics.cmd").value, 
            10
        )

        # Control Loop Timer (e.g., 50Hz)
        self.create_timer(0.02, self.control_loop)

        self.get_logger().info("Longitudinal Controller Started")

    def cb_odom(self, msg):
        # Calculate scalar velocity from twist
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        self.current_v = math.hypot(vx, vy)

    def cb_path(self, msg):
        # Convert path msg to list of tuples for easier processing
        self.current_path = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]

    def get_curvature(self, p1, p2, p3):
        """
        Calculates Menger curvature (1/R) of three points.
        Formula: 4 * Area / (|p1p2| * |p2p3| * |p3p1|)
        """
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3

        # Area of triangle
        area = 0.5 * abs(x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2))
        
        # Side lengths
        d12 = math.hypot(x2-x1, y2-y1)
        d23 = math.hypot(x3-x2, y3-y2)
        d31 = math.hypot(x1-x3, y1-y3)

        if d12 * d23 * d31 == 0:
            return 0.0

        return (4.0 * area) / (d12 * d23 * d31)

    def calculate_target_velocity(self):
        """
        Scans upcoming path points within lookahead distance
        and calculates max cornering speed.
        """
        if len(self.current_path) < 3:
            return 0.0  # Stop if no path

        target_v = self.v_max
        
        # We assume the path starts near the car (index 0)
        # Look ahead a few points
        accumulated_dist = 0.0
        
        for i in range(len(self.current_path) - 2):
            p1 = self.current_path[i]
            p2 = self.current_path[i+1]
            p3 = self.current_path[i+2]

            # Integrate distance
            step_dist = math.hypot(p2[0]-p1[0], p2[1]-p1[1])
            accumulated_dist += step_dist

            if accumulated_dist > self.lookahead:
                break

            kappa = self.get_curvature(p1, p2, p3)
            
            # Max cornering speed: v = sqrt( a_lat_max / curvature )
            if kappa > 1e-3: # Avoid division by zero
                v_corner = math.sqrt(self.lat_acc_max / kappa)
                if v_corner < target_v:
                    target_v = v_corner

        return target_v

    def control_loop(self):
        # 1. Calculate Target Velocity (Profile)
        v_target = self.calculate_target_velocity()

        # Safety: If path is empty or very short, stop
        if len(self.current_path) < 2:
            v_target = 0.0

        # 2. PID Calculation
        # Error
        error = v_target - self.current_v
        
        # Time delta
        now = self.get_clock().now()
        dt = (now - self.last_time).nanoseconds / 1e9
        if dt == 0: return # First run skip
        self.last_time = now

        # P, I, D terms
        p_term = self.kp * error
        self.integral_error += error * dt
        # Clamp integral to prevent windup (optional but recommended)
        self.integral_error = max(min(self.integral_error, 5.0), -5.0)
        i_term = self.ki * self.integral_error
        
        d_term = self.kd * ((error - self.prev_error) / dt)
        self.prev_error = error

        # PID Output = Desired Acceleration
        acc_desired = p_term + i_term + d_term

        # 3. Feedforward / Model Inversion
        # User Plant: acc = k * tau + A * (v^2)
        # We need to find tau.
        # tau = (acc_desired - A * v^2) / k
        
        # Note: A_model is usually negative for drag (e.g., -0.001)
        # If your identified A is positive, ensure the math reflects your sign convention.
        
        numerator = acc_desired - (self.A_model * (self.current_v ** 2))
        tau_cmd = numerator / self.k_model

        # 4. Saturate Command (0 to 1 for throttle, -1 to 0 for brake if needed)
        # Assuming tau is normalized throttle [0, 1]
        if tau_cmd > 1.0: tau_cmd = 1.0
        if tau_cmd < -1.0: tau_cmd = -1.0 
        
        # 5. Publish
        msg = AckermannDriveStamped()
        msg.header.stamp = now.to_msg()
        
        # NOTE: Check if EUFS sim wants Acceleration or Drive/Throttle
        # If it takes raw throttle in 'acceleration' or 'jerk' fields:
        msg.drive.acceleration = float(tau_cmd) 
        msg.drive.speed = float(v_target) # Just for debug/reference
        
        self.cmd_pub.publish(msg)

        # Debug logs
        # self.get_logger().info(f"V_targ: {v_target:.2f} | V_act: {self.current_v:.2f} | Acc_des: {acc_desired:.2f} | Tau: {tau_cmd:.2f}")

def main(args=None):
    rclpy.init(args=args)
    node = LongitudinalController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()