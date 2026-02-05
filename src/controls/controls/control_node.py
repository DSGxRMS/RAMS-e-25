#!/usr/bin/env python3
import rclpy
from rclpy.time import Time
import threading, time, math
import numpy as np
import matplotlib.pyplot as plt

# Import your custom modules
from control_v2.ros_connect import ROSInterface
from control_v2.control_utils import *
from control_v2.telemetryplot import TelemetryVisualizer, generate_turning_arc

# ================================
# Control Constants
# ================================
MAX_VELOCITY = 1.5
VEL_LIMIT_FACTOR = 0.6   # Slow down more in corners to prevent overshoot
ROUTE_IS_LOOP = False
STOP_SPEED_THRESHOLD = -10.1
WHEELBASE_M = 1.5 
MAX_STEER_RAD = 0.7 

# Optimization Constants
PREDICTION_DT = 0.4      # Reduced to 0.4s for tighter reaction on circles
SEARCH_START_IDX = 2    
SEARCH_END_IDX = 12      # Reduced search horizon to prevent cutting corners too early
STEP_SIZE = 1            # Check every point for maximum smoothness

# Stability Constants
STEER_ALPHA = 0.3        # Low Pass Filter: 0.2 = Smooth/Slow, 1.0 = Instant/Jittery
CONTINUITY_WEIGHT = 0.5  # Cost penalty per index jump (prevents teleporting)
SATURATION_WEIGHT = 15.0 # Cost penalty for exceeding steering limits

# ================================
# Visualization Constants
# ================================
VIZ_UPDATE_HZ = 20

def main():
    rclpy.init()
    node = ROSInterface()
    
    # CRITICAL: Sync with Simulator Clock
    node.set_parameters([rclpy.parameter.Parameter('use_sim_time', rclpy.Parameter.Type.BOOL, True)])

    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(node)
    threading.Thread(target=executor.spin, daemon=True).start()

    th_pid = PID(3.2, 0, 0)
    
    print("⏳ Waiting for first odometry message...")
    while True:
        cx, cy, yaw, speed, have_odom = node.get_state()
        if have_odom:
            print(f"✅ First position received.")
            break
        time.sleep(0.1)

    input("Press Enter to start control loop...")

    # ---------------------------------------------------------
    # STATE INITIALIZATION
    # ---------------------------------------------------------
    # Time Tracking
    last_ros_time = node.get_clock().now()
    last_viz_update_real = time.perf_counter() 
    
    # "Memory" variables for stability (The Hysteresis)
    prev_best_idx = 0       
    filtered_steer = 0.0    
    
    cur_idx = 0
    viz = None
    path_yaws = [] 

    plt.ion()

    while rclpy.ok():
        # 1. Time Management (Simulator Clock)
        current_ros_time = node.get_clock().now()
        dt_nano = (current_ros_time - last_ros_time).nanoseconds
        dt = dt_nano / 1e9 
        last_ros_time = current_ros_time
        
        # Skip small steps to protect PID/save CPU
        if dt < 0.001: 
            time.sleep(0.001)
            continue

        # 2. Perception & Path
        new_path = node.get_path()
        if not new_path:
            time.sleep(0.05)
            continue
            
        path_points = np.array(new_path)
        route_x, route_y = path_points[:, 0], path_points[:, 1]
        
        # Calculate Path Headings (Cache)
        if len(path_yaws) != len(route_x):
            if len(route_x) > 1:
                dx = np.gradient(route_x)
                dy = np.gradient(route_y)
                path_yaws = np.arctan2(dy, dx)
            else:
                path_yaws = np.zeros(len(route_x))

        if viz is None:
            viz = TelemetryVisualizer(route_x, route_y, np.full_like(route_x, MAX_VELOCITY))
            plt.show()

        # 3. Localization
        curve = compute_signed_curvature(route_x, route_y)
        cx, cy, yaw, speed, have_odom = node.get_state()
        cur_idx = local_closest_index((cx, cy), route_x, route_y, cur_idx, loop=ROUTE_IS_LOOP)
        
        # ---------------------------------------------------------
        # 4. OPTIMIZATION LOOP (Run EVERY Frame)
        # ---------------------------------------------------------
        best_cost = float('inf')
        
        # Default fallback: stick to previous plan or look slightly ahead
        best_idx = max(cur_idx, prev_best_idx)
        best_steer_req = filtered_steer # Default to keeping wheel steady
        best_pred_xy = (cx, cy)
        
        # DYNAMIC SEARCH WINDOW:
        # Prevent looking backwards. Search starts from where we were last time, 
        # or the car's current position, whichever is further along.
        start_search = max(cur_idx + SEARCH_START_IDX, prev_best_idx)
        end_search = min(len(route_x)-1, cur_idx + SEARCH_END_IDX)
        
        # Reset previous index if the car has looped or we reset the path
        if start_search >= end_search:
            start_search = cur_idx + SEARCH_START_IDX
            prev_best_idx = cur_idx # Reset memory
        
        for i in range(start_search, end_search, STEP_SIZE):
            tx, ty = route_x[i], route_y[i]
            
            # A. Inverse Kinematics (Geometric Steering)
            steer_req = get_steering_to_point(cx, cy, yaw, tx, ty, WHEELBASE_M)
            
            # Clamp for physics prediction (simulating reality)
            valid_steer = max(-MAX_STEER_RAD, min(MAX_STEER_RAD, steer_req))

            # B. Forward Prediction
            approx_arc_curve = math.tan(valid_steer) / WHEELBASE_M
            # Predict using a conservative speed (don't assume we can drift)
            pred_v = min(max(speed, 1.0), 3.0) 
            px, py, pyaw = predict_bicycle_state(cx, cy, yaw, pred_v, valid_steer, WHEELBASE_M, PREDICTION_DT)
            
            # C. Trajectory Cost
            cost, _ = calculate_trajectory_cost(px, py, pyaw, route_x, route_y, path_yaws, i)
            
            # --- STABILITY TERM 1: Saturation Penalty ---
            # Penalize points that require impossible steering angles
            if abs(steer_req) > MAX_STEER_RAD:
                cost += (abs(steer_req) - MAX_STEER_RAD) * SATURATION_WEIGHT

            # --- STABILITY TERM 2: Continuity (Hysteresis) ---
            # Penalize jumping far from the previous index
            dist_from_prev = abs(i - prev_best_idx)
            cost += dist_from_prev * CONTINUITY_WEIGHT

            if cost < best_cost:
                best_cost = cost
                best_idx = i
                best_steer_req = steer_req
                best_pred_xy = (px, py)

        # Update Memory
        prev_best_idx = best_idx 

        # ---------------------------------------------------------
        # 5. CONTROL OUTPUT & SMOOTHING
        # ---------------------------------------------------------
        
        # Clamp the requested steering to limits
        raw_target_steer = max(-MAX_STEER_RAD, min(MAX_STEER_RAD, best_steer_req))
        
        # --- STABILITY TERM 3: Low Pass Filter ---
        # Smooth out the high-frequency jitter
        # Formula: New = (Alpha * Target) + ((1-Alpha) * Old)
        filtered_steer = (STEER_ALPHA * raw_target_steer) + ((1.0 - STEER_ALPHA) * filtered_steer)
        
        steering_norm = filtered_steer / MAX_STEER_RAD

        # 6. Speed Control (Longitudinal)
        # Slow down based on the *actual filtered* steering being applied
        safe_idx = min(best_idx, len(curve) - 1)
        
        # Calculate speed limit based on steering angle (heavier steering = slower speed)
        target_speed = MAX_VELOCITY * (1.0 - (VEL_LIMIT_FACTOR * abs(steering_norm)))
        target_speed = max(1.0, min(MAX_VELOCITY, target_speed))

        accel_cmd = th_pid.update(target_speed - speed, dt=dt)
        accel_cmd = max(-3.0, min(3.0, accel_cmd))

        # 7. Visualization (Wall Time)
        if viz is not None:
            viz.update_path_data(route_x, route_y, np.full_like(route_x, MAX_VELOCITY))
            arc_pts = generate_turning_arc(cx, cy, yaw, filtered_steer, WHEELBASE_M)
            
            viz.log_state(
                x=cx, y=cy, yaw=yaw, speed=speed,
                steering_cmd=steering_norm,
                lookahead_pt=(route_x[best_idx], route_y[best_idx]),
                future_pts=[best_pred_xy], 
                arc_pts=arc_pts,
                target_speed=target_speed
            )

            if time.perf_counter() - last_viz_update_real >= (1.0 / VIZ_UPDATE_HZ):
                viz.update_plot_manual()
                plt.pause(0.001)
                last_viz_update_real = time.perf_counter()

        # 8. Actuation
        node.send_command(steering=filtered_steer, speed=target_speed, accel=accel_cmd)

        # 9. Stop Condition
        if (not ROUTE_IS_LOOP) and cur_idx >= len(route_x) - 5 and speed < STOP_SPEED_THRESHOLD:
            print("✅ End of route.")
            break
        
        time.sleep(0.05) 

    rclpy.shutdown()
    plt.ioff()
    plt.show()

if __name__ == '__main__':
    main()