import time
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
from collections import deque

# ================================
# Constants
# ================================
TELEMETRY_BUFFER_SIZE = 500
REWIND_LENGTH = 100
ERROR_PLOT_HISTORY = 200
MAX_STEER_RAD = math.pi/2  # 90 degrees 

class TelemetryVisualizer:
    def __init__(self, route_x, route_y, route_v):
        """
        :param route_x: Array of x coordinates for the reference path
        :param route_y: Array of y coordinates for the reference path
        :param route_v: Array of reference velocities
        """
        self.route_x = route_x
        self.route_y = route_y
        self.route_v = route_v
        
        # Use deque for O(1) appends/pops, much faster than list.pop(0)
        self.telemetry = deque(maxlen=TELEMETRY_BUFFER_SIZE)
        self.time_history = deque(maxlen=TELEMETRY_BUFFER_SIZE)
        
        self.rewind_mode = False
        self.rewind_index = 0
        self.start_time = time.perf_counter()

        # Setup standard matplotlib styling for scientific plotting
        plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'fast')
        
        self.fig = plt.figure(figsize=(14, 9))
        self.fig.canvas.manager.set_window_title("Autonomous Control Telemetry")
        
        # Grid layout: 2 columns, left is map (large), right is errors (stacked)
        # 5 rows: Velocity, Steering, Cross Track Error, Heading Error, HUD
        gs = self.fig.add_gridspec(5, 2, width_ratios=[2.5, 1], height_ratios=[1, 1, 1, 1, 0.8], hspace=0.4, wspace=0.2)
        
        # Main path plot (occupies all rows of column 0)
        self.ax_path = self.fig.add_subplot(gs[:, 0])
        
        # Velocity error plot (row 0, column 1)
        self.ax_vel = self.fig.add_subplot(gs[0, 1])
        
        # Steering error plot (row 1, column 1)
        self.ax_steer = self.fig.add_subplot(gs[1, 1])
        
        # Cross Track Error plot (row 2, column 1)
        self.ax_cte = self.fig.add_subplot(gs[2, 1])
        
        # Heading Error plot (row 3, column 1)
        self.ax_heading = self.fig.add_subplot(gs[3, 1])
        
        # Text/HUD area (row 4, column 1)
        self.ax_stat = self.fig.add_subplot(gs[4, 1])
        self.ax_stat.axis('off')

        self.setup_plot()
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

    def setup_plot(self):
        # ===== Main Path Plot =====
        # Create segments for colormap line (velocity heatmap)
        points = np.array([self.route_x, self.route_y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        # Normalize velocity for colormap
        norm = plt.Normalize(vmin=min(self.route_v), vmax=max(self.route_v))
        lc = LineCollection(segments, cmap='viridis', norm=norm, linewidths=3, alpha=0.6)
        lc.set_array(self.route_v)
        
        self.path_line = self.ax_path.add_collection(lc)
        
        # Set bounds with padding
        x_min, x_max = min(self.route_x), max(self.route_x)
        y_min, y_max = min(self.route_y), max(self.route_y)
        pad = 10
        self.ax_path.set_xlim(x_min - pad, x_max + pad)
        self.ax_path.set_ylim(y_min - pad, y_max + pad)
        self.ax_path.set_aspect('equal', adjustable='box')
        self.ax_path.set_title("Global Trajectory & State", fontsize=14, fontweight='bold')
        self.ax_path.set_xlabel("East (m)")
        self.ax_path.set_ylabel("North (m)")
        
        # Add colorbar
        self.cbar = self.fig.colorbar(lc, ax=self.ax_path, fraction=0.03, pad=0.04, label='Ref Speed (m/s)')

        # Dynamic Artists (Initialized empty)
        self.trace_line, = self.ax_path.plot([], [], 'k-', alpha=0.3, linewidth=1, label='History')
        self.lookahead_pt, = self.ax_path.plot([], [], 'X', color='orange', markersize=10, markeredgecolor='black', label='Lookahead')
        self.arc_line, = self.ax_path.plot([], [], 'r--', linewidth=2, alpha=0.8, label='Control Arc')
        
        # Vehicle Heading Arrow (Quiver is faster/cleaner than line for rotation)
        self.vehicle_quiver = self.ax_path.quiver([], [], [], [], color='red', scale=20, width=0.005, headwidth=5, zorder=10, label='Vehicle')

        self.ax_path.legend(loc='upper right', framealpha=0.9, fontsize='small')

        # ===== Velocity Plot =====
        self.ax_vel.set_title("Velocity Tracking", fontsize=10, fontweight='bold')
        self.ax_vel.set_ylabel("Speed (m/s)")
        self.vel_target_line, = self.ax_vel.plot([], [], 'g--', linewidth=1.5, label='Target')
        self.vel_actual_line, = self.ax_vel.plot([], [], 'b-', linewidth=2, alpha=0.8, label='Actual')
        self.ax_vel.legend(loc='upper left', fontsize='x-small')
        self.ax_vel.grid(True, linestyle=':', alpha=0.6)

        # ===== Steering Plot =====
        self.ax_steer.set_title("Steering Command", fontsize=10, fontweight='bold')
        self.ax_steer.set_ylabel("Angle (rad)")
        self.steer_cmd_line, = self.ax_steer.plot([], [], 'm-', linewidth=2, label='Command')
        self.steer_ref_line, = self.ax_steer.plot([], [], 'k-', linewidth=0.5, alpha=0.3) # Zero line
        self.ax_steer.grid(True, linestyle=':', alpha=0.6)

        # ===== Cross Track Error Plot =====
        self.ax_cte.set_title("Cross Track Error", fontsize=10, fontweight='bold')
        self.ax_cte.set_ylabel("CTE (m)")
        self.cte_line, = self.ax_cte.plot([], [], 'c-', linewidth=2, label='CTE')
        self.cte_ref_line, = self.ax_cte.plot([], [], 'k-', linewidth=0.5, alpha=0.3) # Zero line
        self.ax_cte.grid(True, linestyle=':', alpha=0.6)

        # ===== Heading Error Plot =====
        self.ax_heading.set_title("Heading Error", fontsize=10, fontweight='bold')
        self.ax_heading.set_ylabel("Error (rad)")
        self.heading_err_line, = self.ax_heading.plot([], [], 'orange', linewidth=2, label='Heading Err')
        self.heading_ref_line, = self.ax_heading.plot([], [], 'k-', linewidth=0.5, alpha=0.3) # Zero line
        self.ax_heading.grid(True, linestyle=':', alpha=0.6)

        # ===== HUD Stats Text =====
        self.hud_text = self.ax_stat.text(0.1, 0.5, "Initializing...", fontsize=12, family='monospace', va='center')

    def update_path_data(self, route_x, route_y, route_v):
        """Updates the global path line with new coordinates."""
        points = np.array([route_x, route_y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        self.path_line.set_segments(segments)
        self.path_line.set_array(route_v)

    def log_state(self, x, y, yaw, speed, steering_cmd, lookahead_pt, future_pts, arc_pts, target_speed, cross_track_error=0.0, heading_error=0.0):
        """
        Thread-safe logging of new state.
        """
        current_time = time.perf_counter() - self.start_time
        
        state = {
            't': current_time,
            'x': x, 'y': y, 'yaw': yaw, 'speed': speed,
            'steering': steering_cmd,
            'lookahead': lookahead_pt,
            'arc': arc_pts,
            'target_speed': target_speed,
            'cte': cross_track_error,
            'heading_err': heading_error
        }
        
        self.telemetry.append(state)
        self.time_history.append(current_time)

    def on_key(self, event):
        if event.key == 'r':
            self.rewind_mode = True
            # Deque doesn't slice, so we calculate start index
            self.rewind_index = max(0, len(self.telemetry) - REWIND_LENGTH)
            print(f"⏪ Rewind: Frame {self.rewind_index}/{len(self.telemetry)}")
        elif event.key == ' ':
            # Toggle pause logic could go here
            self.rewind_mode = False
            print("▶️ Live Mode")

    def update_plot_manual(self):
        """
        Refreshes the plot data. Call this at a throttled rate (e.g., 10-20Hz).
        """
        if not self.telemetry:
            return

        # Handle Rewind Logic
        if self.rewind_mode:
            if self.rewind_index < len(self.telemetry):
                data = self.telemetry[self.rewind_index]
                self.rewind_index += 1
            else:
                self.rewind_mode = False
                data = self.telemetry[-1]
        else:
            data = self.telemetry[-1]

        # 1. Update Vehicle Position & Heading
        # Using quiver set_UVC is very fast
        self.vehicle_quiver.set_offsets([data['x'], data['y']])
        self.vehicle_quiver.set_UVC(np.cos(data['yaw']), np.sin(data['yaw']))

        # Auto-center map on vehicle
        view_radius = 15  # meters
        self.ax_path.set_xlim(data['x'] - view_radius, data['x'] + view_radius)
        self.ax_path.set_ylim(data['y'] - view_radius, data['y'] + view_radius)

        # 2. Update History Trace
        # Convert deque to list only for plotting (fast enough for <1000 pts)
        trace_x = [p['x'] for p in self.telemetry]
        trace_y = [p['y'] for p in self.telemetry]
        self.trace_line.set_data(trace_x, trace_y)

        # 3. Update Lookahead
        if data['lookahead'] is not None:
            self.lookahead_pt.set_data([data['lookahead'][0]], [data['lookahead'][1]])
        else:
            self.lookahead_pt.set_data([], [])

        # 4. Update Control Arc
        if data['arc']:
            # data['arc'] is list of tuples [(x,y), (x,y)...]
            # unzip into two lists
            ax, ay = zip(*data['arc'])
            self.arc_line.set_data(ax, ay)
        else:
            self.arc_line.set_data([], [])

        # 5. Update Error Plots (Subplots)
        # Slicing deque via list conversion
        hist_len = min(len(self.telemetry), ERROR_PLOT_HISTORY)
        recent_data = list(self.telemetry)[-hist_len:]
        
        ts = [d['t'] for d in recent_data]
        v_tgt = [d['target_speed'] for d in recent_data]
        v_act = [d['speed'] for d in recent_data]
        steers = [d['steering'] * MAX_STEER_RAD for d in recent_data] # Convert norm to rads if needed

        self.vel_target_line.set_data(ts, v_tgt)
        self.vel_actual_line.set_data(ts, v_act)
        
        self.steer_cmd_line.set_data(ts, steers)
        self.steer_ref_line.set_data([ts[0], ts[-1]], [0, 0])

        # 5b. Update Cross Track Error Plot
        cte_vals = [d['cte'] for d in recent_data]
        self.cte_line.set_data(ts, cte_vals)
        self.cte_ref_line.set_data([ts[0], ts[-1]], [0, 0])

        # 5c. Update Heading Error Plot
        heading_errs = [d['heading_err'] for d in recent_data]
        self.heading_err_line.set_data(ts, heading_errs)
        self.heading_ref_line.set_data([ts[0], ts[-1]], [0, 0])

        # 6. Smart Axis Rescaling
        if ts:
            # Prevent singular transformation if timestamps are identical
            t_min, t_max = ts[0], ts[-1]
            if t_max <= t_min:
                t_max = t_min + 0.1

            self.ax_vel.set_xlim(t_min, t_max)
            self.ax_steer.set_xlim(t_min, t_max)
            self.ax_cte.set_xlim(t_min, t_max)
            self.ax_heading.set_xlim(t_min, t_max)
            
            # Y-Axis Auto-scaling with padding
            v_min, v_max = min(v_act + v_tgt), max(v_act + v_tgt)
            self.ax_vel.set_ylim(v_min - 1.0, v_max + 1.0)
            
            s_min, s_max = min(steers), max(steers)
            self.ax_steer.set_ylim(min(-0.5, s_min - 0.2), max(0.5, s_max + 0.2))
            
            # CTE scaling
            cte_min, cte_max = min(cte_vals), max(cte_vals)
            cte_pad = max(0.5, (cte_max - cte_min) * 0.2)
            self.ax_cte.set_ylim(cte_min - cte_pad, cte_max + cte_pad)
            
            # Heading error scaling
            he_min, he_max = min(heading_errs), max(heading_errs)
            he_pad = max(0.2, (he_max - he_min) * 0.2)
            self.ax_heading.set_ylim(he_min - he_pad, he_max + he_pad)

        # 7. Update HUD Text
        status_str = (
            f"TIME:  {data['t']:.1f} s\n"
            f"SPEED: {data['speed']:.2f} / {data['target_speed']:.2f} m/s\n"
            f"STEER: {data['steering']:.2f} (norm)\n"
            f"CTE:   {data['cte']:.3f} m\n"
            f"HEAD:  {data['heading_err']:.3f} rad"
        )
        self.hud_text.set_text(status_str)

        # 8. Render
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()


def generate_turning_arc(x, y, yaw, steering_angle, wheelbase, n_points=20, arc_length=8.0):
    """
    Generates (x,y) points representing the vehicle's predicted path 
    based on current steering angle (Ackermann kinematics).
    """
    if abs(steering_angle) < 0.01:
        # Straight line if steering is near zero
        return [
            (x + i * (arc_length/n_points) * math.cos(yaw), 
             y + i * (arc_length/n_points) * math.sin(yaw)) 
            for i in range(n_points)
        ]
        
    R = wheelbase / math.tan(abs(steering_angle))
    direction = 1 if steering_angle > 0 else -1
    
    # Generate points along the circle arc
    arc = []
    for i in range(n_points):
        dist = i * (arc_length / n_points)
        theta = dist / R # Angle subtended
        
        # Center of rotation
        cx = x - direction * R * math.sin(yaw)
        cy = y + direction * R * math.cos(yaw)
        
        # Point on perimeter
        px = cx + direction * R * math.sin(yaw + theta * direction)
        py = cy - direction * R * math.cos(yaw + theta * direction)
        arc.append((px, py))
        
    return arc