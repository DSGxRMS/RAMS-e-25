from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    common = [{"use_sim_time": True}]

    return LaunchDescription([
        # # 1) Perception
        # Node(
        #     package="perception",
        #     executable="lidar_node",
        #     name="lidar",
        #     output="screen",
        #     parameters=common,
        # ),
        # Node(
        #     package="perception",
        #     executable="camera_node",
        #     name="camera",
        #     output="screen",
        #     parameters=common,
        # ),
        Node(
            package="perception",
            executable="fusion_node",
            name="fusion",
            output="screen",
            parameters=common,
        ),

        # 2) SLAM
        Node(
            package="slam",
            executable="pred_node",
            name="pred",
            output="screen",
            parameters=common,
        ),
        Node(
            package="slam",
            executable="slam_node",
            name="slam",
            output="screen",
            parameters=common + [{"gt_passthrough": True}],  # P17: on GT, bypass the PF
        ),

        # 3) Path planning: skidpad template planner (run only on the skidpad map)
        Node(
            package="pathplanning",
            executable="pp_skidpad_tels",
            name="pp_skidpad_tels",
            output="screen",
            parameters=common + [{
                "topics.odom_in": "/slam/odom_raw",
                "skidpad.line": "inner",            # "inner": circles at inner cones + clearance; "centre": track middle (R 9.125)
                "skidpad.inner_clearance_m": 1.0,   # inner cone centre -> car centre [m]; R = 7.625 + this
                "skidpad.drift_comp": True,         # move the circle inward by the car's measured outward drift
                "skidpad.drift_comp_max_m": 0.7,    # at most this much [m]
                "skidpad.min_plan_clearance_m": 0.3,  # shifted line never closer to inner cone centres [m]
                "skidpad.stop_target_back_m": 2.5,  # stop this far before the stop-area end cones [m] (middle of 5 m)
                "skidpad.stop_decel": 0.8,          # braking profile on the exit [m/s^2]
                "skidpad.overrun_m": 4.0,           # car rolls this far past the path end [m]; raise if it stops late
            }],
        ),

        # # 4) Controls
        # Node(
        #     package="controls",
        #     executable="control_node",
        #     name="control",
        #     output="screen",
        #     parameters=common,
        # ),
    ])
