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

        # 3) Path planning
        Node(
            package="pathplanning",
            executable="pp_tls",
            name="pp_tls",
            output="screen",
            parameters=common + [{"topics.odom_in": "/slam/odom_raw"}],
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
