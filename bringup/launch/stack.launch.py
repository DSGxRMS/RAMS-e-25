"""Launch the TLS planner stack with `bash bringup/run.sh` from the checkout."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    common = [{"use_sim_time": True}]
    small_track_cone_memory = {
        "cone_memory.boundary_max_edge": 4.0,
        "cone_memory.anchor_ahead": 7.0,
        "cone_memory.entry_gap": 6.5,
        "cone_memory.nominal_width": 4.8,
        "cone_memory.width_max": 5.5,
        "cone_memory.connector_max_lead": 5.0,
    }

    return LaunchDescription([
        DeclareLaunchArgument(
            "planner",
            default_value="pp_tls",
            description="Pathplanning executable, such as pp_tls, pp_cone_memory, pp_hybrid_turn or pp_shadow_only.",
        ),
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
            executable=LaunchConfiguration("planner"),
            name="pp",
            output="screen",
            parameters=common + [{"topics.odom_in": "/slam/odom_raw", "map.observation_snapshots": True},
                                 small_track_cone_memory],
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
