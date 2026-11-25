from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import LaunchConfigurationEquals
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    common = [{"use_sim_time": True}]

    return LaunchDescription([
        DeclareLaunchArgument(
            "cone_source",
            default_value="ground_truth",
            choices=["ground_truth", "perception"],
            description="Source of cone observations for SLAM.",
        ),

        # 1) Perception (used only in sensor mode)
        Node(
            package="perception",
            executable="lidar_node",
            name="lidar",
            condition=LaunchConfigurationEquals("cone_source", "perception"),
            output="screen",
            parameters=common,
        ),
        Node(
            package="perception",
            executable="camera_node",
            name="camera",
            condition=LaunchConfigurationEquals("cone_source", "perception"),
            output="screen",
            parameters=common,
        ),
        Node(
            package="perception",
            executable="fusion_node",
            name="fusion",
            condition=LaunchConfigurationEquals("cone_source", "perception"),
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
            parameters=common + [{"cones.source": LaunchConfiguration("cone_source")}],
        ),

        # 3) Path planning
        Node(
            package="pathplanning",
            executable="pp_node_skidpad",
            name="pp",
            output="screen",
            parameters=common,
        ),

        # # 4) Controls
        # Node(
        #     package="control_v2",
        #     executable="control_loop",
        #     name="control",
        #     output="screen",
        #     parameters=common,
        # ),
    ])
