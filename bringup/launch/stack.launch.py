from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    common = [{"use_sim_time": True}]

    return LaunchDescription([
        # 1) Perception
        Node(
            package="perception",
            executable="lidar_node",
            name="lidar",
            output="screen",
            parameters=common,
        ),
        Node(
            package="perception",
            executable="camera_node",
            name="camera",
            output="screen",
            parameters=common,
        ),
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
            # parameters=[{"use_sim_time": True}],
            parameters=common,
        ),



        # 3) Path planning
        Node(
            package="pathplanning",
            executable="pp_node_skidpad",
            name="pp",
            output="screen",
            parameters=common,
        ),

        # 4) Controls
        Node(
            package="controls",
            executable="control_node",
            name="control",
            output="screen",
            parameters=common,
        ),
    ])
