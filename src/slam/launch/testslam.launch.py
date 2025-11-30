from launch import LaunchDescription
from launch.actions import TimerAction, ExecuteProcess
from launch_ros.actions import Node
from launch.substitutions import EnvironmentVariable, PathJoinSubstitution


def generate_launch_description():
    # ----------------- PERCEPTION NODES -----------------
    camera_node = Node(
        package="perception",
        executable="camera_node",
        name="camera_node",
        output="screen",
        # parameters=[{...}]  # add if you have YAML/params
    )

    # Start lidar + fusion after camera has had a moment to init
    lidar_node = Node(
        package="perception",
        executable="lidar_node",
        name="lidar_node",
        output="screen",
    )

    fusion_node = Node(
        package="perception",
        executable="fusion_node",
        name="fusion_node",
        output="screen",
    )

    start_lidar_and_fusion = TimerAction(
        period=5.0,   # seconds after launch; tune if needed
        actions=[lidar_node, fusion_node],
    )

    # ----------------- SLAM NODES -----------------
    pred_node = Node(
        package="slam",
        executable="pred_node",
        name="pred_node",
        output="screen",
        # parameters=[{...}]  # e.g. topics.imu, topics.out_odom if needed
    )

    slam_node = Node(
        package="slam",
        executable="slam_node",
        name="slam_node",
        output="screen",
        # parameters=[{...}]
    )

    slam_debug = Node(
        package="slam",
        executable="slam_debug",
        name="slam_debug",
        output="screen",
        # parameters=[{
        #     "topics.odom_in": "/slam/odom",
        #     "topics.map_in": "/slam/map_cones",
        # }]
    )

    # Start slam_node + slam_debug 5 seconds after pred_node
    start_slam_and_debug = TimerAction(
        period=5.0,   # seconds after launch
        actions=[slam_node, slam_debug],
    )

    # ----------------- ROSBAG PLAY -----------------
    bag_path = PathJoinSubstitution([
        EnvironmentVariable("HOME"),
        "eufs_dev",
        "eufs_data",
        "ros_bags",
        "comp_map_control",
    ])

    # ----------------- ASSEMBLE -----------------
    return LaunchDescription([
        # Perception
        camera_node,
        start_lidar_and_fusion,

        # SLAM
        pred_node,
        start_slam_and_debug,
    ])
