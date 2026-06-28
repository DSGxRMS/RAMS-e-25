#!/bin/bash
source /opt/ros/galactic/setup.bash
source ~/eufs_ws/install/setup.bash
source /mnt/d/RAMS-e-25/install/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export PYTHONUNBUFFERED=1

ros2 service call /ros_can/reset std_srvs/srv/Trigger >/dev/null 2>&1
ros2 service call /ros_can/set_mission eufs_msgs/srv/SetCanState "{ami_state: 21}" >/dev/null 2>&1

stdbuf -oL -eL timeout 16 ros2 run controls control_node >/tmp/ctrl.log 2>&1
echo "=== exit code $? ==="
echo "=== captured $(wc -l < /tmp/ctrl.log) lines ==="
cat /tmp/ctrl.log
