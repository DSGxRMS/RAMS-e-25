#!/bin/bash
source /opt/ros/galactic/setup.bash
source /home/eufs/eufs_ws/install/setup.bash
source /mnt/d/RAMS-e-25/install/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp PYTHONUNBUFFERED=1 MPPI_VIZ=0 MPLBACKEND=Agg
cd /mnt/d/RAMS-e-25/src/controls
ros2 service call /ros_can/reset std_srvs/srv/Trigger >/dev/null 2>&1
ros2 service call /ros_can/set_mission eufs_msgs/srv/SetCanState "{ami_state: 21}" >/dev/null 2>&1
# 45s: ~13s torch import + graph capture + ~25s driving
timeout 45 python3 -u -c "from controls.control_node import main; main()" > /mnt/d/RAMS-e-25/src/controls/controls/_drive.log 2>&1
echo "exit=$?"
