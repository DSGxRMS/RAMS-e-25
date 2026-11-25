#!/bin/bash
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
source "$root/scripts/setup_integration.sh" || exit 1
source "${RAMS_INSTALL_SETUP:-$root/install/setup.bash}" || exit 1
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp PYTHONUNBUFFERED=1 MPPI_VIZ=0 MPLBACKEND=Agg
ros2 service call /ros_can/reset std_srvs/srv/Trigger >/dev/null 2>&1
ros2 service call /ros_can/set_mission eufs_msgs/srv/SetCanState "{ami_state: 21}" >/dev/null 2>&1
# 45s: ~13s torch import + graph capture + ~25s driving
timeout 45 python3 -u -c "from controls.control_node import main; main()" > "$root/_drive.log" 2>&1
echo "exit=$?"
