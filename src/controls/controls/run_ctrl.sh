#!/bin/bash
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
source "$root/scripts/setup_integration.sh" || exit 1
source "${RAMS_INSTALL_SETUP:-$root/install/setup.bash}" || exit 1
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export PYTHONUNBUFFERED=1

ros2 service call /ros_can/reset std_srvs/srv/Trigger >/dev/null 2>&1
ros2 service call /ros_can/set_mission eufs_msgs/srv/SetCanState "{ami_state: 21}" >/dev/null 2>&1

stdbuf -oL -eL timeout 16 ros2 run controls control_node >/tmp/ctrl.log 2>&1
echo "=== exit code $? ==="
echo "=== captured $(wc -l < /tmp/ctrl.log) lines ==="
cat /tmp/ctrl.log
