#!/usr/bin/env bash
# Source this file in Bash before building or running the integration.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "Source this script: source scripts/setup_integration.sh" >&2
    exit 1
fi
if [[ ! -r "${ROS_SETUP:-/opt/ros/galactic/setup.bash}" ]]; then
    echo "ROS Galactic setup not found. Set ROS_SETUP to its setup.bash." >&2
    return 1
fi
if [[ ! -r "${EUFS_WS:-$HOME/eufs_ws}/install/setup.bash" ]]; then
    echo "EUFS setup not found. Set EUFS_WS to your built EUFS workspace." >&2
    return 1
fi
source "${ROS_SETUP:-/opt/ros/galactic/setup.bash}" || return
source "${EUFS_WS:-$HOME/eufs_ws}/install/setup.bash" || return
export EUFS_MASTER="${EUFS_MASTER:-${EUFS_WS:-$HOME/eufs_ws}/src/eufs_sim}"
