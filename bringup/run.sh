#!/usr/bin/env bash

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd) || exit 1
source "$repo_root/scripts/setup_integration.sh" || exit 1
source "$repo_root/install/local_setup.bash" || exit 1

export MPLBACKEND="${MPLBACKEND:-TkAgg}"
exec ros2 launch "$repo_root/bringup/launch/stack.launch.py" "$@"
