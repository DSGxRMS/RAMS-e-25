#!/usr/bin/env bash
# Build from this checkout and keep generated files on the Ubuntu filesystem.
usage() {
    printf 'Usage: bash %s {build [package ...]|sim|planner|controls|debug}\n' "${BASH_SOURCE[0]}"
}

action="${1:-}"
case "$action" in
    -h|--help) usage; exit 0 ;;
    build) shift ;;
    sim|planner|controls|debug)
        if [[ $# -ne 1 ]]; then
            usage >&2
            exit 2
        fi
        ;;
    *) usage >&2; exit 2 ;;
esac

source_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)" || exit 1
workspace="$(realpath -m -- "${RAMS_WSL_WORKSPACE:-$HOME/rams-wsl}")" || exit 1
install_dir="$workspace/install"
ros_python="${RAMS_ROS_PYTHON:-$HOME/.venvs/rams-galactic/bin/python3}"
source "$source_root/scripts/setup_integration.sh" || exit 1

if [[ "$action" == build || "$action" == controls || "$action" == debug ]]; then
    if [[ ! -x "$ros_python" ]]; then
        printf 'Galactic Python not found at %s. Set RAMS_ROS_PYTHON to its executable.\n' "$ros_python" >&2
        exit 1
    fi
fi

if [[ "$action" == build ]]; then
    if [[ $# -eq 0 ]]; then
        set -- bringup perception slam pathplanning controls
    fi
    mkdir -p -- "$workspace" && cd -- "$workspace" || exit 1
    printf '[integration] Building source %s into %s\n' "$source_root" "$install_dir"
    exec "$ros_python" /usr/bin/colcon --log-base "$workspace/log" build \
        --build-base "$workspace/build" --install-base "$install_dir" \
        --base-paths "$source_root/bringup" "$source_root/src" --packages-select "$@"
fi

if [[ ! -r "$install_dir/local_setup.bash" ]]; then
    printf 'Build first with bash "%s/scripts/run_integration.sh" build\n' "$source_root" >&2
    exit 1
fi
cd -- "$workspace" || exit 1
source "$install_dir/local_setup.bash" || exit 1
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"

case "$action" in
    sim)
        exec ros2 launch eufs_launcher eufs_launcher.launch.py
        ;;
    planner)
        export MPLBACKEND="${MPLBACKEND:-Agg}"
        exec ros2 launch bringup "${RAMS_PLANNER_LAUNCH:-stack_tels_new.launch.py}"
        ;;
    controls)
        export MPPI_WORKER_PYTHON="${MPPI_WORKER_PYTHON-$HOME/.venvs/rams-mppi-gpu/bin/python}"
        export MPPI_MODEL="${MPPI_MODEL:-exp13}" MPPI_SEED="${MPPI_SEED:-42}"
        export MPPI_RUNS_DIR="${MPPI_RUNS_DIR:-$workspace/runs}"
        if [[ -n "$MPPI_WORKER_PYTHON" && ! -x "$MPPI_WORKER_PYTHON" ]]; then
            printf 'CUDA Python not found at %s. Check MPPI_WORKER_PYTHON.\n' "$MPPI_WORKER_PYTHON" >&2
            exit 1
        fi
        printf '[integration] Source %s\n[integration] Install %s\n' "$source_root" "$install_dir"
        exec ros2 run --prefix "$ros_python" controls control_node
        ;;
    debug)
        export MPLBACKEND="${MPLBACKEND:-TkAgg}"
        exec "$ros_python" "$source_root/tools/birdview.py"
        ;;
esac
