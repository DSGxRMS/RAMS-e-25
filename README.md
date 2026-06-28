````md
# RAMS'e - 25

First edition legacy stack - Version 1.0.

## Documentation

- [System Architecture](ARCHITECTURE.md)

---

## Getting Started

### 1. Initial Setup

> **Note:** Update the project path according to your local directory structure.

Navigate to the repository:

```bash
cd RAMS-e-25
````

Build the ROS 2 workspace:

```bash
colcon build --symlink-install
```

Add the workspace to your `~/.bashrc` so it is sourced automatically in future terminals:

```bash
echo "source ~/eufs_dev/RAMS-e-25/install/setup.bash" >> ~/.bashrc
```

Restart the terminal (or run `source ~/.bashrc`) to apply the changes.

---

## Running the Stack

### Launch the perception, SLAM, and planning stack

```bash
ros2 launch bringup stack.launch.py
```

### Launch the controller

```bash
ros2 run controls control_node
```

> **Important:** Before running the controller, set the EUFS simulator driving mode to **Manual ("Go")** so the vehicle accepts `/cmd` commands.

```
```
