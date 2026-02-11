# RAMS'e - 25
First edition legacy stack - version 1.0

## To Operate, refer:

### 1. Setting up the stack for the FIRST time
Get to the repo -
```
cd RAMS-e-25
```
Then, build the ros packages
```
colcon build --symlink-install
```
Store the package runner in ~/.bashrc for further use
```echo "source ~/eufs_dev/RAMS-e-25/install/setup.bash" >> ~/.bashrc 
```
Restart the terminal (sources the package)

#Note: Ensure to change the path of the folder as per local structure!


### 2. Using the stack
To run the stack until controls - 
``` 
ros2 launch bringup stack.launch.py
```

To run Controls - 
```
ros2 run controls control_node
```

#### #NOTE: Do ensure to set the driving mode to manual drive for the car to move

