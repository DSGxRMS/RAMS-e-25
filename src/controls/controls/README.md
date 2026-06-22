# Controls Code Testing for RL based MPPI Controller
## Notes:
- In this branch ```controls_rl``` the input data from upstream has been set such that - 
    - Perception: Runs a singular node as fusion_node, giving us ground truth cones data thoroughly accounted for without deviations
    - SLAM: Bypasses the prediction node to predict on the ground truth itself - runs with 100% accuracy (afaik)
    - Path Planning: Since SLAM and Perception both run on GT data, the resulting map, and the subsequent path generated is accurate though PP has chances to deviate depending upon the visibility of cones as per camera's FOV (which in this case is restricted)
    - Resulting controls algo should take into account this deviation however. Expect ~5-7m distance wrt points and as low as 3m on curves
    - PP is inconsistent, if possible train your algo to slow down if no path points received, and continue to follow the last yaw command until PP recovers...

- For the purpose of controls, check the repo structure
```
controls\
        |-control_node.py (your active control node - change the logic in here)
        |-control_utils.py (contains all support functions)
        |-ros_connect.py (separated communicator wrt to the simulator)
```

Do not worry about slam/perception/pp, they have already been handled within the scope of the branch
However, the uncertainty is to be worked with as is. Ensure the agent you work with doesn't mess with any of the other nodes only to make the work of controls easier...

# Good Luck guys!!