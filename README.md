# gym_diff_drive
## Overview
Openai gym environment for differential drive robot with 2D lidar. Number of lidar beams can be changed inside the environment file. All obstacles are circular and spawned randomly on each episode. Obstacles are not allowed to intersect, this is taken care of during spawning.

**Actions**: array [ linear velocity (range: [0.0, 1.0]), angular velocity (range:[-2.0, 2.0]) ]  
Vector of size = 2

**Observations**: array [ lidar ranges(all beams), straight line distance to goal, angular deviation between current heading and straight line towards goal ]  
Vector of size = num of lidar beams + 2

## Instructions to use the package

Make sure your ```PYTHONPATH``` environment variable has the path to the directory of this package

In your code
```python
import gym_diff_drive.envs.diff_drive_lidar16
```
Use the following as environment id:
```
DiffDriveLidar16-v0
```
