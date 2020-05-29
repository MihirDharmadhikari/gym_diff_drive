# gym_diff_drive
## Overview
Openai gym environment for differential drive robot with 2D lidar
Number of lidar beams can be changed
All obstacles are circular and spawned randomly at each episode. Obstacles are not allowed to intersect, this is taken care during spawning

Actions: linear velocity [0.0, 1.0], angular velocity [-2.0, 2.0]

Observations: lidar ranges(all beams), straight line distance to goal, Angular deviation between current heading and straight line towards goal

## Instructions to run

Make sure your PYTHONPATH has the path to the directory of this package

In your code
```python
import gym_diff_drive.envs.diff_drive_lidar16
```
Use the following as environment id:
```python
DiffDriveLidar16-v0
```
