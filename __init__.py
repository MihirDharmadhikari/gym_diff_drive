import logging
from gym.envs.registration import register

# Diff Drive
register(
    id='DiffDriveLidar16-v0',
    entry_point='gym_diff_drive.envs:DiffDriveLidar16',
    max_episode_steps=1500
)