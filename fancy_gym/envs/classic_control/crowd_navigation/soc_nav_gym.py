from typing import Union, Tuple, Optional, Any, Dict

import socnavgym
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.core import ObsType


class SocNavEnv(socnavgym.envs.SocNavEnv_v1):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._dt = 0.1


    def set_trajectory(self, positions, velocities=None):
        positions = positions[:10]
        velocities = velocities[:10]

        positions -= positions[0]
        positions += self._agent_pos + self._agent_vel * self._dt
        self.current_trajectory = positions.copy()

        velocities[0] += self._agent_vel * self._dt
        positions = positions * 0
        distances = velocities * self._dt
        positions[0] = self._agent_pos
        positions += distances
        positions = np.cumsum(positions, 0)
        self.current_trajectory_vel = positions.copy()

    @property
    def dt(self) -> Union[float, int]:
        return self._dt


    @property
    def current_pos(self):
        return np.array([self.robot.x, self.robot.y])


    @property
    def current_vel(self):
        return np.array([self.robot.vel_x, self.robot.vel_y, self.vel_a])


    @property
    def crowd_pos_vel(self):
        poss, vels = [], []
        for human in self.dynamic_humans:
            poss.append(np.array([human.x, human.y]))
            vels.append(
                np.array([np.sin(human.orientation), np.cos(human.orientation)]) *
                human.speed
            )
        return (np.array(poss), np.array(vels))


    @property
    def wall_dist(self):
        poss = []
        for wall in self.walls:
            poss.append(np.array([wall.x, wall.y]))
        return np.array(poss)
