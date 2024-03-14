from typing import Union, Tuple, Optional, Any, Dict

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.core import ObsType


class BaseCrowdNavigationEnv(gym.Env):
    """
    Base class crowd navigation. Units are defined to reflect plausible values in the real
    world, e.g. maximum velocity of 3m/s while walking.

    Args:
        n_crowd (int): number of members in the crowd
        width (int): width of the environment (in meters)
        hieght (int): hieght of the environment (in meters)
        allow_collision (bool): collisions between members of the crowd
    """
    def __init__(
        self,
        n_crowd: int,
        width: int = 20,
        height: int = 20,
        interceptor_percentage: float = 0.5,
        allow_collision: bool = False,
    ):
        super().__init__()

        self._dt = 0.1

        self.WIDTH = width
        self.HEIGHT = height
        self.W_BORDER = self.WIDTH / 2
        self.H_BORDER = self.HEIGHT / 2
        self.AGENT_MAX_VEL = 3.0
        self.CROWD_MAX_VEL = 2.5
        self.PHYSICAL_SPACE = 0.4
        self.PERSONAL_SPACE = 1.4
        self.SOCIAL_SPACE = 1.9
        self.MAX_ACC = 1.5
        self.MAX_STOPPING_TIME = self.AGENT_MAX_VEL / self.MAX_ACC
        self.MAX_STOPPING_DIST = self.AGENT_MAX_VEL * self.MAX_STOPPING_TIME -\
            0.5 * self.MAX_ACC * self.MAX_STOPPING_TIME ** 2
        self.INTERCEPTOR_PERCENTAGE = interceptor_percentage

        self.COLLISION_REWARD = -10
        self.Cc = 2 * self.PHYSICAL_SPACE * \
            np.log(-self.COLLISION_REWARD / self.MAX_EPISODE_STEPS + 1)
        self.Cg = -(1 - np.exp(self.Cc / self.SOCIAL_SPACE)) /\
            np.sqrt(self.WIDTH ** 2 + self.HEIGHT ** 2)
        self.Tc = -self.COLLISION_REWARD / 2

        self.n_crowd = n_crowd
        self.allow_collision = allow_collision
        (
            self._agent_pos,
            self._agent_vel,
            self._goal_pos,
            self._crowd_poss,
            self._crowd_vels
        ) = self._start_env_vars()

        state_bound_min = np.hstack([
            [-self.WIDTH, -self.HEIGHT] * (self.n_crowd + 1),
            [0] * (self.n_crowd + 1),
        ])
        state_bound_max = np.hstack([
            [self.WIDTH , self.HEIGHT] * (self.n_crowd + 1),
            [self.AGENT_MAX_VEL],
            [self.CROWD_MAX_VEL] * (self.n_crowd)
        ])

        self.observation_space = spaces.Box(
            low=state_bound_min, high=state_bound_max, shape=state_bound_min.shape
        )

        # containers for plotting
        self.metadata = {'render.modes': ["human"]}
        self.fig = None

        self._steps = 0
        self._current_reward = 0
        self._goal_reached = False
        self.check_goal_reached = lambda : (
            np.linalg.norm(self._agent_pos - self._goal_pos) < self.PHYSICAL_SPACE and
            np.linalg.norm(self._agent_vel) < self.MAX_ACC * self._dt
        )
        self.current_trajectory = np.zeros((40, 2))


    def set_trajectory(self, positions, velocities):
        positions = positions[:10]
        velocities = velocities[:10]

        positions -= positions[0]
        positions += self._agent_pos + self._agent_vel * self._dt

        # velocities[0] += self._agent_vel * self._dt
        # positions *= 0
        # distances = velocities * self._dt
        # positions[0] = self._agent_pos
        # positions += distances
        # positions = np.cumsum(positions, 0)
        self.current_trajectory = positions


    @property
    def dt(self) -> Union[float, int]:
        return self._dt


    @property
    def current_pos(self):
        return self._agent_pos.copy()


    @property
    def current_vel(self):
        return self._agent_vel.copy()


    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
        ) -> Tuple[ObsType, Dict[str, Any]]:
        super(BaseCrowdNavigationEnv, self).reset(seed=seed, options=options)
        (
            self._agent_pos,
            self._agent_vel,
            self._goal_pos,
            self._crowd_poss,
            self._crowd_vels
        ) = self._start_env_vars()
        self._steps = 0
        self._goal_reached = False
        return self._get_obs().copy(), {}


    def _start_env_vars(self):
        agent_pos = np.zeros(2)
        agent_vel = np.zeros(2)
        while True:
            goal_pos = np.random.uniform(
                [-self.WIDTH / 2 + self.PHYSICAL_SPACE,
                 -self.HEIGHT / 2 + self.PHYSICAL_SPACE],
                [self.WIDTH / 2 - self.PHYSICAL_SPACE,
                 self.HEIGHT / 2 - self.PHYSICAL_SPACE]
            )

            # Place the first crowd member between the agent and the goal
            direction_to_goal = goal_pos - agent_pos
            distance_to_goal = np.linalg.norm(direction_to_goal)
            norm_to_goal = direction_to_goal / distance_to_goal
            interceptor_pos = agent_pos + norm_to_goal * np.random.uniform(
                self.PERSONAL_SPACE, distance_to_goal - self.PERSONAL_SPACE
            )
            if np.linalg.norm(interceptor_pos - agent_pos) > self.PERSONAL_SPACE * 2 and\
               np.linalg.norm(interceptor_pos - goal_pos) > self.PERSONAL_SPACE:
                break
        # Add perpendicular noise
        perp_direction = np.array([-norm_to_goal[1], norm_to_goal[0]])
        noise = perp_direction * np.random.uniform(
            -self.PERSONAL_SPACE / self.INTERCEPTOR_PERCENTAGE,
            self.PERSONAL_SPACE / self.INTERCEPTOR_PERCENTAGE)
        noised_interceptor = interceptor_pos + noise

        crowd_poss = np.zeros((self.n_crowd, 2))
        if self.n_crowd > 0:
            crowd_poss[0] = noised_interceptor
        for i in range(self.n_crowd):
            while True:
                sampled_pos = np.random.uniform(
                    [-self.WIDTH / 2, -self.HEIGHT / 2],
                    [self.WIDTH / 2, self.HEIGHT / 2],
                )
                no_crowd_collision = self.allow_collision or i == 0
                if not self.allow_collision and i > 0:
                    no_crowd_collision = np.sum(np.linalg.norm(  # at least one collision
                            crowd_poss[:i] - sampled_pos, axis=-1
                        ) < self.PERSONAL_SPACE * 2
                    ) == 0
                if (np.linalg.norm(sampled_pos - agent_pos) > self.PERSONAL_SPACE * 2 and
                    np.linalg.norm(sampled_pos - goal_pos) > self.PERSONAL_SPACE * 2 and
                    no_crowd_collision):
                    crowd_poss[i] = sampled_pos
                    break

        # Shuffle crowd positions
        np.random.shuffle(crowd_poss)
        crowd_vels = np.random.uniform(
            -self.CROWD_MAX_VEL, self.CROWD_MAX_VEL, (self.n_crowd, 2)
        )
        return agent_pos, agent_vel, goal_pos, crowd_poss, crowd_vels


    def update_state(self, acc):
        """
        Update robot position and velocity for time self._dt based on its dynamics.

        Args:
            acc (numpy.ndarray): 2D array representing the accelaration for current step
        """
        if self.discrete_action:
            acc = np.array([
                self.CARTESIAN_ACC[acc[0]], self.CARTESIAN_ACC[acc[1]]
            ])

        acc_norm = np.linalg.norm(acc)
        if acc_norm > self.MAX_ACC:
            acc *= self.MAX_ACC / acc_norm

        self._agent_pos += self._agent_vel * self._dt + acc * 0.5 * self._dt ** 2
        self._agent_vel += acc * self._dt

        # check bounds of the environment and the bounds of the maximum velocity
        self._agent_pos = np.clip(
            self._agent_pos,
            [-self.W_BORDER, -self.H_BORDER],
            [self.W_BORDER, self.H_BORDER]
        )
        agent_speed = np.linalg.norm(self._agent_vel)
        if agent_speed > self.AGENT_MAX_VEL:
            self._agent_vel *= self.AGENT_MAX_VEL / agent_speed


    def _get_reward(self, action: np.ndarray) -> (float, dict):
        raise NotImplementedError


    def _get_obs(self) -> ObsType:
        raise NotImplementedError


    def _check_collisions(self) -> bool:
        """
        Checks whether agent is to close to at leas one member of the crowd or is
        colliding with a wall
        """
        # Crowd
        if np.sum(np.linalg.norm(self._agent_pos - self._crowd_poss, axis=-1) <
            [self.PHYSICAL_SPACE * 2] * self.n_crowd):
            return True
        # Walls
        if np.sum(np.abs(self._agent_pos) >
            np.array([self.W_BORDER, self.H_BORDER]) - self.PHYSICAL_SPACE):
            return True
        return False


    def _terminate(self, info) -> bool:
        raise NotImplementedError


    def close(self):
        super(BaseCrowdNavigationEnv, self).close()
        del self.fig
