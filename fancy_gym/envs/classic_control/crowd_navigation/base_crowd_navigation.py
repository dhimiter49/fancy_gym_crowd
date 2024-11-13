from typing import Union, Tuple, Optional, Any, Dict

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.core import ObsType


seed = 0
flip = True

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
        discrete_action: bool = False,
        velocity_control: bool = False,
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
        if type(self).__name__ == "CrowdNavigationEnv":
            self.MIN_CROWD_DIST = self.MAX_STOPPING_DIST * 1.1
        else:
            self.MIN_CROWD_DIST = self.PERSONAL_SPACE + self.PHYSICAL_SPACE

        self.COLLISION_REWARD = -10
        self.Cc = 2 * self.PHYSICAL_SPACE * \
            np.log(-self.COLLISION_REWARD / self.MAX_EPISODE_STEPS + 1)
        self.Cg = -(1 - np.exp(self.Cc / self.SOCIAL_SPACE)) /\
            np.sqrt(self.WIDTH ** 2 + self.HEIGHT ** 2)
        self.Tc = -self.COLLISION_REWARD
        self.Cc *= 2

        self.n_crowd = n_crowd
        self.allow_collision = allow_collision
        self.rot_mat = lambda deg: np.array([
            [np.cos(deg), -np.sin(deg)], [np.sin(deg), np.cos(deg)]
        ])
        (
            self._agent_pos,
            self._agent_vel,
            self._goal_pos,
            self._crowd_poss,
            self._crowd_vels
        ) = self._start_env_vars()


        self.discrete_action = discrete_action
        self.velocity_control = velocity_control
        if self.velocity_control:
            if self.discrete_action:
                self.CARTESIAN_VEL = np.arange(
                    -self.AGENT_MAX_VEL, self.AGENT_MAX_VEL, self.AGENT_MAX_VEL * 2 / 20
                )
                self.action_space = spaces.MultiDiscrete(
                    [len(self.CARTESIAN_VEL), len(self.CARTESIAN_VEL)]
                )
            elif self.polar:
                self.action_space = spaces.Box(
                    low=np.array([0, -np.pi]),
                    high=np.array([self.AGENT_MAX_VEL, np.pi]),
                )
            else:
                action_bound = np.array([self.AGENT_MAX_VEL, self.AGENT_MAX_VEL])
                self.action_space = spaces.Box(
                    low=-action_bound, high=action_bound, shape=action_bound.shape
                )
        else:
            if self.discrete_action:
                self.CARTESIAN_ACC = np.arange(
                    -self.MAX_ACC, self.MAX_ACC, self.MAX_ACC * 2 / 20
                )
                self.action_space = spaces.MultiDiscrete(
                    [len(self.CARTESIAN_ACC), len(self.CARTESIAN_ACC)]
                )
            else:
                action_bound = np.array([self.MAX_ACC, self.MAX_ACC])
                self.action_space = spaces.Box(
                    low=-action_bound, high=action_bound, shape=action_bound.shape
                )

        state_bound_min = np.hstack([
            [-self.WIDTH, -self.HEIGHT] * (self.n_crowd + 1),
            [0] * (self.n_crowd + 1),
        ])
        state_bound_max = np.hstack([
            [self.WIDTH, self.HEIGHT] * (self.n_crowd + 1),
            [self.AGENT_MAX_VEL],
            [self.CROWD_MAX_VEL] * (self.n_crowd)
        ])

        self.observation_space = spaces.Box(
            low=state_bound_min, high=state_bound_max, shape=state_bound_min.shape
        )

        # containers for plotting
        self.metadata = {'render_modes': ["human"], 'render_fps': 24}
        self.fig = None

        self._steps = 0
        self._current_reward = 0
        self._goal_reached = False
        self._is_collided = False
        self.check_goal_reached = lambda: (
            np.linalg.norm(self._agent_pos - self._goal_pos) < self.PHYSICAL_SPACE and
            np.linalg.norm(self._agent_vel) < self.MAX_ACC * self._dt
        )
        self.traj_idx = 0
        self.current_trajectory = np.zeros((100, 2))
        self.pred_current_trajectory = np.zeros((100, 2))
        self.exec_traj = []
        self.current_trajectory_vel = np.zeros((100, 2))
        self._traj_index = 0
        self.separating_planes = np.zeros((self.n_crowd, 4))


    def set_trajectory(self, positions, velocities=None):
        self._traj_index = 0
        positions = positions[:]
        # velocities = velocities[:10]

        positions -= positions[0]
        positions += self._agent_pos + self._agent_vel * self._dt
        self.current_trajectory[self.traj_idx * 10:(self.traj_idx + 1) * 10] =\
            positions[:10].copy()
        self.pred_current_trajectory = positions.copy()
        self.traj_idx += 1

        # velocities[0] += self._agent_vel * self._dt
        # positions = positions * 0
        # distances = velocities * self._dt
        # positions[0] = self._agent_pos
        # positions += distances
        # positions = np.cumsum(positions, 0)
        # self.current_trajectory_vel = positions.copy()


    def c2p(self, cart):
        if len(cart.shape) > 1:
            r = np.linalg.norm(cart, axis=-1)
            theta = np.arctan2(cart[:, 1], cart[:, 0])
            return np.array([r, theta]).T
        else:
            r = np.linalg.norm(cart)
            theta = np.arctan2(cart[1], cart[0])
            return np.array([r, theta])


    def p2c(self, pol):
        x = pol[0] * np.cos(pol[1])
        y = pol[0] * np.sin(pol[1])
        return np.array([x, y])


    def set_separating_planes(self):
        for i in range(self.n_crowd):
            pos = self._agent_pos - self._crowd_poss[i]
            vec = pos / np.linalg.norm(pos)
            norm = np.array([-vec[1], vec[0]])
            self.separating_planes[i] = np.concatenate((
                self._crowd_poss[i] + vec * 4 * self.PHYSICAL_SPACE - norm * 50,
                norm * 100
            ))


    @property
    def dt(self) -> Union[float, int]:
        return self._dt


    @property
    def goal_pos(self):
        return self._goal_pos.copy()


    @property
    def current_pos(self):
        return self._agent_pos.copy()


    @property
    def current_vel(self):
        return self._agent_vel.copy()


    @property
    def crowd_pos_vel(self):
        return (self._crowd_poss.copy(), self._crowd_vels.copy())


    @property
    def wall_dist(self):
        return np.array([
            [self.W_BORDER - self._agent_pos[0], self.W_BORDER + self._agent_pos[0]],
            [self.H_BORDER - self._agent_pos[1], self.H_BORDER + self._agent_pos[1]]
        ]).flatten()


    @property
    def optimal_time(self):
        dist = np.linalg.norm(self._goal_pos - self._agent_pos)
        agent_vel = np.linalg.norm(self._agent_vel)
        time_to_max_vel = (self.AGENT_MAX_VEL - agent_vel) / self.MAX_ACC
        time_to_stop = agent_vel / self.MAX_ACC
        dist_to_max_acc = agent_vel * time_to_max_vel +\
            0.5 * self.MAX_ACC * time_to_max_vel ** 2
        dist_to_stop = agent_vel * time_to_stop - 0.5 * self.MAX_ACC * time_to_stop ** 2

        if dist_to_stop >= dist:
            return time_to_stop
        elif dist_to_max_acc + self.MAX_STOPPING_DIST > dist:
            # dx = t_acc * v0 + 0.5 * a * t_acc^2 + a * t_acc * t_dec - 0.5 * a * t_dec^2
            # 0 = v0 + a * t_acc - a * t_dec
            # replace in eq 1 t_dec with t_acc + v0 / a
            a = self.MAX_ACC
            b = 2 * agent_vel
            c = 0.5 * agent_vel ** 2 / self.MAX_ACC - dist
            if a == 0:
                t_acc = - c / b
            else:
                disc = (b ** 2) - (4 * a * c)
                t_acc = (-b + disc ** 0.5) / (2 * a)
            t_dec = t_acc + agent_vel / self.MAX_ACC
            return t_acc + t_dec
        else:
            # dx = t_acc * v0 + 0.5 * a * t_acc^2 +
            #      v_max * t_const +
            #      v_max * t_dec - 0.5 * a * t_dec^2
            t_acc = (self.AGENT_MAX_VEL - agent_vel) / self.MAX_ACC
            t_dec = self.AGENT_MAX_VEL / self.MAX_ACC
            t_const = (
                dist - t_acc * agent_vel - 0.5 * self.MAX_ACC * t_acc ** 2 -
                self.AGENT_MAX_VEL * t_dec + 0.5 * self.MAX_ACC * t_dec ** 2
            ) / self.AGENT_MAX_VEL
            return t_acc + t_dec + t_const


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
        self.traj_idx = 0
        self.exec_traj = []
        self._goal_reached = False
        self._is_collided = False
        self._current_reward = 0
        self.traj_pos = []
        return self._get_obs().copy(), {}


    def _start_env_vars(self):
        """
        Start positions for agent, goal and crowd in the 2D environment. The agent if
        initialized at the center with zero velocity.

        The goal is initialized inside the bounds with padding so the agent does not have
        to crash into the wall. The positions is generated using polar coordinates in
        order to define a minimal distance from the agent. This distance directly affects
        the probability for a member of the crowd to spawn between the agent and the goal.

        In order for the positions of each member of the crowd to be viable it should have
        a dist of at least (PERSONAL_SPACE + PHYSICAL_SPACE) from the agent and at least
        SOCIAL_SPACE to the goal. In order to encourage spawning of a crowd member between
        the agent and the goal, the property is hard coded. The first member of the crowd
        spawned will be placed exactly between the agent and the goal with some uniform
        noise of dimension PERSONAL_SPACE. With the parameter INTERCEPTOR_PERCENTAGE it is
        possible to define the size of the area perpendicular to the semgment connecting
        the agent and the goal. E.g

                       ┌─────────┐
                       │         │
                       │         │
        (agent)O       │    •    │       x(goal)
                       │         │
                       │         │
                       └─────────┘
                       <-PERSONAL>
                       <--SPACE-->

        The rectangle above represents the area from which unifrom sampling happens to
        find position between the agent and the goal. The random sample is rotated based
        on the segment connecting the agent and the goal in order for the sampling are to
        remain in the correct orientation. This sampling process is carried out only for
        the first member sampled while other members are sampled randomly inside the
        bounds. The sampled members of the crowd are shuffled in the end in order for the
        interceptor to be a random index in the list of members.

        The size of the environment and the initial minial goal position (apart from other
        constants set in the environment) directly affect the probability of spawning a
        member of the crowd between the agent and the goal (with some noise in its
        position as described above).
        """
        global seed, flip
        # if seed > 1:
        #     if flip:
        #         seed -= 1
        #     flip = not flip
        np.random.seed(seed)
        seed += 1
        if type(self).__name__ == "CrowdNavigationEnv" and self.const_vel:
            if self.one_way:
                agent_pos = np.array([-self.W_BORDER + self.PHYSICAL_SPACE * 2, 0])
            else:
                agent_pos = np.zeros(2)
        else:
            agent_pos = np.random.uniform(
                [-self.W_BORDER + self.PHYSICAL_SPACE * 1.2,
                 -self.H_BORDER + self.PHYSICAL_SPACE * 1.2],
                [self.W_BORDER - self.PHYSICAL_SPACE * 1.2,
                 self.H_BORDER - self.PHYSICAL_SPACE * 1.2]
            )
        agent_vel = np.zeros(2)
        if type(self).__name__ == "CrowdNavigationEnv" and self.const_vel and\
            self.one_way:
            goal_pos = np.random.uniform(
                [self.W_BORDER / 2,
                 -self.H_BORDER + self.PHYSICAL_SPACE],
                [self.W_BORDER - self.PHYSICAL_SPACE,
                 self.H_BORDER - self.PHYSICAL_SPACE]
            )
        else:
            goal_pos = agent_pos
            while np.linalg.norm(agent_pos - goal_pos) < 2 * self.PERSONAL_SPACE:
                goal_pos = np.random.uniform(
                    [-self.W_BORDER + self.PHYSICAL_SPACE,
                     -self.H_BORDER + self.PHYSICAL_SPACE],
                    [self.W_BORDER - self.PHYSICAL_SPACE,
                     self.H_BORDER - self.PHYSICAL_SPACE]
                )

        crowd_poss = np.zeros((self.n_crowd, 2))
        try_between = True
        for i in range(self.n_crowd):
            while True:
                if try_between:
                    direction = goal_pos - agent_pos
                    rot_deg = np.sign(direction[1]) *\
                        np.arccos(direction[0] / np.linalg.norm(direction))
                    # start from a sample between [-0.5, 0.5] and scale to
                    # [-PHYSICAL_SPACE / 2, INTERCEPTOR_PERCENTAGE * PHYSICAL_SPACE / 2]
                    rand = (np.random.rand(2) - 0.5) * self.PERSONAL_SPACE
                    rand[-1] *= self.INTERCEPTOR_PERCENTAGE
                    sampled_pos = (agent_pos + direction / 2) +\
                        self.rot_mat(rot_deg) @ rand
                    try_between = False
                else:
                    sampled_pos = np.random.uniform(
                        [-self.W_BORDER + self.PHYSICAL_SPACE,
                         -self.H_BORDER + self.PHYSICAL_SPACE],
                        [self.W_BORDER - self.PHYSICAL_SPACE,
                         self.H_BORDER - self.PHYSICAL_SPACE]
                    )
                no_crowd_collision = self.allow_collision or i == 0
                if not self.allow_collision and i > 0:
                    no_crowd_collision = np.sum(np.linalg.norm(  # at least one collision
                        crowd_poss[:i] - sampled_pos, axis=-1
                    ) < self.PERSONAL_SPACE * 2) == 0
                if (np.linalg.norm(sampled_pos - agent_pos) > self.MIN_CROWD_DIST and
                        np.linalg.norm(sampled_pos - goal_pos) > self.SOCIAL_SPACE and
                        no_crowd_collision):
                    crowd_poss[i] = sampled_pos
                    break

        # Shuffle crowd positions so interceptor is at random position
        np.random.shuffle(crowd_poss)

        return agent_pos, agent_vel, goal_pos, crowd_poss, np.zeros(crowd_poss.shape)


    def update_state(self, action):
        """
        Update robot position and velocity for time self._dt based on its dynamics.

        Args:
            action (numpy.ndarray): 2D array representing the acc for current step
        """
        if self.discrete_action:
            if self.velocity_control:
                action = np.array([
                    self.CARTESIAN_VEL[action[0]], self.CARTESIAN_VEL[action[1]]
                ])
            else:
                action = np.array([
                    self.CARTESIAN_ACC[action[0]], self.CARTESIAN_ACC[action[1]]
                ])

        if self.velocity_control:
            vel = self.p2c(action) if self.polar else action
            acc = (vel - self._agent_vel) / self._dt
            acc_norm = np.linalg.norm(acc)
            if acc_norm > self.MAX_ACC:
                vel = self._agent_vel + acc / acc_norm * self.MAX_ACC * self._dt
            vel_norm = np.linalg.norm(vel)
            if vel_norm > self.AGENT_MAX_VEL:
                vel *= self.AGENT_MAX_VEL / vel_norm

            self._agent_pos += (self._agent_vel + vel) * self._dt / 2
            self._agent_vel = vel
        else:
            acc = action
            acc_norm = np.linalg.norm(acc)
            if acc_norm > self.MAX_ACC:
                acc *= self.MAX_ACC / acc_norm

            self._agent_pos += self._agent_vel * self._dt + acc * 0.5 * self._dt ** 2
            self._agent_vel += acc * self._dt

            agent_speed = np.linalg.norm(self._agent_vel)
            if agent_speed > self.AGENT_MAX_VEL:
                self._agent_vel *= self.AGENT_MAX_VEL / agent_speed

        # check bounds of the environment and the bounds of the maximum velocity
        self._agent_pos = np.clip(
            self._agent_pos,
            [-self.W_BORDER, -self.H_BORDER],
            [self.W_BORDER, self.H_BORDER]
        )


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
