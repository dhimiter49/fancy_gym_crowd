from typing import Union, Tuple, Optional, Any, Dict
import inspect

import gymnasium as gym
import numpy as np
import pickle
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
        discrete_action: bool = False,
        velocity_control: bool = False,
        dt: float = 0.1,
        continuous_collision: bool = True,
        var_radius: bool = False,
        test_case: str = "",
    ):
        self.non_polar_action = False
        calling_frames = inspect.getouterframes(inspect.currentframe())[1:]
        for c_frame in calling_frames:
            if "fancy_gym/envs/registry.py" in c_frame.filename:
                self.non_polar_action = True
                break
        super().__init__()

        self._dt = dt
        self.n_crowd = n_crowd
        self.var_radius = var_radius
        self._reset_steps = 0
        self.max_n_crowd = self.n_crowd
        self.run_test_case = test_case != ""
        if self.run_test_case:
            self._test_case_idx = -1  # -1 is not executed it is just test run
            current_dir = __file__.split('/')[:-1]
            with open("/".join(current_dir) + "/" + test_case, "rb") as f:
                self._test_case_array = np.array(
                    pickle.load(f, encoding="latin1"), dtype=np.float64
                )
            self.n_crowd = self.max_n_crowd = self._test_case_array.shape[1] - 1
            if "Inter" in type(self).__name__:
                self.n_crowd += 1
                self.max_n_crowd += 1
        self.current_seed = -1

        self.WIDTH = width
        self.HEIGHT = height
        self.W_BORDER = self.WIDTH / 2
        self.H_BORDER = self.HEIGHT / 2
        self.AGENT_MAX_VEL = 1.0
        self.CROWD_MAX_VEL = 1.5
        # 0 -> agent radius, and then other members of the crowd
        self.MIN_RADIUS = 0.2
        self.MAX_RADIUS = 1.
        if self.var_radius:
            self.PHYSICAL_SPACE = np.random.uniform(
                self.MIN_RADIUS, self.MAX_RADIUS, self.max_n_crowd + 1
            )
        else:
            self.PHYSICAL_SPACE = np.array([0.4] * (self.max_n_crowd + 1))
        self.PERSONAL_SPACE = self.PHYSICAL_SPACE + 1.
        self.SOCIAL_SPACE = self.PHYSICAL_SPACE + 1.5
        self.MAX_ACC = 10.0
        if self.one_way:
            if n_crowd == 20:
                self.MAX_ACC = 1.0
            else:
                self.MAX_ACC = 2.0
        self.MAX_STOPPING_TIME = self.AGENT_MAX_VEL / self.MAX_ACC
        self.MAX_STOPPING_TIME_CROWD = self.CROWD_MAX_VEL / self.MAX_ACC
        self.MAX_STOPPING_DIST = self.AGENT_MAX_VEL * self.MAX_STOPPING_TIME -\
            0.5 * self.MAX_ACC * self.MAX_STOPPING_TIME ** 2
        self.MAX_STOPPING_DIST_CROWD = self.CROWD_MAX_VEL *\
            self.MAX_STOPPING_TIME_CROWD - 0.5 * self.MAX_ACC *\
            self.MAX_STOPPING_TIME_CROWD ** 2
        self.INTERCEPTOR_PERCENTAGE = interceptor_percentage
        self.MIN_SPAWN_DIST = np.max([
            2 * self.CROWD_MAX_VEL,
            float(np.max(self.PERSONAL_SPACE + self.PHYSICAL_SPACE))
        ])


        self.COLLISION_REWARD = -10
        self.Cc = (self.MIN_RADIUS + self.MAX_RADIUS) *\
            np.log(-self.COLLISION_REWARD / self.MAX_EPISODE_STEPS + 1)
        self.Cg = -self.COLLISION_REWARD / (self.AGENT_MAX_VEL * self._dt) ** 2 /\
            self.MAX_EPISODE_STEPS
        self.Tc = -self.COLLISION_REWARD
        self.Ci = -2. if hasattr(self, "const_vel") and self.const_vel else -5.
        self.Cc *= 2

        self.allow_collision = allow_collision
        self.supersample_col = continuous_collision
        self.rot_mat = lambda deg: np.array([
            [np.cos(deg), -np.sin(deg)], [np.sin(deg), np.cos(deg)]
        ])
        if self.run_test_case:
            (
                self._agent_pos,
                self._agent_vel,
                self._goal_pos,
                self._crowd_poss,
                self._crowd_vels
            ) = self._read_test_case()
        else:
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
            elif self.polar and not self.non_polar_action:
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
            [-self.WIDTH, -self.HEIGHT] * (self.max_n_crowd + 1),
            [0] * (self.max_n_crowd + 1),
        ])
        state_bound_max = np.hstack([
            [self.WIDTH, self.HEIGHT] * (self.max_n_crowd + 1),
            [self.AGENT_MAX_VEL],
            [self.CROWD_MAX_VEL] * (self.max_n_crowd)
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
            np.linalg.norm(self._agent_pos - self._goal_pos) <
            self.PHYSICAL_SPACE[0] / 2 and
            np.linalg.norm(self._agent_vel) < self.MAX_ACC * self._dt
        )
        self.exec_traj = []
        self.exec_actions = []
        self.idx_colliding_agents = []
        self.desired_position = None
        self.current_trajectory = np.zeros((self.MAX_EPISODE_STEPS, 2))
        self.current_trajectory_vel = np.zeros((self.MAX_EPISODE_STEPS, 2))
        self.separating_planes = np.zeros((self.max_n_crowd, 4))

        self.num_env_col = 0  # at leat one collision in environment
        self.num_col = 0  # every collision in the environment
        self.col_vel_sum = 0.
        self.col_agent_vel_sum = 0.
        self.col_inters_sum = 0.
        self.all_ttg = []
        self.froze_last = False
        self.freezing_instances = 0


    def hard_set_vars(self, vars):
        """
        Hard set variables that define the whole state of the environment.

        Args:
            action (dict): dictionary of variable names and the value to assign
        """
        for key in vars:
            setattr(self, key, vars[key])


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


    def set_des_position(self, position):
        """
        Set the next desired position from the ProDMP, relevant to calculate the intrinsic
        reward when using MPC. The learned agent must predict (feasible) trajectories that
        MPC can follow.
        """
        self.desired_position = position


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
        if len(pol.shape) > 1:
            x = pol[:, 0] * np.cos(pol[:, 1])
            y = pol[:, 0] * np.sin(pol[:, 1])
            return np.array([x, y]).T
        else:
            x = pol[0] * np.cos(pol[1])
            y = pol[0] * np.sin(pol[1])
            return np.array([x, y])


    def set_separating_planes(self):
        for i in range(self.n_crowd):
            pos = self._agent_pos - self._crowd_poss[i]
            vec = pos / np.linalg.norm(pos)
            norm = np.array([-vec[1], vec[0]])
            self.separating_planes[i] = np.concatenate((
                self._crowd_poss[i] + vec * self.PHYSICAL_SPACE[i] - norm * 50,
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
        if self.run_test_case:
            self._test_case_idx += 1
            (
                self._agent_pos,
                self._agent_vel,
                self._goal_pos,
                self._crowd_poss,
                self._crowd_vels
            ) = self._read_test_case()
        else:
            (
                self._agent_pos,
                self._agent_vel,
                self._goal_pos,
                self._crowd_poss,
                self._crowd_vels
            ) = self._start_env_vars()
        self._reset_steps += 1
        self._steps = 0
        self.exec_traj = [self._agent_pos]
        self.exec_actions = []
        self._goal_reached = False
        self._is_collided = False
        self._current_reward = 0
        self.froze_last = False
        return self._get_obs().copy(), {}


    def _read_test_case(self):
        if "Inter" not in type(self).__name__:
            agent_pos = self._test_case_array[self._test_case_idx, 0, :2]
            goal_pos = self._test_case_array[self._test_case_idx, 0, 2:4]
            crowd_poss = self._test_case_array[self._test_case_idx, 1:, :2]
        else:
            agent_pos = np.zeros(2)
            goal_pos = np.zeros(2)
            crowd_poss = self._test_case_array[self._test_case_idx, 0:, :2]
        return agent_pos, 0 * agent_pos, goal_pos, crowd_poss, 0 * crowd_poss


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
        self.current_seed += 1
        if self.var_radius:
            self.PHYSICAL_SPACE = np.random.uniform(
                self.MIN_RADIUS, self.MAX_RADIUS, self.n_crowd + 1
            )
            self.PERSONAL_SPACE = self.PHYSICAL_SPACE + 1.
            self.SOCIAL_SPACE = self.PHYSICAL_SPACE + 1.5
            self.MIN_SPAWN_DIST = np.max([
                2 * self.CROWD_MAX_VEL,
                float(np.max(self.PERSONAL_SPACE + self.PHYSICAL_SPACE))
            ])
        if type(self).__name__ == "CrowdNavigationEnv" and self.const_vel:
            if self.one_way:
                agent_pos = np.array([-self.W_BORDER + self.PHYSICAL_SPACE[0] * 2, 0])
            else:
                agent_pos = np.zeros(2)
        else:
            agent_pos = np.random.uniform(
                [-self.W_BORDER + self.PHYSICAL_SPACE[0] * 1.2,
                 -self.H_BORDER + self.PHYSICAL_SPACE[0] * 1.2],
                [self.W_BORDER - self.PHYSICAL_SPACE[0] * 1.2,
                 self.H_BORDER - self.PHYSICAL_SPACE[0] * 1.2]
            )
        agent_vel = np.zeros(2)
        if type(self).__name__ == "CrowdNavigationEnv" and self.const_vel and\
            self.one_way:
            goal_pos = np.random.uniform(
                [self.W_BORDER / 2, -self.H_BORDER * 0.5],
                [self.W_BORDER - 4 * self.PHYSICAL_SPACE[0], self.H_BORDER * 0.5]
            )
        else:
            goal_pos = agent_pos
            while np.linalg.norm(agent_pos - goal_pos) < 2 * self.PERSONAL_SPACE[0]:
                goal_pos = np.random.uniform(
                    [-self.W_BORDER + self.PHYSICAL_SPACE[0],
                     -self.H_BORDER + self.PHYSICAL_SPACE[0]],
                    [self.W_BORDER - self.PHYSICAL_SPACE[0],
                     self.H_BORDER - self.PHYSICAL_SPACE[0]]
                )

        crowd_poss = np.zeros((self.n_crowd, 2))
        try_between = (
            "Inter" not in type(self).__name__ and not self.one_way
        )  # no need in case of inter crowd
        tries = 0
        for i in range(self.n_crowd):
            while True:
                tries += 1
                if try_between:
                    direction = goal_pos - agent_pos
                    rot_deg = np.sign(direction[1]) *\
                        np.arccos(direction[0] / np.linalg.norm(direction))
                    # start from a sample between [-0.5, 0.5] and scale to
                    # [-PHYSICAL_SPACE / 2, INTERCEPTOR_PERCENTAGE * PHYSICAL_SPACE / 2]
                    rand = (np.random.rand(2) - 0.5) * self.PERSONAL_SPACE[i]
                    rand[-1] *= self.INTERCEPTOR_PERCENTAGE
                    sampled_pos = (agent_pos + direction / 2) +\
                        self.rot_mat(rot_deg) @ rand
                    try_between = False
                else:
                    if self.one_way:
                        sampled_pos = np.random.uniform(
                            [-self.W_BORDER + self.PHYSICAL_SPACE[i + 1] * 2.8,
                             -self.H_BORDER + self.PHYSICAL_SPACE[i + 1] * 1.5],
                            [self.W_BORDER * 2,
                             self.H_BORDER - self.PHYSICAL_SPACE[i + 1] * 1.5]
                        )
                    else:
                        sampled_pos = np.random.uniform(
                            [-self.W_BORDER + self.PHYSICAL_SPACE[i + 1] * 1.2,
                             -self.H_BORDER + self.PHYSICAL_SPACE[i + 1] * 1.2],
                            [self.W_BORDER - self.PHYSICAL_SPACE[i + 1] * 1.2,
                             self.H_BORDER - self.PHYSICAL_SPACE[i + 1] * 1.2]
                        )
                no_crowd_collision = self.allow_collision or i == 0 or tries > 30
                if not self.allow_collision and i > 0 and not tries > 30:
                    no_crowd_collision = np.sum(np.linalg.norm(  # at least one collision
                        crowd_poss[:i] - sampled_pos, axis=-1
                    ) < self.PERSONAL_SPACE[:i] + self.PERSONAL_SPACE[i]) == 0
                if (
                    np.linalg.norm(sampled_pos - agent_pos) > self.MIN_SPAWN_DIST and
                    (np.linalg.norm(sampled_pos - goal_pos) > self.SOCIAL_SPACE[i] or
                     self.one_way) and
                    no_crowd_collision
                ):
                    crowd_poss[i] = sampled_pos
                    break

        if "Inter" not in type(self).__name__:
            # Shuffle crowd positions so interceptor is at random position
            idxs = np.arange(self.n_crowd)
            np.random.shuffle(idxs)
            crowd_poss = crowd_poss[idxs]
            self.PHYSICAL_SPACE[1:1 + self.n_crowd] =\
                self.PHYSICAL_SPACE[1:1 + self.n_crowd][idxs]

        return agent_pos, agent_vel, goal_pos, crowd_poss, np.zeros(crowd_poss.shape)


    def update_state(self, action):
        """
        Update robot position and velocity for time self._dt based on its dynamics.

        Args:
            action (numpy.ndarray): 1D (x, y) array representing the acc for current step
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

        self._last_agent_pos = self._agent_pos.copy()
        if self.velocity_control:
            vel = self.p2c(action) if self.polar and not self.non_polar_action else action
            acc = (vel - self._agent_vel) / self._dt
            acc_norm = np.linalg.norm(acc)
            if acc_norm > self.MAX_ACC:
                vel = self._agent_vel + acc / acc_norm * self.MAX_ACC * self._dt
            vel_norm = np.linalg.norm(vel)
            if vel_norm > self.AGENT_MAX_VEL:
                vel *= self.AGENT_MAX_VEL / vel_norm

        else:
            acc = action
            acc_norm = np.linalg.norm(acc)
            if acc_norm > self.MAX_ACC:
                acc *= self.MAX_ACC / acc_norm

            vel = self._agent_vel + acc * self._dt
            agent_speed = np.linalg.norm(vel)
            if agent_speed > self.AGENT_MAX_VEL:
                vel *= self.AGENT_MAX_VEL / agent_speed

        self._agent_pos += (self._agent_vel + vel) * self._dt / 2
        self._agent_vel = vel


        # check bounds of the environment
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
        if self.n_crowd > 0:
            agent_poss = self._agent_pos.copy()
            crowd_poss = self._crowd_poss.copy()
            if self.supersample_col:
                over_sample_by = self._dt / 0.01
                agent_poss = self._last_agent_pos + np.einsum(
                    "i,j->ij",
                    np.arange(0, int(over_sample_by) + 1),
                    self._agent_pos - self._last_agent_pos
                ) / over_sample_by
                crowd_poss = self._last_crowd_poss + np.einsum(
                    "i,kj->ikj",
                    np.arange(0, int(over_sample_by) + 1),
                    self._crowd_poss - self._last_crowd_poss
                ) / over_sample_by
                agent_poss = np.expand_dims(agent_poss, axis=1)
            self.idx_colliding_agents = np.where((
                np.linalg.norm(
                    agent_poss - crowd_poss, axis=-1
                ) < (self.PHYSICAL_SPACE[0] + self.PHYSICAL_SPACE[1:1 + self.n_crowd])
            ) > 0)[-1]
            self.idx_colliding_agents = list(set(list(self.idx_colliding_agents)))
            if len(self.idx_colliding_agents) > 0:
                return True
        # Walls
        if np.sum(np.abs(self._agent_pos) >
           np.array([self.W_BORDER, self.H_BORDER]) - self.PHYSICAL_SPACE[0]):
            return True
        return False


    def _terminate(self, info) -> bool:
        raise NotImplementedError


    def close(self):
        super(BaseCrowdNavigationEnv, self).close()
        del self.fig
