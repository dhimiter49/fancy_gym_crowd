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
        self._traj_len = self.replan
        self._safety_traj = 2
        self._plan_traj = 7
        self.n_crowd = n_crowd
        self.var_radius = var_radius
        self._reset_steps = 0
        self.max_n_crowd = self.n_crowd
        self.current_seed = 0
        self.flip = True
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
        self.MAX_ACC = 10.0 if not self.one_way else 1.
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
        self._last_crowd_poss = self._crowd_poss
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
        self.traj_idx = 0
        self.current_trajectory = np.zeros((2 * self.MAX_EPISODE_STEPS, 2))
        self.motions = []
        self.casc_trajectory = np.zeros((self._safety_traj * self._plan_traj, 2))
        self.pred_current_trajectory = np.zeros((100, 2))
        self.exec_traj = []
        self.exec_actions = []
        self.idx_colliding_agents = []
        self.desired_position = None  # desired position when using ProDMP
        self.current_trajectory_vel = np.zeros((self.MAX_EPISODE_STEPS, 2))
        self._traj_index = 0
        self.separating_planes = np.zeros((self.max_n_crowd, 4))

        self.num_env_col = 0  # at leat one collision in environment
        self.num_col = 0  # every collision in the environment
        self.col_vel_sum = 0.
        self.col_agent_vel_sum = 0.
        self.col_inters_sum = 0.
        self.col_severity_index = 0.
        self.all_ttg = []
        self.froze_last = False
        self.freezing_instances = 0
        self.zero_vel_instances = 0
        self.oscillating_instances = 0
        self.far_from_goal_instances = 0


    def hard_set_vars(self, vars):
        """
        Hard set variables that define the whole state of the environment.

        Args:
            action (dict): dictionary of variable names and the value to assign
        """
        for key in vars:
            setattr(self, key, vars[key])


    def set_num_crowd(self, n_crowd: int):
        self.n_crowd = n_crowd
        self.max_n_crowd = n_crowd


    def set_wxh(self, width: float, height: float):
        self.WIDTH = width
        self.HEIGHT = height
        self.W_BORDER = width / 2
        self.H_BORDER = height / 2


    def set_trajectory(self, positions, velocities=None):
        # self._traj_index = 0
        positions = positions[:]
        # velocities = velocities[:self._traj_len]

        positions[:] -= positions[0, 0]
        positions[:] += self._agent_pos + self._agent_vel * self._dt
        # self.current_trajectory[
        #     self.traj_idx * self._traj_len:(self.traj_idx + 1) * self._traj_len
        # ] = positions[:self._traj_len].copy()
        self.pred_current_trajectory = positions.copy()
        self.traj_idx += 1

        # velocities[0] += self._agent_vel * self._dt
        # positions = positions * 0
        # distances = velocities * self._dt
        # positions[0] = self._agent_pos
        # positions += distances
        # positions = np.cumsum(positions, 0)
        # self.current_trajectory_vel = positions.copy()

    def set_all_motions(self, positions):
        self.motions = [self.current_pos + p for p in positions]


    def set_casc_trajectory(self, positions):
        self.casc_trajectory = positions + self._agent_pos
        self.current_trajectory = self.casc_trajectory[np.arange(
            0, self._plan_traj * self._safety_traj, self._safety_traj
        )]


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
        self.traj_idx = 0
        self.exec_traj = [self._agent_pos]
        self.exec_actions = []
        self._goal_reached = False
        self._is_collided = False
        self._current_reward = 0
        self.traj_pos = []
        self.froze_last = False
        return self._get_obs().copy(), {}


    def _read_test_case(self):
        # if self._test_case_idx > 1:
        #     if self.flip:
        #         self._test_case_idx -= 1
        #     self.flip = not self.flip
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
        # if self.current_seed > 1:
        #     if self.flip:
        #         self.current_seed -= 1
        #     self.flip = not self.flip
        np.random.seed(self.current_seed)
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
                [self.W_BORDER - 3 * self.PHYSICAL_SPACE[0], self.H_BORDER * 0.5]
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


    def collision_metrics(self):
        self.num_env_col += 1
        # print("Seed", self.current_seed)
        # print("Num col", self.num_env_col)
        if len(self.idx_colliding_agents) >= 1:
            self.num_col += len(self.idx_colliding_agents)
            col_speed = np.sum(np.linalg.norm(
                self._agent_vel - self._crowd_vels[self.idx_colliding_agents], axis=-1
            ))
            self.col_vel_sum += col_speed
            self.col_agent_vel_sum += np.linalg.norm(self._agent_vel)
            # print("Col vel", self.col_vel_sum / self.num_col)
            # print("Col agent vel", self.col_agent_vel_sum / self.num_col)

            # Find intersection, first find maximum intersection distance between circles
            # Then find the intersection area
            #   Visual representation of collision intersection
            #   c0 x------(--|--)------x c1
            #       <--------d-------->
            #       <---a--->
            #                 <---b--->
            #
            idx_colliding_agent = self.idx_colliding_agents[0]
            r_0 = self.PHYSICAL_SPACE[0]
            r_1 = self.PHYSICAL_SPACE[1 + idx_colliding_agent]
            if np.linalg.norm(self._agent_pos - self._crowd_poss[idx_colliding_agent]) >\
                (self.PHYSICAL_SPACE[0] + self.PHYSICAL_SPACE[idx_colliding_agent + 1]):
                # closest point already reached between the agent and crowd
                # assume maximum intersection for simplicitly
                d = np.sqrt(
                    (r_0 + r_1)**2 - ((
                        self.CROWD_MAX_VEL - (self.CROWD_MAX_VEL - self.AGENT_MAX_VEL) / 2
                    ) * self._dt)**2
                )
            else:
                col_poss = np.stack([
                    self._agent_pos, self._crowd_poss[idx_colliding_agent]
                ])
                col_vels = np.stack([
                    self._agent_vel, self._crowd_vels[idx_colliding_agent]
                ])
                max_time = 5.  # in case that collision velocity is almost zero
                if np.linalg.norm(col_vels[1]) > 1e-4:
                    max_time = (self.PHYSICAL_SPACE[0] * 3) / np.linalg.norm(col_vels[1])
                sample_dt = 0.01
                max_time_steps = int(max_time // sample_dt) * 2
                propagate_col_agents = np.repeat(
                    np.expand_dims(col_poss, axis=0), max_time_steps, axis=0
                ) + np.einsum(
                    "ijk,i->ijk",
                    np.repeat(
                        np.expand_dims(col_vels, axis=0), max_time_steps, axis=0
                    ),
                    np.arange(-max_time_steps // 2, max_time_steps // 2) * sample_dt
                )
                d = np.min(np.linalg.norm(
                    propagate_col_agents[:, 0] - propagate_col_agents[:, 1], axis=-1
                ))
            a = (r_0**2 - r_1**2 + d**2) / (2 * d)
            h = np.sqrt(r_0**2 - a**2)
            alpha = np.arccos(a / r_0)
            arch_area = r_0**2 * alpha
            triangle_area = h * a
            c_0_intersection_area = arch_area - triangle_area
            if r_0 == r_1:
                c_1_intersection_area = c_0_intersection_area
            else:
                b = np.sqrt(r_1**2 - h**2)
                beta = np.arccos(b / r_1)
                arch_area = r_1**2 * beta
                triangle_area = h * b
                c_1_intersection_area = arch_area - triangle_area
            intersection_area = c_0_intersection_area + c_1_intersection_area
            self.col_inters_sum += intersection_area
            intersection_area_percent = intersection_area / (np.pi * r_0 ** 2)
            self.col_severity_index += intersection_area_percent * col_speed /\
                (self.AGENT_MAX_VEL + self.CROWD_MAX_VEL)
            # print(
            #     "Col avg max intersection area: ",
            #     self.col_inters_sum / self.num_env_col
            # )
            # print(
            #     "Col avg, max intersection area rel to agent size:",
            #     round(
            #         (self.col_inters_sum / self.num_env_col) / (np.pi * r_0 ** 2) * 100,
            #         2
            #     ),
            #     "%"
            # )


    def freezing_agent(self):
        num_last_steps = max(int(1. // self._dt), 5)
        if len(self.exec_traj) < num_last_steps:
            return False, dict(freezing=False, oscillating=False, far_from_goal=False)

        exec_traj = np.array(self.exec_traj)
        exec_actions = np.array(self.exec_actions)
        # Check if robot is frozen and not moving
        freezing = np.all(np.linalg.norm(exec_actions[-num_last_steps:], axis=-1) < 0.1)

        # Oscillating
        oscillating = np.all(
            np.linalg.norm(
                exec_traj[-num_last_steps] - exec_traj[-num_last_steps + 1:], axis=-1
            ) < self.PHYSICAL_SPACE[0]
        )

        # Getting further from goal
        far_from_goal = np.linalg.norm(
            self._goal_pos - exec_traj[-num_last_steps]
        ) < np.linalg.norm(
            self.goal_pos - exec_traj[-1]
        )
        frp = freezing or oscillating or far_from_goal
        self.zero_vel_instances += 1 if freezing else 0
        self.oscillating_instances += 1 if oscillating else 0
        self.far_from_goal_instances += 1 if far_from_goal else 0
        # if freezing:
        #     print("Zero vel", self.zero_vel_instances)
        #     input()
        # if oscillating:
        #     print("oscillating", self.oscillating_instances)
        #     input()
        # if far_from_goal:
        #     print("far from goal", self.far_from_goal_instances)
        #     input()
        if frp and not self.froze_last:
            self.froze_last = True
            self.freezing_instances += 1
            # print("Freezing instances:", self.freezing_instances)
        else:
            self.froze_last = False
        return frp, \
            dict(freezing=freezing, oscillating=oscillating, far_from_goal=far_from_goal)


    def ttg(self):
        self.all_ttg.append(self._steps * self._dt)
        # print("Avg ttg:", np.mean(self.all_ttg))


    def stats(self):
        # print("Num col", self.num_env_col)
        # if self.num_col > 0:
        #     print("Col vel", self.col_vel_sum / self.num_col)
        #     print("Col agent vel", self.col_agent_vel_sum / self.num_col)
        #     print(
        #         "Col avg max intersection area: ",
        #         self.col_inters_sum / self.num_env_col
        #     )
        #     print(
        #         "Col avg, max intersection area rel to agent size:",
        #         round(
        #             (self.col_inters_sum / self.num_env_col) /
        #             (np.pi * self.PHYSICAL_SPACE[0] ** 2) * 100,
        #             2
        #         ),
        #         "%"
        #     )
        print("Freezing instances:", self.freezing_instances)
        print("Zero vel instances:", self.zero_vel_instances)
        print("Oscillating instances:", self.oscillating_instances)
        print("Far from goal instances:", self.far_from_goal_instances)
        # print("Avg ttg:", np.mean(self.all_ttg))
        # print("Success instances:", len(self.all_ttg))
        if self.num_col == 0:
            return (
                0, 0, 0, 0, 0, 0,
                self.freezing_instances,
                np.mean(self.all_ttg),
                len(self.all_ttg)
            )
        return (
            self.num_env_col,
            self.col_vel_sum / self.num_col,
            self.col_agent_vel_sum / self.num_col,
            self.col_inters_sum / self.num_env_col,
            round(
                (self.col_inters_sum / self.num_env_col) /
                (np.pi * self.PHYSICAL_SPACE[0] ** 2) * 100,
                2
            ),
            self.col_severity_index / self.num_env_col,
            self.freezing_instances,
            np.mean(self.all_ttg),
            len(self.all_ttg)
        )
