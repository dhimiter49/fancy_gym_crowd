import matplotlib.pyplot as plt
import numpy as np
from gymnasium import spaces
from gymnasium.core import ObsType

from fancy_gym.envs.classic_control.crowd_navigation.base_crowd_navigation\
    import BaseCrowdNavigationEnv


class CrowdNavigationEnv(BaseCrowdNavigationEnv):
    """
    Crowd with linear movement. For each member of the crowd a goal position is sampled.
    Each member of the crowd moves to the goal using basic motion physics based on the
    maximal velocity and maximal acceleration.
    """
    def __init__(
        self,
        n_crowd: int,
        width: int = 20,
        height: int = 20,
        interceptor_percentage: float = 0.5,
        discrete_action: bool = False,
        velocity_control: bool = False,
        lidar_rays: int = 0,
        const_vel: bool = False,
    ):
        self.MAX_EPISODE_STEPS = 100
        self.const_vel = const_vel
        super().__init__(
            n_crowd,
            width,
            height,
            interceptor_percentage,
            allow_collision=False,
            discrete_action=discrete_action,
            velocity_control=velocity_control,
        )

        self.lidar = lidar_rays != 0
        max_dist = np.linalg.norm(np.array([self.WIDTH, self.HEIGHT]))
        if self.lidar:
            self.N_RAYS = lidar_rays
            self._n_frames = 4
            self._last_frames = np.zeros((self._n_frames, self.N_RAYS))
            self.RAY_ANGLES = np.linspace(
                0, 2 * np.pi, self.N_RAYS, endpoint=False
            ) + 1e-6
            self.RAY_COS = np.cos(self.RAY_ANGLES)
            self.RAY_SIN = np.sin(self.RAY_ANGLES)
        if self.lidar:
            state_bound_min = np.hstack([
                [-self.WIDTH, -self.HEIGHT],
                [-self.AGENT_MAX_VEL, -self.AGENT_MAX_VEL],
                [0] * self.N_RAYS * self._n_frames,
            ])
            state_bound_max = np.hstack([
                [self.WIDTH, self.HEIGHT],
                [self.AGENT_MAX_VEL, self.AGENT_MAX_VEL],
                np.full(self.N_RAYS * self._n_frames, max_dist)
            ])
        else:
            state_bound_min = np.hstack([
                [-self.WIDTH, -self.HEIGHT] * (self.n_crowd + 1),
                [-self.AGENT_MAX_VEL, -self.AGENT_MAX_VEL] * (self.n_crowd + 1),
                [0] * 4,  # four directions
            ])
            state_bound_max = np.hstack([
                [self.WIDTH, self.HEIGHT] * (self.n_crowd + 1),
                [self.AGENT_MAX_VEL, self.AGENT_MAX_VEL] * (self.n_crowd + 1),
                np.repeat([self.WIDTH, self.HEIGHT], 2),  # four directions
            ])

        self.observation_space = spaces.Box(
            low=state_bound_min, high=state_bound_max, shape=state_bound_min.shape
        )


    def _get_reward(self, action: np.ndarray):
        dg = np.linalg.norm(self._agent_pos - self._goal_pos)
        if self._goal_reached:
            Rg = self.Tc
        else:
            # Goal distance
            Rg = -self.Cg * dg ** 2

        if self._is_collided:
            Rc = self.COLLISION_REWARD
        else:
            # Crowd distance
            dist_crowd = np.linalg.norm(
                self._agent_pos - self._crowd_poss,
                axis=-1
            )
            Rc = np.sum(
                (1 - np.exp(self.Cc / dist_crowd)) *
                (dist_crowd < [self.SOCIAL_SPACE + self.PHYSICAL_SPACE] * self.n_crowd)
            )

        # Walls, only one of the walls is closer (irrelevant which)
        dist_walls = np.array([
            self.W_BORDER - abs(self._agent_pos[0]),
            self.H_BORDER - abs(self._agent_pos[1]),
        ])
        Rw = np.sum(
            (1 - np.exp(self.Cc / dist_walls)) * (dist_walls < self.PHYSICAL_SPACE * 2)
        )

        reward = Rg + Rc + Rw
        return reward, dict(goal=Rg, collision=Rc, wall=Rw)


    def _terminate(self, info):
        return self._is_collided or self._goal_reached


    def _get_obs(self) -> ObsType:
        if self.lidar:
            wall_distances = np.min([
                (self.W_BORDER - np.where(
                    self.RAY_COS > 0, self._agent_pos[0], -self._agent_pos[0]
                )) / np.abs(self.RAY_COS),
                (self.H_BORDER - np.where(
                    self.RAY_SIN > 0, self._agent_pos[1], -self._agent_pos[1]
                )) / np.abs(self.RAY_SIN)
            ], axis=0)

            x_crowd_rel, y_crowd_rel = self._crowd_poss[:, 0] - self._agent_pos[0], \
                self._crowd_poss[:, 1] - self._agent_pos[1]
            orthog_dist = np.abs(
                np.outer(x_crowd_rel, self.RAY_SIN) - np.outer(y_crowd_rel, self.RAY_COS)
            )
            intersections_mask = orthog_dist <= self.PHYSICAL_SPACE
            along_dist = np.outer(x_crowd_rel, self.RAY_COS) +\
                np.outer(y_crowd_rel, self.RAY_SIN)
            orthog_to_intersect_dist = np.sqrt(np.maximum(
                self.PHYSICAL_SPACE ** 2 - orthog_dist ** 2, 0
            ))
            intersect_distances = np.where(
                intersections_mask, along_dist - orthog_to_intersect_dist, np.inf
            )
            min_intersect_distances = np.min(np.where(
                intersect_distances > 0, intersect_distances, np.inf), axis=0
            )
            ray_distances = np.minimum(min_intersect_distances, wall_distances)
            self.ray_distances = ray_distances

            self._last_frames = np.concatenate([
                np.array([ray_distances]),
                self._last_frames[:self._n_frames - 1]
            ])
            return np.concatenate([
                self._agent_vel,
                self._goal_pos - self._agent_pos,
                self._last_frames.flatten()
            ]).astype(np.float32).flatten()
        else:
            rel_crowd_poss = self._crowd_poss - self._agent_pos
            dist_walls = np.array([
                [self.W_BORDER - self._agent_pos[0], self.W_BORDER + self._agent_pos[0]],
                [self.H_BORDER - self._agent_pos[1], self.H_BORDER + self._agent_pos[1]]
            ])
            return np.concatenate([
                [self._goal_pos - self._agent_pos],
                rel_crowd_poss if self.n_crowd > 1 else [rel_crowd_poss],
                [self._agent_vel],
                self._crowd_vels,
                dist_walls
            ]).astype(np.float32).flatten()


    def _start_env_vars(self):
        agent_pos, agent_vel, goal_pos, crowd_poss, _ = super()._start_env_vars()
        next_crowd_vels = np.zeros(crowd_poss.shape)

        if self.const_vel:
            for i, c in enumerate(crowd_poss):
                if c[0] > 0:
                    idx = np.random.choice([0, 1])
                    if idx == 0:
                        pol_vel = np.random.uniform(
                            [0.5, np.pi * 5 / 6], [self.AGENT_MAX_VEL, np.pi]
                        )
                    else:
                        pol_vel = np.random.uniform(
                            [0.5, -np.pi], [self.AGENT_MAX_VEL, -np.pi * 5 / 6]
                        )
                else:
                    pol_vel = np.random.uniform(
                        [0.5, -np.pi * 1 / 6], [self.AGENT_MAX_VEL, np.pi * 1 / 6]
                    )
                next_crowd_vels[i] = self.p2c(pol_vel)
        else:
            (
                self._crowd_goal_pos, self._planned_crowd_vels, next_crowd_vels
            ) = self._gen_crowd_goal_and_plan(crowd_poss)

        return agent_pos, agent_vel, goal_pos, crowd_poss, next_crowd_vels


    def _gen_crowd_goal_and_plan(self, crowd_poss):
        """
        The velocities of each member are planned by minimizing the motion equations for
        time. Given a maximum acceleration and velocity for the agent the plan consists
        of two options. In case that the goal is further then double the minimal distance
        for accelerating to the maximum velocity then the motion equation is made up of
        three components: acceleration, moving an maximum velcoity and deceleration. In
        the other case when the goal is closer the crowd member does not need to achieve
        the maximum velcoity and the running time is computes from the quation
        x = at^2.

        Args:
            crowd_poss (numpy.ndarray): list of crowd members

        Returns:
            (numpy.ndarray, numpy.ndarray, numpy.ndarray): the goal positions, the plans
                for the velcoity of each member and the next velocity to be applied
        """
        if len(crowd_poss.shape) == 1:
            crowd_poss = np.array([crowd_poss])
        crowd_goal_pos = np.random.uniform(
            [-self.W_BORDER, -self.H_BORDER],
            [self.W_BORDER, self.H_BORDER],
            (len(crowd_poss), 2)
        )
        crowd_vels = []
        next_crowd_vels = np.zeros(crowd_poss.shape)
        max_step_acc = self.MAX_ACC * self._dt
        for i, goal in enumerate(crowd_goal_pos):
            dist = np.linalg.norm(goal - crowd_poss[i])
            if dist > self.MAX_STOPPING_DIST * 2:
                t_max_vel = (dist - self.MAX_STOPPING_DIST * 2) / self.AGENT_MAX_VEL
                acc_vels = np.arange(
                    max_step_acc, self.AGENT_MAX_VEL + 1e-8, max_step_acc
                )
                dec_vels = np.arange(
                    self.AGENT_MAX_VEL - max_step_acc, 0 - 1e-8, -max_step_acc
                )
                vels = np.concatenate([
                    acc_vels,
                    np.full(int(t_max_vel / self._dt), self.AGENT_MAX_VEL),
                    dec_vels
                ])
            else:
                t_acc = np.sqrt(dist / self.MAX_ACC)
                acc_vels = np.arange(
                    max_step_acc, t_acc * self.MAX_ACC, max_step_acc
                )
                dec_vels = np.arange(
                    t_acc * self.MAX_ACC - max_step_acc, 0 - 1e-8, -max_step_acc
                )
                vels = np.concatenate([acc_vels, dec_vels])

            # Fix direction
            direction = (goal - crowd_poss[i]) / dist
            vels = np.outer(vels, direction).reshape(-1, 2)
            crowd_vels.append(vels)
            next_crowd_vels[i] = vels[0]

        return crowd_goal_pos, crowd_vels, next_crowd_vels


    def render(self):
        if self.fig is None:
            # Create base figure once on the beginning. Afterwards only update
            plt.ion()
            self.fig = plt.figure()
            ax = self.fig.add_subplot(1, 1, 1)

            # limits
            ax.set_xlim(-self.W_BORDER - 1, self.W_BORDER + 1)
            ax.set_ylim(-self.H_BORDER - 1, self.H_BORDER + 1)

            # LiDAR
            if self.lidar:
                self.lidar_rays = []
                for angle, distance in zip(self.RAY_ANGLES, self.ray_distances):
                    self.lidar_rays.append(ax.arrow(
                        self._agent_pos[0], self._agent_pos[1],
                        distance * np.cos(angle), distance * np.sin(angle),
                        head_width=0.0,
                        ec=(0.5, 0.5, 0.5, 0.3),
                        linestyle="--"
                    ))

            # Agent and crowd velocity
            self.vel_agent = ax.arrow(
                self._agent_pos[0], self._agent_pos[1],
                self._agent_vel[0], self._agent_vel[1],
                head_width=self.PERSONAL_SPACE / 4,
                overhang=1,
                head_length=0.2,
                ec="g"
            )
            self.vel_crowd = []
            for i in range(self.n_crowd):
                self.vel_crowd.append(ax.arrow(
                    self._crowd_poss[i][0], self._crowd_poss[i][1],
                    self._crowd_vels[i][0], self._crowd_vels[i][1],
                    head_width=self.PERSONAL_SPACE / 4,
                    overhang=1,
                    head_length=0.2,
                    ec="r"
                ))

            self.sep_planes = []
            for i in range(self.n_crowd):
                self.sep_planes.append(ax.arrow(
                    self.separating_planes[i][0], self.separating_planes[i][1],
                    self.separating_planes[i][2], self.separating_planes[i][3],
                    head_width=0.0,
                    ec="r"
                ))

            # Agent
            self.space_agent = plt.Circle(
                self._agent_pos, self.PHYSICAL_SPACE, color="g", alpha=0.5
            )
            ax.add_patch(self.space_agent)

            # Social space, Personal space, Physical space, Crowd goal positions
            self.ScS_crowd = []
            self.PrS_crowd = []
            self.PhS_crowd = []
            self.crowd_goal_points = []
            for m in self._crowd_poss:
                self.ScS_crowd.append(
                    plt.Circle(
                        m, self.SOCIAL_SPACE, color="r", fill=False, linestyle="--"
                    )
                )
                ax.add_patch(self.ScS_crowd[-1])
                self.PrS_crowd.append(
                    plt.Circle(
                        m, self.PERSONAL_SPACE, color="r", fill=False
                    )
                )
                ax.add_patch(self.PrS_crowd[-1])
                self.PhS_crowd.append(
                    plt.Circle(
                        m, self.PHYSICAL_SPACE, color="r", alpha=0.5
                    )
                )
                ax.add_patch(self.PhS_crowd[-1])
            if not self.const_vel:
                for g in self._crowd_goal_pos:
                    self.crowd_goal_points.append(ax.plot(g[0], g[1], 'yx')[0])

            # Goal
            self.goal_point, = ax.plot(self._goal_pos[0], self._goal_pos[1], 'gx')

            # Trajectory
            self.trajectory_line, = ax.plot(
                self.current_trajectory[:, 0],
                self.current_trajectory[:, 1],
                "k",
            )
            self.trajectory_line_vel, = ax.plot(
                self.current_trajectory_vel[:, 0],
                self.current_trajectory_vel[:, 1],
                "b",
            )

            # Walls
            ax.axvspan(self.W_BORDER, self.W_BORDER + 100, hatch='.')
            ax.axvspan(-self.W_BORDER - 100, -self.W_BORDER, hatch='.')
            ax.axhspan(self.H_BORDER, self.H_BORDER + 100, hatch='.')
            ax.axhspan(-self.H_BORDER - 100, -self.H_BORDER, hatch='.')
            ax.set_aspect(1.0)

            # Walls penalization
            border_penalization = self.PHYSICAL_SPACE * 2
            ax.add_patch(plt.Rectangle(
                (
                    -self.W_BORDER + border_penalization,
                    -self.H_BORDER + border_penalization
                ),
                2 * (self.W_BORDER - border_penalization),
                2 * (self.H_BORDER - border_penalization),
                fill=False, linestyle=":", edgecolor="r", linewidth=0.7
            ))

            self.fig.show()

        self.fig.suptitle(f"Iteration: {self._steps}")
        self.fig.gca().set_title(
            f"Reward at this step: {self._current_reward:.4f}",
            fontsize=11,
            fontweight='bold'
        )

        if self._steps == 1:
            self.goal_point.set_data(self._goal_pos[0], self._goal_pos[1])

        self.vel_agent.set_data(
            x=self._agent_pos[0], y=self._agent_pos[1],
            dx=self._agent_vel[0], dy=self._agent_vel[1]
        )
        self.space_agent.center = self._agent_pos
        for i, member in enumerate(self._crowd_poss):
            self.ScS_crowd[i].center = member
            self.PrS_crowd[i].center = member
            self.PhS_crowd[i].center = member
            if not self.const_vel:
                self.crowd_goal_points[i].set_data(
                    self._crowd_goal_pos[i][0], self._crowd_goal_pos[i][1]
                )
        for i in range(self.n_crowd):
            self.vel_crowd[i].set_data(
                x=self._crowd_poss[i][0], y=self._crowd_poss[i][1],
                dx=self._crowd_vels[i][0], dy=self._crowd_vels[i][1]
            )
        if self.lidar:
            for i, (angle, distance) in \
                enumerate(zip(self.RAY_ANGLES, self.ray_distances)):
                self.lidar_rays[i].set_data(
                    x=self._agent_pos[0], y=self._agent_pos[1],
                    dx=distance * np.cos(angle), dy=distance * np.sin(angle)
                )
        self.trajectory_line.set_data(
            self.current_trajectory[:, 0], self.current_trajectory[:, 1]
        )
        self.trajectory_line_vel.set_data(
            self.current_trajectory_vel[:, 0], self.current_trajectory_vel[:, 1]
        )
        for i in range(self.n_crowd):
            self.sep_planes[i].set_data(
                x=self.separating_planes[i][0], y=self.separating_planes[i][1],
                dx=self.separating_planes[i][2], dy=self.separating_planes[i][3]
            )

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


    def step(self, action: np.ndarray):
        """
        A single step with action in angular velocity space
        """
        self.update_state(action)
        self._crowd_poss += self._crowd_vels * self._dt
        if not self.const_vel:
            for i in range(self.n_crowd):
                self._planned_crowd_vels[i] = np.delete(self._planned_crowd_vels[i], 0, 0)
                if len(self._planned_crowd_vels[i]) == 0:
                    self._crowd_goal_pos[i], self._planned_crowd_vels[i], _ = \
                        self._gen_crowd_goal_and_plan(self._crowd_poss[i])
                    self._planned_crowd_vels[i] = self._planned_crowd_vels[i][0]
                self._crowd_vels[i] = self._planned_crowd_vels[i][0]

        self._goal_reached = self.check_goal_reached()
        self._is_collided = self._check_collisions()
        self._current_reward, info = self._get_reward(action)

        self._steps += 1
        terminated = self._terminate(info)
        truncated = False

        return self._get_obs().copy(), self._current_reward, terminated, truncated, info
