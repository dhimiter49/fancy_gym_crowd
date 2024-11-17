from typing import Tuple, Optional, Any, Dict
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate as interp
from gymnasium import spaces
from gymnasium.core import ObsType
import rvo2

from fancy_gym.envs.classic_control.crowd_navigation.base_crowd_navigation\
    import BaseCrowdNavigationEnv


class CrowdNavigationORCAEnv(BaseCrowdNavigationEnv):
    """
    Crowd with linear movement. For each member of the crowd a goal position is sampled.
    Each member of the crowd moves to the goal using basic motion physics based on the
    maximal velocity and maximal acceleration.

    Args:
        lidar_rays: number of lidar rays, if 0 no lidar is used
        const_vel: sets the dynamics to using constant velocity
        polar: polar observation and action space
        time_frame: time from which to sample and stack the last frames of obs
        lidar_vel: use a velocity representation for each direction of the lidar
        n_frames: number of frames to stack for lidar, irrelevant if lidar_vel
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
        sequence_obs: bool = False,
        const_vel: bool = False,
        one_way: bool = False,
        polar: bool = False,
        time_frame: int = 0,
        lidar_vel: bool = False,
        n_frames: int = 4,
    ):
        assert time_frame == 0 or not lidar_vel
        assert not sequence_obs or lidar_rays == 0  # cannot be seq ob and lidar obs
        self.MAX_EPISODE_STEPS = 100
        self.const_vel = const_vel
        self.one_way = one_way
        self.polar = polar
        super().__init__(
            n_crowd,
            width,
            height,
            interceptor_percentage,
            allow_collision=False,
            discrete_action=discrete_action,
            velocity_control=velocity_control,
        )
        self.neighbor_dist = self.PHYSICAL_SPACE * 6 + 0.1
        self.safety_space = self.PHYSICAL_SPACE / 2
        self.time_horizon = self.MAX_STOPPING_TIME * 8
        self.time_horizon_obst = self.MAX_STOPPING_TIME
        self._start_sim()

        self.seq_obs = sequence_obs
        self.lidar = lidar_rays != 0
        max_dist = np.linalg.norm(np.array([self.WIDTH, self.HEIGHT]))
        if self.lidar:
            self.lidar_vel = lidar_vel
            self.N_RAYS = lidar_rays
            self._n_frames = n_frames if not self.lidar_vel else 2  # one for each pos-vel
            self.use_time_frame = time_frame != 0
            self.time_frame = time_frame
            self.frame_steps = int((time_frame * 10) / (self.dt * 10)) \
                if self.use_time_frame else None
            self._last_frames = np.zeros((self._n_frames, self.N_RAYS))
            self._last_second_frames = np.zeros(
                (self.frame_steps, self.N_RAYS)
            ) if self.use_time_frame else None
            self.RAY_ANGLES = np.linspace(
                0, 2 * np.pi, self.N_RAYS, endpoint=False
            ) + 1e-6
            self.RAY_COS = np.cos(self.RAY_ANGLES)
            self.RAY_SIN = np.sin(self.RAY_ANGLES)
        if self.lidar:
            if self.lidar_vel:
                if self.polar:
                    state_bound_min = np.hstack([
                        [0, -np.pi],
                        [0, -np.pi],
                        [0] * self.N_RAYS * 2,
                    ])
                    state_bound_max = np.hstack([
                        [max_dist, np.pi],
                        [self.AGENT_MAX_VEL, np.pi],
                        [max_dist] * self.N_RAYS,
                        [self.CROWD_MAX_VEL] * self.N_RAYS,
                    ])
                else:
                    state_bound_min = np.hstack([
                        [-self.WIDTH, -self.HEIGHT],
                        [-self.AGENT_MAX_VEL, -self.AGENT_MAX_VEL],
                        [0] * self.N_RAYS * 2,
                    ])
                    state_bound_max = np.hstack([
                        [self.WIDTH, self.HEIGHT],
                        [self.AGENT_MAX_VEL, self.AGENT_MAX_VEL],
                        [max_dist] * self.N_RAYS,
                        [self.CROWD_MAX_VEL] * self.N_RAYS,
                    ])
            elif self.polar:
                state_bound_min = np.hstack([
                    [0, -np.pi],
                    [0, -np.pi],
                    [0] * self.N_RAYS * self._n_frames,
                ])
                state_bound_max = np.hstack([
                    [max_dist, np.pi],
                    [self.AGENT_MAX_VEL, np.pi],
                    [max_dist] * self.N_RAYS * self._n_frames,
                ])
            else:
                state_bound_min = np.hstack([
                    [-self.WIDTH, -self.HEIGHT],
                    [-self.AGENT_MAX_VEL, -self.AGENT_MAX_VEL],
                    [0] * self.N_RAYS * self._n_frames,
                ])
                state_bound_max = np.hstack([
                    [self.WIDTH, self.HEIGHT],
                    [self.AGENT_MAX_VEL, self.AGENT_MAX_VEL],
                    [max_dist] * self.N_RAYS * self._n_frames,
                ])
        elif self.seq_obs:
            state_bound_min = np.hstack([
                [-self.WIDTH, -self.HEIGHT, -self.AGENT_MAX_VEL, -self.AGENT_MAX_VEL],
                [-self.WIDTH, -self.HEIGHT, -self.AGENT_MAX_VEL, -self.AGENT_MAX_VEL],
                [-self.WIDTH, -self.HEIGHT, -self.CROWD_MAX_VEL, -self.CROWD_MAX_VEL] *
                self.n_crowd,
            ])
            state_bound_max = np.hstack([
                [self.WIDTH, self.HEIGHT, self.AGENT_MAX_VEL, self.AGENT_MAX_VEL],
                [self.WIDTH, self.HEIGHT, self.AGENT_MAX_VEL, self.AGENT_MAX_VEL],
                [self.WIDTH, self.HEIGHT, self.CROWD_MAX_VEL, self.CROWD_MAX_VEL] *
                self.n_crowd,
            ])
        else:
            state_bound_min = np.hstack([
                [-self.WIDTH, -self.HEIGHT] * (self.n_crowd + 1),
                [-self.AGENT_MAX_VEL, -self.AGENT_MAX_VEL],
                [-self.CROWD_MAX_VEL, -self.CROWD_MAX_VEL] * self.n_crowd,
                [0] * 4,  # four directions
            ])
            state_bound_max = np.hstack([
                [self.WIDTH, self.HEIGHT] * (self.n_crowd + 1),
                [self.AGENT_MAX_VEL, self.AGENT_MAX_VEL],
                [self.CROWD_MAX_VEL, self.CROWD_MAX_VEL] * self.n_crowd,
                np.repeat([self.WIDTH, self.HEIGHT], 2),  # four directions
            ])

        self.observation_space = spaces.Box(
            low=state_bound_min, high=state_bound_max, shape=state_bound_min.shape
        )


    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[ObsType, Dict[str, Any]]:
        if self.lidar:
            self._last_frames *= 0
        obs, info = super().reset(seed=seed, options=options)
        self._start_sim()
        return obs, info


    def _get_reward(self, action: np.ndarray):
        dg = np.linalg.norm(self._agent_pos - self._goal_pos)
        if self._goal_reached:
            Rg = self.Tc
        else:
            # Goal distance
            Rg = -self.Cg * np.clip(dg, 1, np.inf) ** 2

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
            max(self.W_BORDER - abs(self._agent_pos[0]), self.PHYSICAL_SPACE),
            max(self.H_BORDER - abs(self._agent_pos[1]), self.PHYSICAL_SPACE),
        ])
        Rw = np.sum(
            (1 - np.exp(self.Cc / dist_walls)) * (dist_walls < self.PHYSICAL_SPACE * 2)
        )

        reward = Rg + Rc + Rw
        return reward, dict(goal=Rg, collision=Rc, wall=Rw)


    def _terminate(self, info):
        return self._is_collided or self._goal_reached


    def _get_obs(self) -> ObsType:
        rel_goal_pos = self._goal_pos - self._agent_pos
        rel_goal_pos = self.c2p(rel_goal_pos) if self.polar else rel_goal_pos
        agent_vel = self.c2p(self._agent_vel) if self.polar else self._agent_vel
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

            if not self.use_time_frame and not self.lidar_vel:
                if not np.any(self._last_frames):
                    self._last_frames[list(range(len(self._last_frames)))] = \
                        np.array(ray_distances)
                else:
                    self._last_frames[:-1] = self._last_frames[1:]
                    self._last_frames[-1] = ray_distances
            elif self.lidar_vel:
                ray_velocities = np.zeros(ray_distances.shape)
                for i, (member_pos, member_vel) in enumerate(zip(
                    self._crowd_poss, self._crowd_vels
                )):
                    intersection = intersections_mask[i]
                    if np.any(intersection):
                        dir_idxs = np.where(intersection == 1)[0]
                        for dir_idx in dir_idxs:
                            lidar_vec = np.array([
                                self.RAY_COS[dir_idx], self.RAY_SIN[dir_idx]
                            ])
                            if min_intersect_distances[dir_idx] == np.inf or\
                               np.dot(lidar_vec, member_pos - self._agent_pos) < 0:
                                continue  # account for correct direction
                            vel_along_dir = np.dot(lidar_vec, member_vel)
                            if ray_velocities[dir_idx] > 0 and\
                               np.linalg.norm(member_pos - self._agent_pos) >\
                               ray_distances[dir_idx]:  # keep only the closest one
                                continue
                            ray_velocities[dir_idx] = vel_along_dir

                self._last_frames[0] = ray_distances
                self._last_frames[1] = ray_velocities
            else:
                if not np.any(self._last_frames):
                    self._last_second_frames[
                        list(range(len(self._last_second_frames)))
                    ] = np.array([ray_distances])
                else:
                    self._last_second_frames[:-1] = self._last_second_frames[1:]
                    self._last_second_frames[-1] = ray_distances
                for i, ray in enumerate(range(self.N_RAYS)):
                    r_interp = interp.interp1d(
                        np.arange(self.frame_steps), self._last_second_frames[:, i]
                    )
                    self._last_frames[:, i] = r_interp(
                        np.linspace(0, self.frame_steps - 1, self._n_frames)
                    )

            return np.concatenate([
                rel_goal_pos,
                agent_vel,
                self._last_frames.flatten()
            ]).astype(np.float32).flatten()
        elif self.seq_obs:
            return np.concatenate([
                [np.concatenate([self._agent_pos, self._agent_vel])],
                [np.concatenate([self._goal_pos, self._agent_vel * 0])],
                np.concatenate([self._crowd_poss, self._crowd_vels], axis=-1)
            ]).astype(np.float32).flatten()
        else:
            rel_crowd_poss = self._crowd_poss - self._agent_pos
            dist_walls = np.array([
                [self.W_BORDER - self._agent_pos[0], self.W_BORDER + self._agent_pos[0]],
                [self.H_BORDER - self._agent_pos[1], self.H_BORDER + self._agent_pos[1]]
            ])
            return np.concatenate([
                [rel_goal_pos],
                rel_crowd_poss if self.n_crowd > 1 else [rel_crowd_poss],
                [agent_vel],
                self._crowd_vels,
                dist_walls
            ]).astype(np.float32).flatten()


    def _start_sim(self):
        max_neighbors = self.n_crowd
        params = (
            self.neighbor_dist, max_neighbors, self.time_horizon, self.time_horizon_obst
        )
        self.sim = rvo2.PyRVOSimulator(
            self._dt, *params, self.PHYSICAL_SPACE, self.CROWD_MAX_VEL
        )
        self.sim.addAgent(
            tuple(self._agent_pos),
            *params,
            self.PHYSICAL_SPACE + self.safety_space,
            self.AGENT_MAX_VEL,
            tuple(self._agent_vel)
        )
        for pos, vel in zip(self._crowd_poss, self._crowd_vels):
            self.sim.addAgent(
                tuple(pos),
                *params,
                self.PHYSICAL_SPACE + self.safety_space,
                self.CROWD_MAX_VEL,
                tuple(vel)
            )


    def _start_env_vars(self):
        agent_pos, agent_vel, goal_pos, crowd_poss, _ = super()._start_env_vars()
        self._crowd_goal_poss = self._gen_crowd_goal(crowd_poss)

        return agent_pos, agent_vel, goal_pos, crowd_poss, crowd_poss * 0


    def _gen_crowd_goal(self, crowd_poss):
        """
        Generated random goals for each member of the crowd.

        Args:
            crowd_poss (numpy.ndarray): list of crowd members

        Returns:
            (numpy.ndarray, numpy.ndarray, numpy.ndarray): the goal positions
        """
        if len(crowd_poss.shape) == 1:
            crowd_poss = np.array([crowd_poss])
        crowd_goal_poss = np.random.uniform(
            [-self.W_BORDER, -self.H_BORDER],
            [self.W_BORDER, self.H_BORDER],
            (len(crowd_poss), 2)
        )

        return crowd_goal_poss


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
            self.crowd_goal_poss_cross = []
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
                for g in self._crowd_goal_poss:
                    self.crowd_goal_poss_cross.append(ax.plot(g[0], g[1], 'yx')[0])

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
                self.crowd_goal_poss_cross[i].set_data(
                    self._crowd_goal_poss[i][0], self._crowd_goal_poss[i][1]
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


    def update_crowd(self):
        """
        Create a rvo2 simulation at each time step and run one step

        Agent doesn't stop moving after it reaches the goal,
        because once it stops moving, the reciprocal rule is broken
        """
        self.sim.setAgentPosition(0, tuple(self._agent_pos))
        self.sim.setAgentVelocity(0, tuple(self._agent_vel))
        for i, (pos, vel) in enumerate(zip(self._crowd_poss, self._crowd_vels)):
            self.sim.setAgentPosition(i + 1, tuple(pos))
            self.sim.setAgentVelocity(i + 1, tuple(vel))

        # Set the preferred velocity to be a vector of unit magnitude (speed) in the direction of the goal.
        velocity = np.array(self._goal_pos - self._agent_vel)
        speed = np.linalg.norm(velocity)
        pref_vel = velocity / speed if speed > 1 else velocity
        self.sim.setAgentPrefVelocity(0, tuple(pref_vel))

        crowd_pref_vels = self._crowd_goal_poss - self._crowd_poss
        crowd_pref_vels[
            np.linalg.norm(crowd_pref_vels, axis=-1) < self.PHYSICAL_SPACE
        ] = 0
        crowd_pref_vels_speed = np.linalg.norm(crowd_pref_vels, axis=-1)
        crowd_vels_speed = np.linalg.norm(self._crowd_vels, axis=-1)
        diff_vel = crowd_pref_vels - self._crowd_vels
        diff_speed = np.linalg.norm(diff_vel, axis=-1)

        over = diff_speed > self.MAX_ACC * self._dt
        under = diff_speed < -self.MAX_ACC * self._dt
        crowd_pref_vels[over] = self._crowd_vels[over] +\
            np.einsum(
                "ij,i->ij",
                diff_vel[over],
                1 / diff_speed[over]
            ) * self.MAX_ACC * self._dt
        crowd_pref_vels[under] = self._crowd_vels[under] -\
            np.einsum(
                "ij,i->ij",
                diff_vel[under],
                1 / diff_speed[under]
            ) * self.MAX_ACC * self._dt
        crowd_pref_vels_speed = np.linalg.norm(crowd_pref_vels, axis=-1)

        over_vel = crowd_pref_vels_speed > self.CROWD_MAX_VEL
        crowd_pref_vels[over_vel] = np.einsum(
            "ij,i->ij", crowd_pref_vels[over_vel], 1 / crowd_pref_vels_speed[over_vel]
        ) * self.CROWD_MAX_VEL
        for i in range(self.n_crowd):
            self.sim.setAgentPrefVelocity(i + 1, tuple(crowd_pref_vels[i]))


        self.sim.doStep()

        actions = np.empty((self.n_crowd, 2))
        for i in range(self.n_crowd):
            actions[i] = np.array(self.sim.getAgentVelocity(i + 1))

        return actions


    def step(self, action: np.ndarray):
        """
        A single step with action in angular velocity space
        """
        self.update_state(action)
        self._crowd_vels = self.update_crowd()
        self._crowd_poss += self._crowd_vels * self._dt

        self._goal_reached = self.check_goal_reached()
        self._is_collided = self._check_collisions()
        self._current_reward, info = self._get_reward(action)

        self._steps += 1
        terminated = self._terminate(info)
        truncated = False

        return self._get_obs().copy(), self._current_reward, terminated, truncated, info
