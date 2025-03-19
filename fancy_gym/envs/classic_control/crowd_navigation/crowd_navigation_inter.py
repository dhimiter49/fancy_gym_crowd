from typing import Tuple, Optional, Any, Dict
import matplotlib.pyplot as plt
import numpy as np
from gymnasium import spaces
from gymnasium.core import ObsType

from fancy_gym.envs.classic_control.crowd_navigation.crowd_navigation\
    import CrowdNavigationEnv


class CrowdNavigationInterEnv(CrowdNavigationEnv):
    """
    Inter crowd control. All members of the crowd or a part of them are controled by the
    same learned policy.

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
        dt: float = 0.1,
        width: int = 20,
        height: int = 20,
        interceptor_percentage: float = 0.5,
        velocity_control: bool = False,
        lidar_rays: int = 0,
        sequence_obs: bool = False,
        one_way: bool = False,
        polar: bool = False,
        time_frame: int = 0,
        lidar_vel: bool = False,
        n_frames: int = 4,
    ):
        assert time_frame == 0 or not lidar_vel
        assert not sequence_obs or lidar_rays == 0  # cannot be seq ob and lidar obs
        self.INTER_CROWD = True
        self.MAX_EPISODE_STEPS = 100
        self.one_way = one_way
        self.polar = polar
        super().__init__(
            n_crowd,
            dt,
            width,
            height,
            interceptor_percentage,
            velocity_control=velocity_control,
        )

        self.seq_obs = sequence_obs
        self.lidar = lidar_rays != 0
        if self.lidar:
            self.lidar_vel = lidar_vel
            self.N_RAYS = lidar_rays
            self._n_frames = n_frames if not self.lidar_vel else 2  # one for each pos-vel
            self.use_time_frame = time_frame != 0
            self.time_frame = time_frame
            self.frame_steps = int((time_frame * 10) / (self.dt * 10)) \
                if self.use_time_frame else None
            self._last_frames = np.zeros((self._n_frames, self.N_RAYS))
            self.RAY_ANGLES = np.linspace(
                0, 2 * np.pi, self.N_RAYS, endpoint=False
            ) + 1e-6
            self.RAY_COS = np.cos(self.RAY_ANGLES)
            self.RAY_SIN = np.sin(self.RAY_ANGLES)

        state_bound_min = np.hstack([self.observation_space.low] * self.n_crowd)
        state_bound_max = np.hstack([self.observation_space.high] * self.n_crowd)
        action_bound_min = np.hstack([self.action_space.low] * self.n_crowd)
        action_bound_max = np.hstack([self.action_space.high] * self.n_crowd)

        self.observation_space = spaces.Box(
            low=state_bound_min, high=state_bound_max, shape=state_bound_min.shape
        )
        self.action_space = spaces.Box(
            low=action_bound_min, high=action_bound_max, shape=action_bound_min.shape
        )


    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[ObsType, Dict[str, Any]]:
        if self.lidar:
            self._last_frames *= 0
        return super().reset(seed=seed, options=options)


    def _get_reward(self, action: np.ndarray):
        dg = np.linalg.norm(self._crowd_poss - self._crowd_goal_poss, axis=-1)
        self._goal_reached = np.logical_and(
            dg < self.PHYSICAL_SPACE,  # close enought to goal
            np.linalg.norm(self._crowd_vels) < self.MAX_ACC * self._dt  # low velocity
        )
        Rg = self._goal_reached * self.Tc
        # Goal distance when task is not completed yet
        idx_no_goal = np.where(self._goal_reached == 0)[0]
        Rg = Rg.astype(np.float32)
        Rg[idx_no_goal] += -self.Cg * np.clip(dg[idx_no_goal], 1, np.inf) ** 2

        Rc = self._is_collided * self.COLLISION_REWARD
        # Crowd distance
        rel_crowd_poss = np.repeat(self._crowd_poss, (self.n_crowd - 1), axis=0) -\
            np.stack([self._crowd_poss] * (self.n_crowd - 1)).reshape(-1, 2)[[
                j for i in range(self.n_crowd) for j in range(self.n_crowd) if i != j
            ]]
        rel_crowd_poss = rel_crowd_poss.reshape(self.n_crowd, self.n_crowd - 1, 2)
        dist_crowd = np.linalg.norm(rel_crowd_poss, axis=-1)
        Rc = np.sum(
            (1 - np.exp(self.Cc / dist_crowd)) *
            (dist_crowd < self.SOCIAL_SPACE + self.PHYSICAL_SPACE),
            axis=-1
        )

        # Walls, only one of the walls is closer (irrelevant which)
        dist_walls = np.array([
            np.max(
                np.stack([
                    self.W_BORDER - abs(self._crowd_poss[:, 0]),
                    [self.PHYSICAL_SPACE] * self.n_crowd
                ]).T,
                axis=-1
            ),
            np.max(
                np.stack([
                    self.H_BORDER - abs(self._crowd_poss[:, 1]),
                    [self.PHYSICAL_SPACE] * self.n_crowd
                ]).T,
                axis=-1
            ),
        ]).T
        Rw = np.sum(
            np.einsum(
                'ij,i->ij',
                (1 - np.exp(self.Cc / dist_walls)),
                np.sum((dist_walls < self.PHYSICAL_SPACE * 2), axis=-1) > 0
            ),
            axis=-1

        )

        reward = Rg + Rc + Rw
        return reward, dict(goal=Rg, collision=Rc, wall=Rw)


    def _terminate(self, info):
        return np.any(self._is_collided)


    def _get_obs(self) -> ObsType:
        rel_goal_poss = self._crowd_goal_poss - self._crowd_poss
        rel_goal_poss = self.c2p(rel_goal_poss) if self.polar else rel_goal_poss
        crowd_vels = self.c2p(self._crowd_vels) if self.polar else self._crowd_vels
        if self.lidar:
            obs = []
            for i, pos in enumerate(self._crowd_poss):
                others_poss = np.delete(self._crowd_poss, i, axis=0)
                others_vels = np.delete(self._crowd_vels, i, axis=0)
                wall_distances = np.min([
                    (self.W_BORDER - np.where(
                        self.RAY_COS > 0, pos[0], -pos[0]
                    )) / np.abs(self.RAY_COS),
                    (self.H_BORDER - np.where(
                        self.RAY_SIN > 0, pos[1], -pos[1]
                    )) / np.abs(self.RAY_SIN)
                ], axis=0)

                x_crowd_rel, y_crowd_rel = others_poss[:, 0] - pos[0], \
                    others_poss[:, 1] - pos[1]
                orthog_dist = np.abs(
                    np.outer(x_crowd_rel, self.RAY_SIN) -
                    np.outer(y_crowd_rel, self.RAY_COS)
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
                        others_poss, others_vels
                    )):
                        intersection = intersections_mask[i]
                        if np.any(intersection):
                            dir_idxs = np.where(intersection == 1)[0]
                            for dir_idx in dir_idxs:
                                lidar_vec = np.array([
                                    self.RAY_COS[dir_idx], self.RAY_SIN[dir_idx]
                                ])
                                if min_intersect_distances[dir_idx] == np.inf or\
                                   np.dot(lidar_vec, member_pos - pos) < 0:
                                    continue  # account for correct direction
                                vel_along_dir = np.dot(lidar_vec, member_vel)
                                if ray_velocities[dir_idx] > 0 and\
                                   np.linalg.norm(member_pos - pos) >\
                                   ray_distances[dir_idx]:  # keep only the closest one
                                    continue
                                ray_velocities[dir_idx] = vel_along_dir

                    self._last_frames[0] = ray_distances
                    self._last_frames[1] = ray_velocities
                obs.append(rel_goal_poss[i])
                obs.append(crowd_vels[i])
                obs.append(self._last_frames.flatten())

            return np.concatenate(obs).astype(np.float32).flatten()
        else:
            rel_crowd_poss = -np.repeat(self._crowd_poss, (self.n_crowd - 1), axis=0) +\
                np.stack([self._crowd_poss] * (self.n_crowd - 1)).reshape(-1, 2)[[
                    j for i in range(self.n_crowd) for j in range(self.n_crowd) if i != j
                ]]
            rel_goal_poss = self._crowd_goal_poss - self._crowd_poss
            crowd_vels = np.stack([self._crowd_vels] * (self.n_crowd - 1)).\
                reshape(-1, 2)[[
                    j for i in range(self.n_crowd) for j in range(self.n_crowd) if i != j
                ]]

            if self.seq_obs:
                rel_crowd_poss = rel_crowd_poss.reshape(self.n_crowd, self.n_crowd - 1, 2)
                crowd_vels = crowd_vels.reshape(self.n_crowd, self.n_crowd - 1, 2)
                zip_member_obs = list(zip(
                    np.hstack([self._crowd_poss, self._crowd_vels]),
                    np.hstack([rel_goal_poss, 0 * self._crowd_vels]),
                    np.concatenate([
                        rel_crowd_poss,
                        crowd_vels
                    ], axis=-1).reshape(self.n_crowd, -1),
                ))
                array_member_obs = np.array([np.concatenate(x) for x in zip_member_obs])
                return np.concatenate([array_member_obs]).astype(np.float32).flatten()

            dist_walls = []
            for member in self._crowd_poss:
                dist_walls.append(np.array([
                    [self.W_BORDER - member[0], self.W_BORDER + member[0]],
                    [self.H_BORDER - member[1], self.H_BORDER + member[1]]
                ]))
            dist_walls = np.array(dist_walls)
            zip_member_obs = list(zip(
                rel_goal_poss.reshape(self.n_crowd, -1),
                rel_crowd_poss.reshape(self.n_crowd, -1),
                self._crowd_vels,
                crowd_vels.reshape(self.n_crowd, -1),
                dist_walls.reshape(self.n_crowd, -1)
            ))
            array_member_obs = np.array([np.concatenate(x) for x in zip_member_obs])

            return np.concatenate([array_member_obs]).astype(np.float32).flatten()


    def _start_env_vars(self):
        agent_pos, agent_vel, goal_pos, crowd_poss, _ = super()._start_env_vars()
        next_crowd_vels = np.zeros(crowd_poss.shape)

        self._crowd_goal_poss = self._gen_crowd_goal(crowd_poss)

        return agent_pos, agent_vel, goal_pos, crowd_poss, next_crowd_vels


    def _gen_crowd_goal(self, crowd_poss):
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
                for angle, distance in zip(self.RAY_ANGLES, self._last_frames[0]):
                    self.lidar_rays.append(ax.arrow(
                        self._crowd_poss[0][0], self._crowd_poss[0][1],
                        distance * np.cos(angle), distance * np.sin(angle),
                        head_width=0.0,
                        ec=(0.5, 0.5, 0.5, 0.3),
                        linestyle="--"
                    ))

            # Crowd velocity
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
                for g in self._crowd_goal_poss:
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
            "Reward at this step:" + str(np.around(self._current_reward, 3)),
            fontsize=11,
            fontweight='bold'
        )

        if self._steps == 1:
            self.goal_point.set_data(self._goal_pos[0], self._goal_pos[1])

        for i, member in enumerate(self._crowd_poss):
            self.ScS_crowd[i].center = member
            self.PrS_crowd[i].center = member
            self.PhS_crowd[i].center = member
            if not self.const_vel:
                self.crowd_goal_points[i].set_data(
                    self._crowd_goal_poss[i][0], self._crowd_goal_poss[i][1]
                )
        for i in range(self.n_crowd):
            self.vel_crowd[i].set_data(
                x=self._crowd_poss[i][0], y=self._crowd_poss[i][1],
                dx=self._crowd_vels[i][0], dy=self._crowd_vels[i][1]
            )
        if self.lidar:
            for i, (angle, distance) in \
                enumerate(zip(self.RAY_ANGLES, self._last_frames[0])):
                self.lidar_rays[i].set_data(
                    x=self._crowd_poss[0][0], y=self._crowd_poss[0][1],
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


    def update_state(self, action):
        """
        Update robot position and velocity for time self._dt based on its dynamics.

        Args:
            action (numpy.ndarray): 2D (n_crowd, 2) array representing the acc for current
                step
        """
        self._last_crowd_poss = self._crowd_poss.copy()
        if self.velocity_control:
            vels = self.p2c(action) if self.polar else action
            accs = (vels - self._crowd_vels) / self._dt
            accs_norm = np.linalg.norm(accs, axis=-1)
            idxs_acc_too_high = np.where(accs_norm > self.MAX_ACC)[0]
            if len(idxs_acc_too_high) > 0:
                vels[idxs_acc_too_high] = self._crowd_vels[idxs_acc_too_high] + np.einsum(
                    'ij,i->ij',
                    accs[idxs_acc_too_high],
                    1 / accs_norm[idxs_acc_too_high] * self.MAX_ACC * self._dt
                )
            vels_norm = np.linalg.norm(vels, axis=-1)
            idxs_vel_too_high = np.where(vels_norm > self.AGENT_MAX_VEL)[0]
            if len(idxs_vel_too_high) > 0:
                vels[idxs_acc_too_high] = np.einsum(
                    'ij,i->ij',
                    vels[idxs_acc_too_high],
                    self.AGENT_MAX_VEL / vels_norm[idxs_vel_too_high]
                )

            self._crowd_poss += (self._crowd_vels + vels) * self._dt / 2
            self._crowd_vels = vels
        else:
            accs = action
            accs_norm = np.linalg.norm(accs, axis=-1)
            idxs_acc_too_high = np.where(accs_norm > self.MAX_ACC)[0]
            if len(idxs_acc_too_high) > 0:
                accs[idxs_acc_too_high] = np.einsum(
                    'ij,i->ij',
                    accs[idxs_acc_too_high],
                    self.MAX_ACC / accs_norm[idxs_acc_too_high]
                )

            self._crowd_poss += self._crowd_vels * self._dt + accs * 0.5 * self._dt ** 2
            self._crowd_vels += accs * self._dt

            crowd_speeds = np.linalg.norm(self._crowd_vels, axis=-1)
            idxs_vel_too_high = np.where(crowd_speeds > self.AGENT_MAX_VEL)[0]
            if len(idxs_vel_too_high) > 0:
                self._crowd_vels[idxs_vel_too_high] = np.einsum(
                    'ij,i->ij',
                    self._crowd_vels[idxs_vel_too_high],
                    self.AGENT_MAX_VEL / crowd_speeds[idxs_vel_too_high]
                )

        # check bounds of the environment and the bounds of the maximum velocity
        self._crowd_poss = np.clip(
            self._crowd_poss,
            [-self.W_BORDER, -self.H_BORDER],
            [self.W_BORDER, self.H_BORDER]
        )


    def _check_collisions(self) -> np.ndarray:
        """
        Checks whether any pair of crowd members are too close to each other or the wall
        """
        # Crowd
        if self.n_crowd > 0:
            if self.supersample_col:
                over_sample_by = self._dt / 0.01
                crowd_poss = self._last_crowd_poss + np.einsum(
                    "i,kj->ikj",
                    np.arange(0, int(over_sample_by) + 1),
                    self._crowd_poss - self._last_crowd_poss
                ) / over_sample_by
                time_dim = crowd_poss.shape[0]

                rel_crowd_poss = np.repeat(crowd_poss, (self.n_crowd - 1), axis=1) -\
                    np.stack([crowd_poss] * (self.n_crowd - 1)).reshape(time_dim, -1, 2)[
                        :, [
                            j for i in range(self.n_crowd)
                            for j in range(self.n_crowd) if i != j
                        ]
                ]
                rel_crowd_poss = rel_crowd_poss.reshape(
                    time_dim, self.n_crowd, self.n_crowd - 1, 2
                )

                inter_crowd_crash = np.sum(
                    np.linalg.norm(rel_crowd_poss, axis=-1) < self.PHYSICAL_SPACE * 2,
                    axis=(0, -1)  # sum over time for each crowd member
                ) > 0  # if bigger than one than at least one crash for one member
            else:
                rel_crowd_poss = np.repeat(
                    self._crowd_poss, (self.n_crowd - 1), axis=0
                ) - np.stack([self._crowd_poss] * (self.n_crowd - 1)).reshape(-1, 2)[[
                    j for i in range(self.n_crowd)
                    for j in range(self.n_crowd) if i != j
                ]]
                rel_crowd_poss = rel_crowd_poss.reshape(self.n_crowd, self.n_crowd - 1, 2)
                inter_crowd_crash = np.sum(
                    np.linalg.norm(rel_crowd_poss, axis=-1) < [self.PHYSICAL_SPACE * 2],
                    axis=-1
                ) > 0

        # TODO: Walls
        wall_crash = np.sum(
            np.abs(self._crowd_poss) >
            np.array([self.W_BORDER, self.H_BORDER]) - self.PHYSICAL_SPACE,
            axis=-1
        )

        return np.logical_or(inter_crowd_crash, wall_crash)


    def step(self, action: np.ndarray):
        """
        A single step with action in angular velocity space
        """
        assert action.shape == (self.n_crowd * 2,)
        action = action.reshape(self.n_crowd, 2)
        self._last_crowd_poss = self._crowd_poss.copy()
        self.update_state(action)

        self._is_collided = self._check_collisions()
        self._current_reward, info = self._get_reward(action)
        dummy_rew = np.sum(self._current_reward)
        info["terminal"] = self._is_collided
        info["rewards"] = self._current_reward

        self._steps += 1
        terminated = self._terminate(info)
        truncated = False

        return self._get_obs().copy(), dummy_rew, terminated, truncated, info
