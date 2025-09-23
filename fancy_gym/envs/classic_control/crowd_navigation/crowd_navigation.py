from typing import Tuple, Optional, Any, Dict
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate as interp
from gymnasium import spaces
from gymnasium.core import ObsType

from fancy_gym.envs.classic_control.crowd_navigation.base_crowd_navigation\
    import BaseCrowdNavigationEnv
from fancy_gym.envs.classic_control.crowd_navigation.utils import REPLAN_MOVING


class CrowdNavigationEnv(BaseCrowdNavigationEnv):
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
        dt: float = 0.1,
        width: int = 20,
        height: int = 20,
        interceptor_percentage: float = 0.5,
        discrete_action: bool = False,
        velocity_control: bool = False,
        lidar_rays: int = 0,
        lidar_max: float = 0.0,
        sequence_obs: bool = False,
        const_vel: bool = False,
        one_way: bool = False,
        polar: bool = False,
        time_frame: int = 0,
        lidar_vel: bool = False,
        n_frames: int = 4,
        intrinsic_rew: bool = False,
        one_goal: bool = False,
    ):
        assert time_frame == 0 or not lidar_vel
        assert not sequence_obs or lidar_rays == 0  # cannot be seq ob and lidar obs
        # need to specify num of rays if there is a maximum distance to the lidar
        assert not lidar_max > 0.0 or lidar_rays > 0
        self.MAX_EPISODE_STEPS = 100
        self.const_vel = const_vel
        self.one_way = one_way
        self.polar = polar
        self.replan = REPLAN_MOVING
        self.one_goal = one_goal
        super().__init__(
            n_crowd,
            width,
            height,
            interceptor_percentage,
            allow_collision=False,
            discrete_action=discrete_action,
            velocity_control=velocity_control,
            dt=dt,
        )

        self.seq_obs = sequence_obs
        self.intrinsic_rew = intrinsic_rew
        self.lidar = lidar_rays != 0
        self.lidar_max = lidar_max if lidar_max > 0.0 else np.inf
        max_dist = np.linalg.norm(np.array([self.WIDTH, self.HEIGHT]))
        if self.lidar:
            self.lidar_vel = lidar_vel
            self.N_RAYS = lidar_rays
            self._n_frames = n_frames if not self.lidar_vel else 2  # one for each pos-vel
            self.use_time_frame = time_frame != 0
            self.time_frame = time_frame
            if self.use_time_frame:
                self.frame_steps = int((time_frame * 10) / (self.dt * 10))
            self._last_frames = np.zeros((self._n_frames, self.N_RAYS))
            if self.use_time_frame:
                self._last_second_frames = np.zeros((self.frame_steps, self.N_RAYS))
            self.RAY_ANGLES = np.linspace(
                0, 2 * np.pi, self.N_RAYS, endpoint=False
            ) + 1e-6
            self.RAY_COS = np.cos(self.RAY_ANGLES)
            self.RAY_SIN = np.sin(self.RAY_ANGLES)
        if hasattr(self, 'INTER_CROWD'):
            self.n_crowd -= 1
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
            if self.polar:
                if self.var_radius:
                    max_dist = np.linalg.norm([self.W_BORDER, self.H_BORDER])
                    state_bound_min = np.hstack([
                        [-max_dist, -np.pi, 0, self.MIN_RADIUS] * (2 + self.n_crowd),
                    ])
                    state_bound_max = np.hstack([
                        [max_dist, np.pi, self.AGENT_MAX_VEL, self.MAX_RADIUS] *
                        (2 + self.n_crowd)
                    ])
                else:
                    max_dist = np.linalg.norm([self.W_BORDER, self.H_BORDER])
                    state_bound_min = np.hstack([
                        [-max_dist, -np.pi, 0] * (2 + self.n_crowd),
                    ])
                    state_bound_max = np.hstack([
                        [max_dist, np.pi, self.AGENT_MAX_VEL] * (2 + self.n_crowd)
                    ])
            else:
                if self.var_radius:
                    state_bound_min = np.hstack([
                        [
                            -self.W_BORDER,
                            -self.H_BORDER,
                            -self.AGENT_MAX_VEL,
                            -self.AGENT_MAX_VEL,
                            self.MIN_RADIUS
                        ],
                        [
                            -self.WIDTH,
                            -self.HEIGHT,
                            -self.AGENT_MAX_VEL,
                            -self.AGENT_MAX_VEL,
                            0
                        ],
                        [
                            -self.WIDTH,
                            -self.HEIGHT,
                            -self.CROWD_MAX_VEL,
                            -self.CROWD_MAX_VEL,
                            self.MIN_RADIUS
                        ] * self.n_crowd,
                    ])
                    state_bound_max = np.hstack([
                        [
                            self.W_BORDER,
                            self.H_BORDER,
                            self.AGENT_MAX_VEL,
                            self.AGENT_MAX_VEL,
                            self.MAX_RADIUS
                        ],
                        [
                            self.WIDTH,
                            self.HEIGHT,
                            self.AGENT_MAX_VEL,
                            self.AGENT_MAX_VEL,
                            0
                        ],
                        [
                            self.WIDTH,
                            self.HEIGHT,
                            self.CROWD_MAX_VEL,
                            self.CROWD_MAX_VEL,
                            self.MAX_RADIUS
                        ] * self.n_crowd,
                    ])
                else:
                    state_bound_min = np.hstack([
                        [
                            -self.W_BORDER,
                            -self.H_BORDER,
                            -self.AGENT_MAX_VEL,
                            -self.AGENT_MAX_VEL
                        ],
                        [
                            -self.WIDTH,
                            -self.HEIGHT,
                            -self.AGENT_MAX_VEL,
                            -self.AGENT_MAX_VEL
                        ],
                        [
                            -self.WIDTH,
                            -self.HEIGHT,
                            -self.CROWD_MAX_VEL,
                            -self.CROWD_MAX_VEL
                        ] * self.n_crowd,
                    ])
                    state_bound_max = np.hstack([
                        [
                            self.W_BORDER,
                            self.H_BORDER,
                            self.AGENT_MAX_VEL,
                            self.AGENT_MAX_VEL
                        ],
                        [self.WIDTH, self.HEIGHT, self.AGENT_MAX_VEL, self.AGENT_MAX_VEL],
                        [
                            self.WIDTH,
                            self.HEIGHT,
                            self.CROWD_MAX_VEL,
                            self.CROWD_MAX_VEL
                        ] * self.n_crowd,
                    ])
        else:
            if self.var_radius:
                state_bound_min = np.hstack([
                    [-self.WIDTH, -self.HEIGHT] * (self.n_crowd + 1),
                    [-self.AGENT_MAX_VEL, -self.AGENT_MAX_VEL],
                    [-self.CROWD_MAX_VEL, -self.CROWD_MAX_VEL] * self.n_crowd,
                    [0] * 4,  # four directions
                    [self.MIN_RADIUS] * (self.n_crowd + 1)
                ])
                state_bound_max = np.hstack([
                    [self.WIDTH, self.HEIGHT] * (self.n_crowd + 1),
                    [self.AGENT_MAX_VEL, self.AGENT_MAX_VEL],
                    [self.CROWD_MAX_VEL, self.CROWD_MAX_VEL] * self.n_crowd,
                    np.repeat([self.WIDTH, self.HEIGHT], 2),  # four directions
                    [self.MAX_RADIUS] * (self.n_crowd + 1),
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
        if hasattr(self, 'INTER_CROWD'):
            self.n_crowd += 1

        self.observation_space = spaces.Box(
            low=state_bound_min, high=state_bound_max, shape=state_bound_min.shape
        )


    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[ObsType, Dict[str, Any]]:
        if self.lidar:
            self._last_frames *= 0
        return super().reset(seed=seed, options=options)


    def _get_reward(self, action: np.ndarray):
        dg = np.linalg.norm(self._agent_pos - self._goal_pos)
        if self._goal_reached:
            Rg = self.Tc
        else:
            # Goal distance
            dg_old = np.linalg.norm(self._last_agent_pos - self._goal_pos)
            dg_diff = dg_old - dg
            Rg = self.Cg * np.sign(dg_diff) * dg_diff ** 2

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
                (dist_crowd < self.SOCIAL_SPACE[1:1 + self.n_crowd] +
                    self.PHYSICAL_SPACE[0])
            )

        # Walls, only one of the walls is closer (irrelevant which)
        dist_walls = np.array([
            max(self.W_BORDER - abs(self._agent_pos[0]), self.PHYSICAL_SPACE[0]),
            max(self.H_BORDER - abs(self._agent_pos[1]), self.PHYSICAL_SPACE[0]),
        ])
        Rw = np.sum(
            (1 - np.exp(self.Cc / dist_walls)) *
            (dist_walls < self.PHYSICAL_SPACE[0] * 2)
        )

        reward = Rg + Rc + Rw
        return reward, dict(goal=Rg, collision=Rc, wall=Rw)


    def _get_intrinsic_reward(self):
        """
        Check how far the current position after the action is relative to the desired
        position proposed by the ProDMP.
        """
        Ri = 0
        if self.desired_position is not None:
            Ri = self.Ci * np.linalg.norm(self._agent_pos - self.desired_position)
        return Ri, dict(intrinsic=Ri)


    def _terminate(self, info):
        return self._is_collided or self._goal_reached


    def _get_obs(self) -> ObsType:
        rel_goal_pos = self._goal_pos - self._agent_pos
        rel_goal_pos = self.c2p(rel_goal_pos) if self.polar else rel_goal_pos
        agent_vel = self.c2p(self._agent_vel) if self.polar else self._agent_vel
        if self.lidar:
            wall_or_max_distances = np.min([
                (self.W_BORDER - np.where(
                    self.RAY_COS > 0, self._agent_pos[0], -self._agent_pos[0]
                )) / np.abs(self.RAY_COS),
                (self.H_BORDER - np.where(
                    self.RAY_SIN > 0, self._agent_pos[1], -self._agent_pos[1]
                )) / np.abs(self.RAY_SIN)
            ], axis=0)
            if self.lidar_max is not None:
                wall_or_max_distances = np.minimum(
                    wall_or_max_distances,
                    np.ones_like(wall_or_max_distances) * self.lidar_max
                )

            if self.n_crowd > 0:
                x_crowd_rel, y_crowd_rel = self._crowd_poss[:, 0] - self._agent_pos[0], \
                    self._crowd_poss[:, 1] - self._agent_pos[1]
                orthog_dist = np.abs(
                    np.outer(x_crowd_rel, self.RAY_SIN) -
                    np.outer(y_crowd_rel, self.RAY_COS)
                )
                all_rays_physical_space = np.repeat(
                    self.PHYSICAL_SPACE[1:1 + self.n_crowd],
                    orthog_dist.shape[-1]
                ).reshape(orthog_dist.shape)
                intersections_mask = orthog_dist <= all_rays_physical_space
                along_dist = np.outer(x_crowd_rel, self.RAY_COS) +\
                    np.outer(y_crowd_rel, self.RAY_SIN)
                orthog_to_intersect_dist = np.sqrt(np.maximum(
                    all_rays_physical_space ** 2 - orthog_dist ** 2, 0
                ))
                intersect_distances = np.where(
                    intersections_mask, along_dist - orthog_to_intersect_dist, np.inf
                )
                intersect_distances = np.where(
                    intersect_distances < self.lidar_max, intersect_distances, np.inf
                )
                min_intersect_distances = np.min(np.where(
                    intersect_distances > 0, intersect_distances, np.inf), axis=0
                )
                ray_distances = np.minimum(min_intersect_distances, wall_or_max_distances)
            else:
                ray_distances = wall_or_max_distances
            self.ray_distances = ray_distances

            if not self.use_time_frame and not self.lidar_vel:
                if not np.any(self._last_frames):
                    self._last_frames[list(range(len(self._last_frames)))] = \
                        np.array(ray_distances)
                else:
                    self._last_frames[:-1] = self._last_frames[1:]
                    self._last_frames[-1] = ray_distances
            elif self.n_crowd > 0 and self.lidar_vel:
                ray_velocities = np.zeros(ray_distances.shape)
                if self.n_crowd > 0:
                    vel_along_all_dir_all_crowd = np.einsum(
                        "ij,ij->i",
                        np.concatenate(
                            [np.array(list(zip(self.RAY_COS, self.RAY_SIN)))] *
                            self.n_crowd
                        ),
                        np.repeat(self._crowd_vels, self.N_RAYS, axis=0)
                    )
                    vel_along_all_dir_all_crowd *= intersections_mask.flatten()
                    viable_distances = np.where(
                        intersect_distances > 0, intersect_distances, np.inf
                    )
                    crowd_min_dist_idx = np.argmin(  # which one is closer
                        viable_distances, axis=0
                    )
                    vel_along_dir = vel_along_all_dir_all_crowd[
                        crowd_min_dist_idx * self.N_RAYS + np.arange(self.N_RAYS)
                    ]
                    intersection_mask_dir = min_intersect_distances != np.inf
                    ray_velocities = vel_along_dir * intersection_mask_dir
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
            if self.polar:
                if np.linalg.norm(self._agent_vel) > 0:
                    orient = self._agent_vel / np.linalg.norm(self._agent_vel)
                else:
                    orient = np.array([1, 0])
                rel_goal_pos = self.goal_pos - self._agent_pos
                goal_angle_rel_orient = np.sign(np.cross(rel_goal_pos, orient)) *\
                    np.arccos(np.clip(
                        np.dot(
                            rel_goal_pos / np.linalg.norm(rel_goal_pos),
                            orient
                        ),
                        -1.0, 1.0
                    ))
                rel_crowd_pos = self._crowd_poss - self._agent_pos
                crowd_angle_rel_orient = np.sign(np.cross(rel_crowd_pos, orient)) *\
                    np.arccos(np.clip(
                        np.dot(
                            np.einsum(  # normalize
                                "ij,i->ij",
                                rel_crowd_pos,
                                1 / np.linalg.norm(rel_crowd_pos, axis=-1)
                            ),
                            orient,
                        ),
                        -1.0, 1.0
                    ))
                crowd_vel_rel_norm = np.dot(self._crowd_vels, orient)
                if self.var_radius:
                    return np.concatenate([
                        [np.concatenate([
                            self.c2p(self._agent_pos), [np.linalg.norm(self._agent_vel)],
                            [self.PHYSICAL_SPACE[0]]
                        ])],
                        [np.concatenate([
                            [np.linalg.norm(rel_goal_pos), goal_angle_rel_orient],
                            [0, 0],
                        ])],
                        np.concatenate([
                            np.linalg.norm(rel_crowd_pos, axis=-1).reshape(-1, 1),
                            crowd_angle_rel_orient.reshape(-1, 1),
                            crowd_vel_rel_norm.reshape(-1, 1),
                            self.PHYSICAL_SPACE[1:1 + self.n_crowd].reshape(-1, 1)
                        ], axis=-1),
                    ]).astype(np.float32).flatten()
                else:
                    return np.concatenate([
                        [np.concatenate([
                            self.c2p(self._agent_pos), [np.linalg.norm(self._agent_vel)]
                        ])],
                        [np.concatenate([
                            [np.linalg.norm(rel_goal_pos), goal_angle_rel_orient],
                            [0]
                        ])],
                        np.concatenate([
                            np.linalg.norm(rel_crowd_pos, axis=-1).reshape(-1, 1),
                            crowd_angle_rel_orient.reshape(-1, 1),
                            crowd_vel_rel_norm.reshape(-1, 1)
                        ], axis=-1),
                    ]).astype(np.float32).flatten()
            else:
                if self.var_radius:
                    return np.concatenate([
                        [np.concatenate([
                            self._agent_pos, self._agent_vel, [self.PHYSICAL_SPACE[0]]
                        ])],
                        [np.concatenate([
                            self._goal_pos - self._agent_pos, self._agent_vel * 0, [0]
                        ])],
                        np.concatenate([
                            self._crowd_poss - self._agent_pos,
                            self._crowd_vels,
                            self.PHYSICAL_SPACE[1:1 + self.n_crowd].reshape(-1, 1)
                        ], axis=-1)
                    ]).astype(np.float32).flatten()
                else:
                    return np.concatenate([
                        [np.concatenate([self._agent_pos, self._agent_vel])],
                        [np.concatenate([
                            self._goal_pos - self._agent_pos, self._agent_vel * 0
                        ])],
                        np.concatenate([
                            self._crowd_poss - self._agent_pos, self._crowd_vels
                        ], axis=-1)
                    ]).astype(np.float32).flatten()
        else:
            rel_crowd_poss = self._crowd_poss - self._agent_pos
            rel_crowd_poss = self.c2p(rel_crowd_poss) if self.polar else rel_crowd_poss
            dist_walls = np.array([
                [self.W_BORDER - self._agent_pos[0], self.W_BORDER + self._agent_pos[0]],
                [self.H_BORDER - self._agent_pos[1], self.H_BORDER + self._agent_pos[1]]
            ])
            if self.var_radius:
                return np.concatenate([
                    rel_goal_pos,
                    rel_crowd_poss.flatten(),
                    agent_vel,
                    self._crowd_vels.flatten(),
                    dist_walls.flatten(),
                    self.PHYSICAL_SPACE
                ]).astype(np.float32).flatten()
            else:
                return np.concatenate([
                    [rel_goal_pos],
                    rel_crowd_poss,
                    [agent_vel],
                    self._crowd_vels,
                    dist_walls
                ]).astype(np.float32).flatten()


    def _read_test_case(self):
        agent_pos, agent_vel, goal_pos, crowd_poss, _ = super()._read_test_case()
        if "Inter" not in type(self).__name__:
            crowd_goal_poss = self._test_case_array[self._test_case_idx, 1:, 2:4]
        else:
            crowd_goal_poss = self._test_case_array[self._test_case_idx, 0:, 2:4]
        next_crowd_vels = np.zeros(crowd_poss.shape)

        if self.const_vel:
            next_crowd_vels = (crowd_goal_poss - crowd_poss)
            next_crowd_vels = np.einsum(
                "ij,i->ij",
                next_crowd_vels,
                np.random.uniform(0.5, self.CROWD_MAX_VEL) /
                np.linalg.norm(next_crowd_vels, axis=-1)
            )
        else:
            (
                self._crowd_goal_poss, self._planned_crowd_vels, next_crowd_vels
            ) = self._gen_crowd_goal_and_plan(crowd_poss)

        return agent_pos, agent_vel, goal_pos, crowd_poss, next_crowd_vels


    def _start_env_vars(self):
        agent_pos, agent_vel, goal_pos, crowd_poss, _ = super()._start_env_vars()
        next_crowd_vels = np.zeros(crowd_poss.shape)

        if self.const_vel:
            for i, c in enumerate(crowd_poss):
                if c[0] > 0 or self.one_way:
                    idx = np.random.choice([0, 1])
                    if idx == 0:
                        pol_vel = np.random.uniform(
                            [0.5, np.pi * 5 / 6], [self.CROWD_MAX_VEL, np.pi]
                        )
                    else:
                        pol_vel = np.random.uniform(
                            [0.5, -np.pi], [self.CROWD_MAX_VEL, -np.pi * 5 / 6]
                        )
                else:
                    pol_vel = np.random.uniform(
                        [0.5, -np.pi * 1 / 6], [self.CROWD_MAX_VEL, np.pi * 1 / 6]
                    )
                next_crowd_vels[i] = self.p2c(pol_vel)
        else:
            (
                self._crowd_goal_poss, self._planned_crowd_vels, next_crowd_vels
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
        if self.run_test_case:
            if "Inter" not in type(self).__name__:
                crowd_goal_poss = self._test_case_array[self._test_case_idx, 1:, 2:4]
            else:
                crowd_goal_poss = self._test_case_array[self._test_case_idx, 0:, 2:4]
        else:
            crowd_goal_poss = np.random.uniform(
                [-self.W_BORDER, -self.H_BORDER],
                [self.W_BORDER, self.H_BORDER],
                (len(crowd_poss), 2)
            )

        crowd_vels = []
        next_crowd_vels = np.zeros(crowd_poss.shape)
        max_step_acc = self.MAX_ACC * self._dt
        for i, goal in enumerate(crowd_goal_poss):
            dist = np.linalg.norm(goal - crowd_poss[i])
            if dist > self.MAX_STOPPING_DIST_CROWD * 2:
                t_max_vel = (dist - self.MAX_STOPPING_DIST_CROWD * 2) / self.CROWD_MAX_VEL
                acc_vels = np.arange(
                    max_step_acc, self.CROWD_MAX_VEL + 1e-8, max_step_acc
                )
                dec_vels = np.arange(
                    self.CROWD_MAX_VEL - max_step_acc, 0 - 1e-8, -max_step_acc
                )
                vels = np.concatenate([
                    acc_vels,
                    np.full(int(t_max_vel / self._dt), self.CROWD_MAX_VEL),
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
            if len(vels) == 0:
                vels = np.array([0])

            # Fix direction
            direction = (goal - crowd_poss[i]) / dist
            vels = np.outer(vels, direction).reshape(-1, 2)
            crowd_vels.append(np.concatenate([np.zeros((1, 2)), vels]))
            next_crowd_vels[i] = np.zeros(2)

        return crowd_goal_poss, crowd_vels, next_crowd_vels


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
                head_width=self.PERSONAL_SPACE[0] / 4,
                overhang=1,
                head_length=0.2,
                ec="g"
            )
            self.vel_crowd = []
            for i in range(self.max_n_crowd):
                if i < self.n_crowd:
                    x, y = self._crowd_poss[i][0], self._crowd_poss[i][1]
                    dx, dy = self._crowd_vels[i][0], self._crowd_vels[i][1]
                else:
                    x, y, dx, dy = 100, 100, 1, 1
                self.vel_crowd.append(ax.arrow(
                    x, y, dx, dy,
                    head_width=self.PERSONAL_SPACE[i + 1] / 4,
                    overhang=1,
                    head_length=0.2,
                    ec="r"
                ))

            self.sep_planes = []
            for i in range(self.max_n_crowd):
                if i < self.n_crowd:
                    x, y = self.separating_planes[i][0], self.separating_planes[i][1],
                    dx, dy = self.separating_planes[i][2], self.separating_planes[i][3],
                else:
                    x, y, dx, dy = 100, 100, 1, 1
                self.sep_planes.append(ax.arrow(
                    x, y, dx, dy,
                    head_width=0.0,
                    ec="r"
                ))

            # Agent
            self.space_agent = plt.Circle(
                tuple(self._agent_pos), self.PHYSICAL_SPACE[0], color="g", alpha=0.5
            )
            ax.add_patch(self.space_agent)

            # Social space, Personal space, Physical space, Crowd goal positions
            self.ScS_crowd = []
            self.PrS_crowd = []
            self.PhS_crowd = []
            self.crowd_goal_points = []
            assert self.max_n_crowd == len(self.SOCIAL_SPACE) - 1
            assert self.n_crowd == len(self._crowd_poss)
            for i, (soc, per, phy) in enumerate(zip(
                self.SOCIAL_SPACE[1:], self.PERSONAL_SPACE[1:], self.PHYSICAL_SPACE[1:]
            )):
                pos = self._crowd_poss[i] if i < self.n_crowd else np.array([-100, -100])
                self.ScS_crowd.append(
                    plt.Circle(pos, soc, color="r", fill=False, linestyle="--")
                )
                ax.add_patch(self.ScS_crowd[-1])
                self.PrS_crowd.append(
                    plt.Circle(pos, per, color="r", fill=False)
                )
                ax.add_patch(self.PrS_crowd[-1])
                self.PhS_crowd.append(
                    plt.Circle(pos, phy, color="r", alpha=0.5)
                )
                ax.add_patch(self.PhS_crowd[-1])
            if not self.const_vel:
                for i in range(self.max_n_crowd):
                    if i < self.n_crowd:
                        g = self._crowd_goal_poss[i]
                    else:
                        g = np.array([100, 100])
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
            border_penalization = self.PHYSICAL_SPACE[0] * 2
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
            if self.var_radius:
                self.space_agent.radius = self.PHYSICAL_SPACE[0]
            for i in range(self.n_crowd):
                if self.var_radius:
                    self.ScS_crowd[i].radius = self.SOCIAL_SPACE[i + 1]
                    self.PrS_crowd[i].radius = self.PERSONAL_SPACE[i + 1]
                    self.PhS_crowd[i].radius = self.PHYSICAL_SPACE[i + 1]

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
        self._crowd_poss += self._crowd_vels * self._dt
        if not self.const_vel:
            for i in range(self.n_crowd):
                self._planned_crowd_vels[i] = np.delete(self._planned_crowd_vels[i], 0, 0)
                if len(self._planned_crowd_vels[i]) == 0:
                    if not self.one_goal:
                        self._crowd_goal_poss[i], self._planned_crowd_vels[i], _ = \
                            self._gen_crowd_goal_and_plan(self._crowd_poss[i])
                        self._planned_crowd_vels[i] = self._planned_crowd_vels[i][0]
                    else:
                        self._planned_crowd_vels[i] = np.zeros((100, 2))
                self._crowd_vels[i] = self._planned_crowd_vels[i][0]


    def get_obs(self):
        return self._get_obs().copy()


    def step(self, action: np.ndarray):
        """
        A single step with action in angular velocity space
        """
        self.update_state(action)
        self._last_crowd_poss = self._crowd_poss.copy()
        self.update_crowd()

        self._goal_reached = self.check_goal_reached()
        self._is_collided = self._check_collisions()
        # if self._is_collided:
        #     self.collision_metrics()
        self._current_reward, info = self._get_reward(action)
        if self.intrinsic_rew:
            rew, new_info = self._get_intrinsic_reward()
            self._current_reward += rew
            info.update(new_info)

        self._steps += 1
        terminated = self._terminate(info)
        truncated = False

        return self._get_obs().copy(), self._current_reward, terminated, truncated, info


    def collision_metrics(self):
        self.num_env_col += 1
        print("Seed", self.current_seed)
        print("Num col", self.num_env_col)
        if len(self.idx_colliding_agents) >= 1:
            self.num_col += len(self.idx_colliding_agents)
            self.col_vel_sum += np.sum(np.linalg.norm(
                self._agent_vel - self._crowd_vels[self.idx_colliding_agents], axis=-1
            ))
            self.col_agent_vel_sum += np.linalg.norm(self._agent_vel)
            print("Col vel", self.col_vel_sum / self.num_col)
            print("Col agent vel", self.col_agent_vel_sum / self.num_col)

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
            print(
                "Col avg max intersection area: ", self.col_inters_sum / self.num_env_col
            )
            print(
                "Col avg, max intersection area rel to agent size:",
                round(
                    (self.col_inters_sum / self.num_env_col) / (np.pi * r_0 ** 2) * 100,
                    2
                ),
                "%"
            )


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
