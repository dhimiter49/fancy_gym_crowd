import matplotlib.pyplot as plt
import numpy as np
from gymnasium import spaces
from gymnasium.core import ObsType

from fancy_gym.envs.classic_control.crowd_navigation.base_crowd_navigation\
    import BaseCrowdNavigationEnv
from fancy_gym.envs.classic_control.crowd_navigation.utils import REPLAN_STATIC


class LShapeCrowdNavigationEnv(BaseCrowdNavigationEnv):
    def __init__(
        self,
        n_crowd: int = 0,
        width: int = 20,
        height: int = 20,
        interceptor_percentage: float = 0.5,
        allow_collision: bool = False,
        discrete_action: bool = False,
        velocity_control: bool = False,
        lidar_rays: int = 0,
        polar: bool = False,
    ):
        self.MAX_EPISODE_STEPS = 80
        self.polar = polar
        self.replan = REPLAN_STATIC
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
            self.RAY_ANGLES = np.linspace(
                0, 2 * np.pi, self.N_RAYS, endpoint=False
            ) + 1e-6
            self.RAY_COS = np.cos(self.RAY_ANGLES)
            self.RAY_SIN = np.sin(self.RAY_ANGLES)
        if self.lidar:
            if self.polar:
                state_bound_min = np.hstack([
                    [0, -np.pi],
                    [0, -np.pi],
                    [0] * self.N_RAYS,
                ])
                state_bound_max = np.hstack([
                    [max_dist, np.pi],
                    [self.AGENT_MAX_VEL, np.pi],
                    np.full(self.N_RAYS, max_dist)
                ])
            else:
                state_bound_min = np.hstack([
                    [-self.WIDTH, -self.HEIGHT],
                    [-self.AGENT_MAX_VEL, -self.AGENT_MAX_VEL],
                    [0] * self.N_RAYS,
                ])
                state_bound_max = np.hstack([
                    [self.WIDTH, self.HEIGHT],
                    [self.AGENT_MAX_VEL, self.AGENT_MAX_VEL],
                    np.full(self.N_RAYS, max_dist)
                ])
        elif polar:
            state_bound_min = np.hstack([
                [0, -np.pi] * (self.n_crowd + 1),
                [0, -np.pi],
                [0] * 4,  # four directions
            ])
            state_bound_max = np.hstack([
                [max_dist, np.pi] * (self.n_crowd + 1),
                [self.AGENT_MAX_VEL, np.pi],
                [self.MAX_STOPPING_DIST] * 4,  # four directions
            ])
        else:
            state_bound_min = np.hstack([
                [-self.WIDTH, -self.HEIGHT] * (self.n_crowd + 1),
                [-self.AGENT_MAX_VEL, -self.AGENT_MAX_VEL],
                [0] * 4,  # four directions
            ])
            state_bound_max = np.hstack([
                [self.WIDTH, self.HEIGHT] * (self.n_crowd + 1),
                [self.AGENT_MAX_VEL, self.AGENT_MAX_VEL],
                np.repeat([self.WIDTH, self.HEIGHT], 2),  # four directions
            ])

        print(state_bound_max.shape)
        self.observation_space = spaces.Box(
            low=state_bound_min, high=state_bound_max, shape=state_bound_min.shape
        )


    def sample_in_L(self):
        return np.array([
            np.random.uniform(
                [-self.W_BORDER + 2 * self.PHYSICAL_SPACE,
                 -self.H_BORDER + 2 * self.PHYSICAL_SPACE],
                [0 - 2 * self.PHYSICAL_SPACE,
                 self.H_BORDER - 2 * self.PHYSICAL_SPACE]
            ),
            np.random.uniform(
                [0 - 2 * self.PHYSICAL_SPACE,
                 -self.H_BORDER + 2 * self.PHYSICAL_SPACE],
                [self.W_BORDER - 2 * self.PHYSICAL_SPACE,
                 0 - 2 * self.PHYSICAL_SPACE]
            ),
        ])[np.random.choice(2)]


    def _get_obs(self) -> ObsType:
        rel_goal_pos = self._goal_pos - self._agent_pos
        rel_goal_pos = self.c2p(rel_goal_pos) if self.polar else rel_goal_pos
        if self._agent_pos[0] < 0:
            dy, dy_ =\
                self.H_BORDER - self._agent_pos[1], self.H_BORDER + self._agent_pos[1]
            if self._agent_pos[1] < 0:
                dx, dx_ =\
                    self.W_BORDER - self._agent_pos[0], self.W_BORDER + self._agent_pos[0]
            else:
                dx, dx_ =\
                    -self._agent_pos[0], self.W_BORDER + self._agent_pos[0]
        else:
            dx, dx_ =\
                self.W_BORDER - self._agent_pos[0], self.W_BORDER + self._agent_pos[0]
            dy, dy_ =\
                -self._agent_pos[1], self.H_BORDER + self._agent_pos[1]
        dist_walls = np.array([[dx, dx_], [dy, dy_]])
        if self.n_crowd > 0:
            rel_crowd_poss = self._crowd_poss - self._agent_pos
            return np.concatenate([
                [rel_goal_pos],
                rel_crowd_poss if self.n_crowd > 1 else [rel_crowd_poss],
                [self._agent_vel],
                dist_walls
            ]).astype(np.float32).flatten()
        else:
            return np.concatenate([
                [rel_goal_pos],
                [self._agent_vel],
                dist_walls
            ]).astype(np.float32).flatten()



    def step(self, action: np.ndarray):
        """
        A single step with action in angular velocity space
        """
        self.update_state(action)
        self._goal_reached = self.check_goal_reached()
        self._is_collided = self._check_collisions()
        self._current_reward, info = self._get_reward(action)

        self._steps += 1
        terminated = self._terminate(info)
        truncated = False

        return self._get_obs().copy(), self._current_reward, terminated, truncated, info


    def _start_env_vars(self):
        """
        ┌───────┐
        │       │
        │       │
        │       │
        │       └───────┐
        │               │
        │               │
        └───────────────┘
        """
        agent_pos = self.sample_in_L()
        agent_vel = np.zeros(2)
        goal_pos = agent_pos
        while np.linalg.norm(agent_pos - goal_pos) < 2 * self.PERSONAL_SPACE:
            goal_pos = self.sample_in_L()

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
                    sampled_pos = (direction) / 2 + self.rot_mat(rot_deg) @ rand
                    try_between = False
                    if sampled_pos[0] > -self.PHYSICAL_SPACE and\
                        sampled_pos[1] > -self.PHYSICAL_SPACE:  # spawned in first quad
                        continue
                else:
                    sampled_pos = self.sample_in_L()
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


    def render(self):
        if self.fig is None:
            # Create base figure once on the beginning. Afterwards only update
            plt.ion()
            self.fig = plt.figure()
            ax = self.fig.add_subplot(1, 1, 1)

            # limits
            ax.set_xlim(-self.W_BORDER - 1, self.W_BORDER + 1)
            ax.set_ylim(-self.H_BORDER - 1, self.H_BORDER + 1)

            # Agent velocity
            self.vel_agent = ax.arrow(
                self._agent_pos[0], self._agent_pos[1],
                self._agent_vel[0], self._agent_vel[1],
                head_width=self.PERSONAL_SPACE / 4,
                overhang=1,
                head_length=0.2,
                ec="g"
            )

            # Agent
            self.space_agent = plt.Circle(
                self._agent_pos, self.PHYSICAL_SPACE, color="g", alpha=0.5
            )
            ax.add_patch(self.space_agent)

            # Social space
            self.ScS_crowd = []
            for m in self._crowd_poss:
                self.ScS_crowd.append(
                    plt.Circle(
                        m, self.SOCIAL_SPACE, color="r", fill=False, linestyle="--"
                    )
                )
                ax.add_patch(self.ScS_crowd[-1])

            # Personal space
            self.PrS_crowd = []
            for m in self._crowd_poss:
                self.PrS_crowd.append(
                    plt.Circle(
                        m, self.PERSONAL_SPACE, color="r", fill=False
                    )
                )
                ax.add_patch(self.PrS_crowd[-1])

            # Physical space
            self.PhS_crowd = []
            for m in self._crowd_poss:
                self.PhS_crowd.append(
                    plt.Circle(
                        m, self.PHYSICAL_SPACE, color="r", alpha=0.5
                    )
                )
                ax.add_patch(self.PhS_crowd[-1])

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

            ax.axvspan(self.W_BORDER, self.W_BORDER + 100, hatch='.')
            ax.axvspan(-self.W_BORDER - 100, -self.W_BORDER, hatch='.')
            ax.axhspan(self.H_BORDER, self.H_BORDER + 100, hatch='.')
            ax.axhspan(-self.H_BORDER - 100, -self.H_BORDER, hatch='.')
            ax.axhspan(0, self.H_BORDER + 100, xmin=.5, xmax=1, hatch='.')
            ax.set_aspect(1.0)

            # Walls penalization
            border_penalization = self.PHYSICAL_SPACE * 2
            self.border_penalty, = ax.plot(
                [-self.W_BORDER + border_penalization,
                 -self.H_BORDER + border_penalization,
                 - border_penalization,
                 - border_penalization,
                 self.W_BORDER - border_penalization,
                 self.W_BORDER - border_penalization,
                 -self.W_BORDER + border_penalization],
                [-self.H_BORDER + border_penalization,
                 self.H_BORDER - border_penalization,
                 self.H_BORDER - border_penalization,
                 - border_penalization,
                 - border_penalization,
                 -self.H_BORDER + border_penalization,
                 -self.H_BORDER + border_penalization],
                color="r",
                linestyle=":",
                linewidth=0.7
            )

            self.fig.show()

        self.fig.suptitle(f"Iteration: {self._steps}")
        self.fig.gca().set_title(
            f"Reward at this step: {self._current_reward:.4f}",
            fontsize=11,
            fontweight='bold'
        )

        if self._steps == 1:
            self.goal_point.set_data(self._goal_pos[0], self._goal_pos[1])
            for i, member in enumerate(self._crowd_poss):
                self.ScS_crowd[i].center = member
                self.PrS_crowd[i].center = member
                self.PhS_crowd[i].center = member

        self.vel_agent.set_data(
            x=self._agent_pos[0], y=self._agent_pos[1],
            dx=self._agent_vel[0], dy=self._agent_vel[1]
        )
        self.space_agent.center = self._agent_pos
        self.trajectory_line.set_data(
            self.current_trajectory[:, 0], self.current_trajectory[:, 1]
        )
        self.trajectory_line_vel.set_data(
            self.current_trajectory_vel[:, 0], self.current_trajectory_vel[:, 1]
        )

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


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
           np.array([self.W_BORDER, self.H_BORDER]) - self.PHYSICAL_SPACE) or\
           (self._agent_pos[0] > -self.PHYSICAL_SPACE and
           self._agent_pos[1] > - self.PHYSICAL_SPACE) or\
           (self._agent_pos[1] > -self.PHYSICAL_SPACE and
           self._agent_pos[0] > - self.PHYSICAL_SPACE):
            return True
        return False


    def _terminate(self, info) -> bool:
        return self._goal_reached or self._is_collided


    def _get_reward(self, action: np.ndarray):
        if self._goal_reached:
            Rg = self.Tc
        else:
            # Goal distance
            dg = np.linalg.norm(self._agent_pos - self._goal_pos)
            Rg = -self.Cg * dg ** 2

        if self._is_collided:
            Rc = self.COLLISION_REWARD
        else:
            if self.n_crowd > 0:
                # Crowd distance
                dist_crowd = np.linalg.norm(
                    self._agent_pos - self._crowd_poss,
                    axis=-1
                )
                Rc = np.sum(
                    (1 - np.exp(self.Cc / dist_crowd)) * (dist_crowd <
                        [self.SOCIAL_SPACE + self.PHYSICAL_SPACE] * self.n_crowd)
                )
            else:
                Rc = 0

        # Walls, only one of the walls is closer (irrelevant which)
        dist_walls = np.array([
            self.W_BORDER - abs(self._agent_pos[0]) if self._agent_pos[1] < 0
            else min(
                abs(self._agent_pos[0]), self.W_BORDER - abs(self._agent_pos[0])
            ),
            self.H_BORDER - abs(self._agent_pos[1]) if self._agent_pos[0] < 0
            else min(
                abs(self._agent_pos[1]), self.H_BORDER - abs(self._agent_pos[1])
            )
        ])
        Rw = np.sum(
            (1 - np.exp(self.Cc / dist_walls)) *
            (dist_walls < self.PHYSICAL_SPACE * 2)
        )

        reward = Rg + Rc + Rw
        return reward, dict(goal=Rg, collision=Rc, wall=Rw)
