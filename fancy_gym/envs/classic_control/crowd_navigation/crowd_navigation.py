import matplotlib.pyplot as plt
import numpy as np
from gymnasium import spaces
from gymnasium.core import ObsType

from fancy_gym.envs.classic_control.crowd_navigation.base_crowd_navigation\
    import BaseCrowdNavigationEnv


class CrowdNavigationEnv(BaseCrowdNavigationEnv):
    """
    No real crowd, just obstacles.
    """
    def __init__(
        self,
        n_crowd: int,
        width: int = 20,
        height: int = 20,
        discrete_action: bool = False
    ):
        self.MAX_EPISODE_STEPS = 100
        super().__init__(n_crowd, width, height, allow_collision=False)

        self.discrete_action = discrete_action
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
            [0, 0] * (self.n_crowd + 1),
        ])
        state_bound_max = np.hstack([
            [self.WIDTH, self.HEIGHT] * (self.n_crowd + 1),
            [self.AGENT_MAX_VEL, self.AGENT_MAX_VEL] * (self.n_crowd + 1),
        ])

        self.observation_space = spaces.Box(
            low=state_bound_min, high=state_bound_max, shape=state_bound_min.shape
        )


    def _get_reward(self, action: np.ndarray):
        dg = np.linalg.norm(self._agent_pos - self._goal_pos)
        Rg = np.exp(self.Cg / max(dg, self.PHYSICAL_SPACE)) -\
            np.exp(self.Cg / self.PHYSICAL_SPACE)

        if self._is_collided:
            Rc = self.COLLISION_REWARD
        else:
            dist_crowd = np.linalg.norm(
                self._agent_pos - self._crowd_poss,
                axis=-1
            )
            Rc = np.sum(
                (1 - np.exp(self.Cc / dist_crowd)) *
                (dist_crowd < [self.SOCIAL_SPACE + self.PHYSICAL_SPACE] * self.n_crowd)
            )

        reward = Rg + Rc
        return reward, dict(goal=Rg, collision=Rc)


    def _terminate(self, info):
        return self._is_collided


    def _get_obs(self) -> ObsType:
        rel_crowd_poss = self._crowd_poss - self._agent_pos
        return np.concatenate([
            [self._goal_pos - self._agent_pos],
            rel_crowd_poss if self.n_crowd > 1 else [rel_crowd_poss],
            [self._agent_vel],
            self._crowd_vels
        ]).astype(np.float32).flatten()


    def render(self):
        if self.fig is None:
            # Create base figure once on the beginning. Afterwards only update
            plt.ion()
            self.fig = plt.figure()
            ax = self.fig.add_subplot(1, 1, 1)

            # limits
            ax.set_xlim(-self.W_BORDER - 1, self.W_BORDER + 1)
            ax.set_ylim(-self.H_BORDER - 1, self.H_BORDER + 1)

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

            self.space_agent = plt.Circle(
                self._agent_pos, self.PHYSICAL_SPACE, color="g", alpha=0.5
            )
            ax.add_patch(self.space_agent)
            self.space_crowd = []
            for m in self._crowd_poss:
                self.space_crowd.append(
                    plt.Circle(
                        m, self.PERSONAL_SPACE, color="r", fill=False
                    )
                )
                ax.add_patch(self.space_crowd[-1])
            self.personal_space_crowd = []
            for m in self._crowd_poss:
                self.personal_space_crowd.append(
                    plt.Circle(
                        m, self.PHYSICAL_SPACE, color="r", alpha=0.5
                    )
                )
                ax.add_patch(self.personal_space_crowd[-1])

            self.goal_point, = ax.plot(self._goal_pos[0], self._goal_pos[1], 'gx')

            self.trajectory_line, = ax.plot(
                self.current_trajectory[:, 0],
                self.current_trajectory[:, 1],
                "k",
            )

            ax.axvspan(self.W_BORDER, self.W_BORDER + 100, hatch='.')
            ax.axvspan(-self.W_BORDER - 100, -self.W_BORDER, hatch='.')
            ax.axhspan(self.H_BORDER, self.H_BORDER + 100, hatch='.')
            ax.axhspan(-self.H_BORDER - 100, -self.H_BORDER, hatch='.')
            ax.set_aspect(1.0)

            self.fig.show()

        self.fig.gca().set_title(f"Iteration: {self._steps}")

        if self._steps == 1:
            self.goal_point.set_data(self._goal_pos[0], self._goal_pos[1])

        self.vel_agent.set_data(
            x=self._agent_pos[0], y=self._agent_pos[1],
            dx=self._agent_vel[0], dy=self._agent_vel[1]
        )
        self.space_agent.center = self._agent_pos
        for i, member in enumerate(self._crowd_poss):
            self.space_crowd[i].center = member
            self.personal_space_crowd[i].center = member
        for i in range(self.n_crowd):
            self.vel_crowd[i].set_data(
                x=self._crowd_poss[i][0], y=self._crowd_poss[i][1],
                dx=self._crowd_vels[i][0], dy=self._crowd_vels[i][1]
            )
        self.trajectory_line.set_data(
            self.current_trajectory[:, 0], self.current_trajectory[:, 1]
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
        # crowd_vels = np.linalg.norm(self._crowd_vels, axis=-1)
        # crowd_vels_over = crowd_vels > self.CROWD_MAX_VEL
        # clipped_max_vels = self.CROWD_MAX_VEL / crowd_vels
        # self._crowd_vels[crowd_vels_over] = clipped_max_vels[crowd_vels_over]
        self._crowd_poss += self._crowd_vels * self._dt
        self._crowd_vels *= 1 - (
            np.abs(self._crowd_poss) >
            np.stack([np.array([self.W_BORDER, self.H_BORDER])] * self.n_crowd)
        ).astype(float)
        self._crowd_poss = np.clip(
            self._crowd_poss,
            [-self.W_BORDER, -self.H_BORDER],
            [self.W_BORDER, self.H_BORDER]
        )

        self._is_collided = self._check_collisions()
        reward, info = self._get_reward(action)

        self._steps += 1
        terminated = self._terminate(info)
        truncated = False

        return self._get_obs().copy(), reward, terminated, truncated, info
