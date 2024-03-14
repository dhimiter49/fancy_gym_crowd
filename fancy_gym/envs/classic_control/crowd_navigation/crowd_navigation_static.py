import matplotlib.pyplot as plt
import numpy as np
from gymnasium import spaces
from gymnasium.core import ObsType

from fancy_gym.envs.classic_control.crowd_navigation.base_crowd_navigation\
    import BaseCrowdNavigationEnv


class CrowdNavigationStaticEnv(BaseCrowdNavigationEnv):
    """
    No real crowd, just obstacles.
    """
    def __init__(
        self,
        n_crowd: int,
        width: int = 20,
        height: int = 20,
        interceptor_percentage: float = 0.5,
        discrete_action : bool = False,
    ):
        self.MAX_EPISODE_STEPS = 100
        super().__init__(
            n_crowd, width, height, interceptor_percentage,  allow_collision=False
        )

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
            [0, 0],
        ])
        state_bound_max = np.hstack([
            [self.WIDTH, self.HEIGHT] * (self.n_crowd + 1),
            [self.AGENT_MAX_VEL, self.AGENT_MAX_VEL],
        ])

        self.observation_space = spaces.Box(
            low=state_bound_min, high=state_bound_max, shape=state_bound_min.shape
        )


    def _start_env_vars(self):
        agent_pos, agent_vel, goal_pos, crowd_poss, _ = super()._start_env_vars()
        return agent_pos, agent_vel, goal_pos, crowd_poss, np.zeros(crowd_poss.shape)


    def _get_reward(self, action: np.ndarray):
        dg = np.linalg.norm(self._agent_pos - self._goal_pos)
        if self._goal_reached:
            Rg = self.Tc
        else:
            # Goal distance
            Rg = -self.Cg * dg

        if self._is_collided:
            Rc = self.COLLISION_REWARD
        else:
            # Crowd distance
            dist_crowd = np.linalg.norm(
                self._agent_pos - self._crowd_poss,
                axis=-1
            )
            Rc = np.sum(
                (1 - np.exp(self.Cc / dist_crowd)) * \
                (dist_crowd < [self.SOCIAL_SPACE + self.PHYSICAL_SPACE] * self.n_crowd)
            )

        # Walls
        min_wall_distance = min(self._agent_pos[0] + self.WIDTH / 2, self.WIDTH / 2 - self._agent_pos[0], self._agent_pos[1] + self.HEIGHT / 2, self.HEIGHT / 2 - self._agent_pos[1])
        Rw = np.where(min_wall_distance < self.PHYSICAL_SPACE * 2, (1 - np.exp(self.Cc / min_wall_distance)), 0)

        # Stalling reward
        Rs = max(dg - self.max_stopping_distance, 0) / np.sqrt(self.WIDTH ** 2 + self.HEIGHT ** 2) * \
            (np.linalg.norm(action) - self.MAX_ACC) / self.MAX_ACC

        reward = Rg + Rc + Rs + Rw
        return reward, dict(goal=Rg, collision=Rc, stalling=Rs)


    def _terminate(self, info):
        return self._is_collided or self._goal_reached


    def _get_obs(self) -> ObsType:
        rel_crowd_poss = self._crowd_poss - self._agent_pos
        return np.concatenate([
            [self._goal_pos - self._agent_pos],
            rel_crowd_poss if self.n_crowd > 1  else [rel_crowd_poss],
            [self._agent_vel]
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

            # Walls
            ax.axvspan(self.W_BORDER, self.W_BORDER + 100, hatch='.')
            ax.axvspan(-self.W_BORDER - 100, -self.W_BORDER,hatch='.')
            ax.axhspan(self.H_BORDER, self.H_BORDER + 100, hatch='.')
            ax.axhspan(-self.H_BORDER - 100, -self.H_BORDER, hatch='.')
            ax.set_aspect(1.0)

            # Walls penalization
            border_penalization = self.PHYSICAL_SPACE * 2
            ax.add_patch(plt.Rectangle((-self.W_BORDER + border_penalization, -self.H_BORDER + border_penalization),
                           2 * (self.W_BORDER - border_penalization), 2 * (self.H_BORDER - border_penalization),
                           fill=False, linestyle=":", edgecolor="r", linewidth=0.7))

            self.fig.show()

        self.fig.suptitle(f"Iteration: {self._steps}")
        self.fig.gca().set_title(f"Reward at this step: {self._current_reward:.4f}", fontsize=11, fontweight='bold')

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

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


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
