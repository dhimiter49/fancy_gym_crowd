import matplotlib.pyplot as plt
import numpy as np
from gymnasium import spaces
from gymnasium.core import ObsType

from fancy_gym.envs.classic_control.crowd_navigation.base_crowd_navigation\
    import BaseCrowdNavigationEnv


class NavigationEnv(BaseCrowdNavigationEnv):
    """
    No real crowd, just obstacles.
    """
    def __init__(
        self,
        width: int = 20,
        height: int = 20,
        discrete_action: bool = False,
        velocity_control: bool = False,
    ):
        self.MAX_EPISODE_STEPS = 60
        super().__init__(0, width, height, allow_collision=False)

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
            [-self.WIDTH, -self.HEIGHT],
            [0, 0],
            [0] * 4,  # four directions
        ])
        state_bound_max = np.hstack([
            [self.WIDTH, self.HEIGHT],
            [self.AGENT_MAX_VEL, self.AGENT_MAX_VEL],
            [self.MAX_STOPPING_DIST] * 4,  # four directions
        ])

        self.observation_space = spaces.Box(
            low=state_bound_min, high=state_bound_max, shape=state_bound_min.shape
        )


    def _start_env_vars(self):
        agent_pos, agent_vel, goal_pos, crowd_poss, _ = super()._start_env_vars()
        return agent_pos, agent_vel, goal_pos, crowd_poss, np.zeros(crowd_poss.shape)


    def _get_reward(self, action: np.ndarray):
        if self._goal_reached:
            Rg = self.Tc
        else:
            # Goal distance
            dg = np.linalg.norm(self._agent_pos - self._goal_pos)
            Rg = -self.Cg * dg ** 2

        if self._is_collided:
            Rw = self.COLLISION_REWARD
        else:
            # Walls, only one of the walls is closer (irrelevant which)
            dist_walls = np.array([
                self.W_BORDER - abs(self._agent_pos[0]),
                self.H_BORDER - abs(self._agent_pos[1]),
            ])
            Rw = np.sum(
                (1 - np.exp(self.Cc / dist_walls)) *
                (dist_walls < self.PHYSICAL_SPACE * 2)
            )

        return Rg + Rw, dict(goal=Rg, wall=Rw)


    def _terminate(self, info) -> bool:
        return self._goal_reached or self._is_collided


    def _get_obs(self) -> ObsType:
        dist_walls = np.clip(np.array([
            [self.W_BORDER - self._agent_pos[0], self.W_BORDER + self._agent_pos[0]],
            [self.H_BORDER - self._agent_pos[1], self.H_BORDER + self._agent_pos[1]]
        ]), 0, self.MAX_STOPPING_DIST)
        return np.concatenate([
            [self._goal_pos - self._agent_pos],
            [self._agent_vel],
            dist_walls
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
            self.space_agent = plt.Circle(
                self._agent_pos, self.PHYSICAL_SPACE, color="g", alpha=0.5
            )
            ax.add_patch(self.space_agent)

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

        self.fig.gca().set_title(f"Iteration: {self._steps}")

        if self._steps == 1:
            self.goal_point.set_data(self._goal_pos[0], self._goal_pos[1])

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
        reward, info = self._get_reward(action)

        self._steps += 1
        terminated = self._terminate(info)
        truncated = False

        return self._get_obs().copy(), reward, terminated, truncated, info
