import matplotlib.pyplot as plt
import numpy as np
from gymnasium import spaces
from gymnasium.core import ObsType

from fancy_gym.envs.classic_control.crowd_navigation.base_crowd_navigation\
    import BaseCrowdNavigationEnv
from fancy_gym.envs.classic_control.crowd_navigation.utils import replan_close


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
        polar: bool = False,
        sequence_obs: bool = False,
    ):
        self.MAX_EPISODE_STEPS = 60
        self.seq_obs = sequence_obs
        self.polar = polar
        super().__init__(
            0,
            width,
            height,
            discrete_action=discrete_action,
            velocity_control=velocity_control
        )

        if self.polar:
            max_dist = np.linalg.norm(np.array([self.WIDTH, self.HEIGHT]))
            state_bound_min = np.hstack([
                [0, -np.pi],
                [0, -np.pi],
                [0] * 4,  # four directions
            ])
            state_bound_max = np.hstack([
                [max_dist, np.pi],
                [self.AGENT_MAX_VEL, np.pi],
                [self.MAX_STOPPING_DIST] * 4,  # four directions
            ])
        elif self.seq_obs:
            state_bound_min = np.hstack([
                [-self.WIDTH, -self.HEIGHT],
                [-self.WIDTH, -self.HEIGHT],
                [-self.AGENT_MAX_VEL, -self.AGENT_MAX_VEL],
            ])
            state_bound_max = np.hstack([
                [self.WIDTH, self.HEIGHT],
                [self.WIDTH, self.HEIGHT],
                [self.AGENT_MAX_VEL, self.AGENT_MAX_VEL],
            ])
        else:
            state_bound_min = np.hstack([
                [-self.WIDTH, -self.HEIGHT],
                [-self.AGENT_MAX_VEL, -self.AGENT_MAX_VEL],
                [0] * 4,  # four directions
            ])
            state_bound_max = np.hstack([
                [self.WIDTH, self.HEIGHT],
                [self.AGENT_MAX_VEL, self.AGENT_MAX_VEL],
                np.repeat([self.WIDTH, self.HEIGHT], 2),  # four directions
            ])

        self.observation_space = spaces.Box(
            low=state_bound_min, high=state_bound_max, shape=state_bound_min.shape
        )


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
        if self.seq_obs:
            return np.concatenate([
                [self._agent_pos],
                [self._goal_pos],
                [self._agent_vel]
            ]).flatten()
        else:
            rel_goal_pos = self._goal_pos - self._agent_pos
            rel_goal_pos = self.c2p(rel_goal_pos) if self.polar else rel_goal_pos
            agent_vel = self.c2p(self._agent_vel) if self.polar else self._agent_vel
            dist_walls = np.array([
                [self.W_BORDER - self._agent_pos[0], self.W_BORDER + self._agent_pos[0]],
                [self.H_BORDER - self._agent_pos[1], self.H_BORDER + self._agent_pos[1]]
            ])
            return np.concatenate([
                [rel_goal_pos],
                [agent_vel],
                dist_walls
            ]).flatten()


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
        self.trajectory_line_vel.set_data(
            self.current_trajectory_vel[:, 0], self.current_trajectory_vel[:, 1]
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
