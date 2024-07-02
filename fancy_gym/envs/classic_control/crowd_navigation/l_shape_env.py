import matplotlib.pyplot as plt
import numpy as np
from gymnasium.core import ObsType

from fancy_gym.envs.classic_control.crowd_navigation.base_crowd_navigation\
    import BaseCrowdNavigationEnv


class LShapeCrowdNavigationEnv(BaseCrowdNavigationEnv):
    def __init__(
        self,
        width: int = 20,
        height: int = 20,
        interceptor_percentage: float = 0.5,
        allow_collision: bool = False,
        discrete_action: bool = False,
        velocity_control: bool = False,
    ):
        self.MAX_EPISODE_STEPS = 60
        super().__init__(
            0,
            width,
            height,
            interceptor_percentage,
            allow_collision,
            discrete_action,
            velocity_control,
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
        return np.concatenate([
            [self._goal_pos - self._agent_pos],
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
        reward, info = self._get_reward(action)

        self._steps += 1
        terminated = self._terminate(info)
        truncated = False

        return self._get_obs().copy(), reward, terminated, truncated, info


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
            ax.axhspan(0, self.H_BORDER + 100, xmin=.5, xmax=1, hatch='.')
            ax.set_aspect(1.0)

            # Walls penalization
            border_penalization = self.PHYSICAL_SPACE * 2
            # ax.add_patch(plt.Rectangle(
            #     (
            #         -self.W_BORDER + border_penalization,
            #         -self.H_BORDER + border_penalization
            #     ),
            #     2 * (self.W_BORDER - border_penalization),
            #     2 * (self.H_BORDER - border_penalization),
            #     fill=False, linestyle=":", edgecolor="r", linewidth=0.7
            # ))
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
            Rw = self.COLLISION_REWARD
        else:
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

        return Rg + Rw, dict(goal=Rg, wall=Rw)
