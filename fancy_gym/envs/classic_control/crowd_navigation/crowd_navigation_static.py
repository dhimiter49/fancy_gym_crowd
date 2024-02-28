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
        discrete_action : bool = False
    ):
        super().__init__(n_crowd, width, height, allow_collision=False)

        self.CROWD_MAX_VEL = 0
        self.REW_TASK_COEFF = 10
        self.REW_GOAL_POS_COEFF = 1
        self.REW_COLLISION_COEFF = -10
        self.REW_GOAL_DIST_COEFF = -2 * self.REW_GOAL_POS_COEFF / np.linalg.norm(
            np.array([self.WIDTH, self.HEIGHT])
        ) / self.MAX_EPISODE_STEPS
        self.REW_COLLISION_DIST_COEFF = 2 * self.REW_COLLISION_COEFF / np.linalg.norm(
            np.array([self.WIDTH, self.HEIGHT])
        ) / self.MAX_EPISODE_STEPS

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

        action_bound = np.array([self.MAX_ACC, np.pi])
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
        dist_goal = np.linalg.norm(self._agent_pos - self._goal_pos)
        if self._goal_reached:
            rew_dist = self.REW_TASK_COEFF
        elif dist_goal < self.PERSONAL_SPACE / 4:
            rew_dist = self.REW_GOAL_POS_COEFF - np.linalg.norm(self._agent_vel)
        else:
            rew_dist = self.REW_GOAL_DIST_COEFF * dist_goal ** 2

        if self._is_collided:
            rew_collision = self.REW_COLLISION_COEFF
        else:
            dist_crowd = np.linalg.norm(
                self._agent_pos - self._crowd_poss,
                axis=-1
            ) - self.PERSONAL_SPACE * 3
            rew_collision = self.REW_COLLISION_DIST_COEFF * np.sum(
                dist_crowd ** 2 * (dist_crowd < [0] * self.n_crowd)
            )

        # print(dict(dist_goal=rew_dist, collision=rew_collision))

        reward = rew_dist + rew_collision
        return reward, dict(
            dist_goal=rew_dist,
            collision=rew_collision,
        )


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

            self.pos_agent, = ax.plot(
                self._agent_pos[0],
                self._agent_pos[1],
                "go", markersize=4, ls='',
            )
            self.vel_agent = ax.arrow(
                self._agent_pos[0], self._agent_pos[1],
                self._agent_vel[0], self._agent_vel[1],
                head_width=self.PERSONAL_SPACE / 4,
                overhang=1,
                head_length=0.2,
                ec="g"
            )
            self.pos_crowd, = ax.plot(
                self._crowd_poss[:, 0],
                self._crowd_poss[:, 1],
                "ro", markersize=4, ls='',
            )
            self.space_agent = plt.Circle(
                self._agent_pos, self.PERSONAL_SPACE, color="g", fill=False
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
            self.goal_point, = ax.plot(self._goal_pos[0], self._goal_pos[1], 'gx')

            ax.axvspan(self.W_BORDER, self.W_BORDER + 100, hatch='.')
            ax.axvspan(-self.W_BORDER - 100, -self.W_BORDER,hatch='.')
            ax.axhspan(self.H_BORDER, self.H_BORDER + 100, hatch='.')
            ax.axhspan(-self.H_BORDER - 100, -self.H_BORDER, hatch='.')
            ax.set_aspect(1.0)

            self.fig.show()

        self.fig.gca().set_title(f"Iteration: {self._steps}")

        if self._steps == 1:
            self.goal_point.set_data(self._goal_pos[0], self._goal_pos[1])
            self.pos_crowd.set_data(self._crowd_poss[:, 0], self._crowd_poss[:, 1])
            for i, member in enumerate(self._crowd_poss):
                self.space_crowd[i].center = member

        self.pos_agent.set_data(self._agent_pos[0], self._agent_pos[1])
        self.vel_agent.set_data(
            x=self._agent_pos[0], y=self._agent_pos[1],
            dx=self._agent_vel[0], dy=self._agent_vel[1]
        )
        self.space_agent.center = self._agent_pos

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


    def step(self, action: np.ndarray):
        """
        A single step with action in angular velocity space
        """
        self.update_state(action)
        self._is_collided = self._check_collisions()
        self._goal_reached = self.check_goal_reached()
        reward, info = self._get_reward(action)

        self._steps += 1
        terminated = self._terminate(info)
        truncated = False

        return self._get_obs().copy(), reward, terminated, truncated, info
