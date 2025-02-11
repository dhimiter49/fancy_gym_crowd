import numpy as np
import socialforce

from fancy_gym.envs.classic_control.crowd_navigation.crowd_navigation\
    import CrowdNavigationEnv


class CrowdNavigationSFMEnv(CrowdNavigationEnv):
    """
    Crowd with SFM policy.

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
        sequence_obs: bool = False,
        const_vel: bool = False,
        one_way: bool = False,
        polar: bool = False,
        time_frame: int = 0,
        lidar_vel: bool = False,
        n_frames: int = 4,
    ):
        super().__init__(
            n_crowd,
            dt,
            width,
            height,
            interceptor_percentage,
            discrete_action=discrete_action,
            velocity_control=velocity_control,
            lidar_rays=lidar_rays,
            sequence_obs=sequence_obs,
            const_vel=const_vel,
            one_way=one_way,
            polar=polar,
            time_frame=time_frame,
            lidar_vel=lidar_vel,
            n_frames=n_frames,
        )
        self.initial_speed = self.CROWD_MAX_VEL
        self.v0 = 10
        self.sigma = 0.3
        self.sim = None


    def _start_env_vars(self):
        agent_pos, agent_vel, goal_pos, crowd_poss, _ = super(
            CrowdNavigationEnv, self
        )._start_env_vars()
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


    def update_crowd(self):
        """
        Create a rvo2 simulation at each time step and run one step

        Agent doesn't stop moving after it reaches the goal,
        because once it stops moving, the reciprocal rule is broken
        """
        # Handle crowd members that reached the goal, a new goal will be generated
        crowd_goal_complete = np.logical_and(
            np.linalg.norm(self._crowd_goal_poss - self._crowd_poss, axis=-1) <
            self.PHYSICAL_SPACE,
            np.linalg.norm(self._crowd_vels, axis=-1) < self.MAX_ACC * self._dt
        )

        if len(crowd_goal_complete) > 0:
            self._crowd_goal_poss[crowd_goal_complete] = self._gen_crowd_goal(
                self._crowd_poss[crowd_goal_complete]
            )

        sf_state = np.concatenate([
            [np.concatenate([self._agent_pos, self._agent_vel, self._goal_pos])],
            np.concatenate(
                [self._crowd_poss, self._crowd_vels, self._crowd_goal_poss], axis=-1
            )
        ])
        sim = socialforce.Simulator(
            sf_state,
            delta_t=self._dt,
            initial_speed=self.initial_speed,
            v0=self.v0,
            sigma=self.sigma
        )
        sim.step()
        actions = sim.state[1:, 2:4]

        self._crowd_vels = actions
        self._crowd_poss += self._crowd_vels * self._dt
        return actions
