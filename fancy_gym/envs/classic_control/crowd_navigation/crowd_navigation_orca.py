from typing import Tuple, Optional, Any, Dict
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate as interp
from gymnasium import spaces
from gymnasium.core import ObsType
import rvo2

from fancy_gym.envs.classic_control.crowd_navigation.crowd_navigation\
    import CrowdNavigationEnv


class CrowdNavigationORCAEnv(CrowdNavigationEnv):
    """
    Crowd with ORCA policy.

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
        self.neighbor_dist = self.PHYSICAL_SPACE * 6 + 0.1
        self.safety_space = self.PHYSICAL_SPACE / 2
        self.time_horizon = self.MAX_STOPPING_TIME * 8
        self.time_horizon_obst = self.MAX_STOPPING_TIME
        self._start_sim()


    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[ObsType, Dict[str, Any]]:
        if self.lidar:
            self._last_frames *= 0
        obs, info = super().reset(seed=seed, options=options)
        self._start_sim()
        return obs, info


    def _start_sim(self):
        max_neighbors = self.n_crowd
        params = (
            self.neighbor_dist, max_neighbors, self.time_horizon, self.time_horizon_obst
        )
        self.sim = rvo2.PyRVOSimulator(
            self._dt, *params, self.PHYSICAL_SPACE, self.CROWD_MAX_VEL
        )
        self.sim.addAgent(
            tuple(self._agent_pos),
            *params,
            self.PHYSICAL_SPACE + self.safety_space,
            self.AGENT_MAX_VEL,
            tuple(self._agent_vel)
        )
        for pos, vel in zip(self._crowd_poss, self._crowd_vels):
            self.sim.addAgent(
                tuple(pos),
                *params,
                self.PHYSICAL_SPACE + self.safety_space,
                self.CROWD_MAX_VEL,
                tuple(vel)
            )


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
        self.sim.setAgentPosition(0, tuple(self._agent_pos))
        self.sim.setAgentVelocity(0, tuple(self._agent_vel))
        for i, (pos, vel) in enumerate(zip(self._crowd_poss, self._crowd_vels)):
            self.sim.setAgentPosition(i + 1, tuple(pos))
            self.sim.setAgentVelocity(i + 1, tuple(vel))

        # Set the preferred velocity to be a vector of unit magnitude (speed) in the
        # direction of the goal.
        velocity = np.array(self._goal_pos - self._agent_vel)
        speed = np.linalg.norm(velocity)
        pref_vel = velocity / speed if speed > 1 else velocity
        self.sim.setAgentPrefVelocity(0, tuple(pref_vel))

        crowd_pref_vels = self._crowd_goal_poss - self._crowd_poss
        crowd_pref_vels[
            np.linalg.norm(crowd_pref_vels, axis=-1) < self.PHYSICAL_SPACE
        ] = 0
        crowd_pref_vels_speed = np.linalg.norm(crowd_pref_vels, axis=-1)

        # update crowd goals
        crowd_goal_complete = np.logical_and(
            crowd_pref_vels_speed < self.PHYSICAL_SPACE,
            np.linalg.norm(self._crowd_vels, axis=-1) < self.MAX_ACC * self._dt
        )
        if len(crowd_goal_complete) > 0:
            self._crowd_goal_poss[crowd_goal_complete] = self._gen_crowd_goal(
                self._crowd_poss[crowd_goal_complete]
            )
            crowd_pref_vels = self._crowd_goal_poss - self._crowd_poss
            crowd_pref_vels_speed = np.linalg.norm(crowd_pref_vels, axis=-1)

        diff_vel = crowd_pref_vels - self._crowd_vels
        diff_speed = np.linalg.norm(diff_vel, axis=-1)

        over = diff_speed > self.MAX_ACC * self._dt
        under = diff_speed < -self.MAX_ACC * self._dt
        if np.any(over):
            crowd_pref_vels[over] = self._crowd_vels[over] + np.einsum(
                "ij,i->ij", diff_vel[over], 1 / diff_speed[over]
            ) * self.MAX_ACC * self._dt
        if np.any(under):
            crowd_pref_vels[under] = self._crowd_vels[under] - np.einsum(
                "ij,i->ij", diff_vel[under], 1 / diff_speed[under]
            ) * self.MAX_ACC * self._dt
        crowd_pref_vels_speed = np.linalg.norm(crowd_pref_vels, axis=-1)

        over_vel = crowd_pref_vels_speed > self.CROWD_MAX_VEL
        crowd_pref_vels[over_vel] = np.einsum(
            "ij,i->ij", crowd_pref_vels[over_vel], 1 / crowd_pref_vels_speed[over_vel]
        ) * self.CROWD_MAX_VEL
        for i in range(self.n_crowd):
            self.sim.setAgentPrefVelocity(i + 1, tuple(crowd_pref_vels[i]))


        self.sim.doStep()

        actions = np.empty((self.n_crowd, 2))
        for i in range(self.n_crowd):
            actions[i] = np.array(self.sim.getAgentVelocity(i + 1))

        self._crowd_vels = actions
        self._crowd_poss += self._crowd_vels * self._dt
        return actions
