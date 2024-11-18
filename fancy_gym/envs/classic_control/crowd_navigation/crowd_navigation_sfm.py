from typing import Tuple, Optional, Any, Dict
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate as interp
from gymnasium import spaces
from gymnasium.core import ObsType

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
        self.inv_relocation_time = 1
        self.inter_strength = 2
        self.inter_range = 1


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
        rel_goal = self._goal_pos - self._agent_pos
        rel_crowd_goal = self._crowd_goal_poss - self._crowd_poss

        dist_to_goal = np.linalg.norm(rel_goal)
        dist_crowd_to_goal = np.linalg.norm(rel_crowd_goal, axis=-1)

        pref_vel = (rel_goal / dist_to_goal) * self.AGENT_MAX_VEL
        pref_crowd_vel = np.einsum(
            "ij,i->ij", rel_crowd_goal, 1 / dist_crowd_to_goal
        ) * self.CROWD_MAX_VEL
        pref_crowd_vel[dist_crowd_to_goal < self.PHYSICAL_SPACE] = np.zeros(2)

        pref_vel = self.inv_relocation_time * (pref_vel - self._agent_vel)
        pref_crowd_vel = self.inv_relocation_time * (pref_crowd_vel - self._crowd_vels)

        # Push force(s) from other agents
        interact_crowd_others = []
        for i, member in enumerate(self._crowd_poss):
            rel_member_others = member - np.concatenate([
                np.delete(self._crowd_poss, i, axis=0),
                self._agent_pos.reshape(1, -1)
            ])  # other crowd members and agent
            dist_member_others = np.linalg.norm(rel_member_others, axis=-1)

            interact_crowd_others.append(np.sum(
                np.einsum(
                    "i,ij->ij",
                    self.inter_strength * np.exp(
                        (2 * self.PHYSICAL_SPACE - dist_member_others) / self.inter_range
                    ),
                    np.einsum("ij,i->ij", rel_member_others, 1 / dist_member_others)
                ),
                axis=0
            ))

        # Sum of push & pull forces
        aggregate_vel = (pref_crowd_vel + np.array(interact_crowd_others)) * self._dt

        # clip the speed so that sqrt(vx^2 + vy^2) <= v_pref
        actions = self._crowd_vels + aggregate_vel
        act_norm = np.linalg.norm(actions, axis=-1)

        over = act_norm > self.AGENT_MAX_VEL
        if np.any(over):
            actions[over] = actions[over] / act_norm[over] * self.AGENT_MAX_VEL

        self._crowd_vels = actions
        self._crowd_poss += self._crowd_vels * self._dt
        return actions
