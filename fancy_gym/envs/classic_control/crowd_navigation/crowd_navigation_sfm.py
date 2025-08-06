from typing import Callable
import numpy as np
import torch
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
        lidar_max: float = 0.0,
        intrinsic_rew: bool = False,
        curriculum: Callable = lambda _: 8,
        one_goal: bool = False,
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
            lidar_max=lidar_max,
            intrinsic_rew=intrinsic_rew,
            curriculum=curriculum,
            one_goal=one_goal,
        )
        self.Ci = -1.
        self.initial_speed = self.CROWD_MAX_VEL
        self.ped_ped = socialforce.potentials.PedPedPotential(
            v0=5, sigma=1.
        )
        self.ped_space = socialforce.potentials.PedSpacePotential(
            [], r=self.PHYSICAL_SPACE[0] * 2
        )
        self.ped_ped.delta_t_step = 0.1
        self.sim = socialforce.Simulator(
            ped_space=self.ped_space,
            ped_ped=self.ped_ped,
            # delta_t=0.1,
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
        # Handle crowd members that reached the goal, a new goal will be generated
        if not self.one_goal and not self.run_test_case:
            crowd_goal_complete = np.logical_and(
                np.linalg.norm(self._crowd_goal_poss - self._crowd_poss, axis=-1) <
                self.PHYSICAL_SPACE[1:1 + self.n_crowd],
                np.linalg.norm(self._crowd_vels, axis=-1) < self.MAX_ACC * self._dt
            )

            if len(crowd_goal_complete) > 0:
                self._crowd_goal_poss[crowd_goal_complete] = self._gen_crowd_goal(
                    self._crowd_poss[crowd_goal_complete]
                )
        agent_pref_vel = self._goal_pos - self._agent_pos
        agent_pref_vel /= np.linalg.norm(agent_pref_vel) * self.AGENT_MAX_VEL
        agent_pref_acc = (agent_pref_vel - self._agent_vel) / self._dt
        agent_pref_acc_norm = np.linalg.norm(agent_pref_acc)
        if agent_pref_acc_norm > self.MAX_ACC:
            agent_pref_vel = self._agent_vel + agent_pref_acc / agent_pref_acc_norm *\
                self.MAX_ACC * self._dt

        crowd_pref_vels = self._crowd_goal_poss - self._crowd_poss
        crowd_pref_vels = np.einsum(
            "ij,i->ij",
            crowd_pref_vels,
            1 / np.linalg.norm(crowd_pref_vels, axis=-1)
        ) * self.CROWD_MAX_VEL
        crowd_pref_accs = (crowd_pref_vels - self._crowd_vels) / self._dt
        crowd_pref_accs_norm = np.linalg.norm(crowd_pref_accs, axis=-1)
        idxs_acc_too_high = np.where(crowd_pref_accs_norm > self.MAX_ACC)[0]
        if len(idxs_acc_too_high) > 0:
            crowd_pref_vels[idxs_acc_too_high] = self._crowd_vels[idxs_acc_too_high] +\
                np.einsum(
                    'ij,i->ij',
                    crowd_pref_accs[idxs_acc_too_high],
                    1 / crowd_pref_accs_norm[idxs_acc_too_high] * self.MAX_ACC * self._dt
            )
        sf_state = np.concatenate([
            [np.concatenate([
                self._agent_pos,
                agent_pref_vel,
                self._goal_pos
            ])],
            np.concatenate([
                self._crowd_poss,
                crowd_pref_vels,
                self._crowd_goal_poss
            ], axis=-1)
        ])
        new_state = self.sim(torch.from_numpy(sf_state))
        actions = new_state[1:, 2:4].detach().numpy()

        self._crowd_vels = actions.copy()
        self._crowd_poss += self._crowd_vels * self._dt
        return actions
