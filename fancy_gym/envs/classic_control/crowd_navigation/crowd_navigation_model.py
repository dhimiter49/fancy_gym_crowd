from typing import Tuple, Optional, Any, Dict
import numpy as np
from gymnasium.core import ObsType
import rvo2

from fancy_gym.envs.classic_control.crowd_navigation.crowd_navigation\
    import CrowdNavigationEnv
from fancy_gym.envs.classic_control.crowd_navigation.trp.utils.custom_store \
    import CustomStore
from fancy_gym.envs.classic_control.crowd_navigation.trp.algorithms.pg.pg \
    import PolicyGradient
from fancy_gym.envs.classic_control.crowd_navigation.trp.utils.torch_utils \
    import tensorize, get_numpy


class CrowdNavigationModelEnv(CrowdNavigationEnv):
    """
    Args:
        lidar_rays: number of lidar rays, if 0 no lidar is used
        const_vel: sets the dynamics to using constant velocity
        polar: polar observation and action space
        time_frame: time from which to sample and stack the last frames of obs
        lidar_vel: use a velocity representation for each direction of the lidar
        n_frames: number of frames to stack for lidar, irrelevant if lidar_vel
        avoid_agent_parameter: 1 to avoid completely, lower to increase safety distance
            higher values to avoid the agent less
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
            one_goal=one_goal,
        )
        self.Ci = -1.


    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[ObsType, Dict[str, Any]]:
        if self.lidar:
            self._last_frames *= 0
        obs, info = super().reset(seed=seed, options=options)
        return obs, info


    def set_model(self, path):
        store = CustomStore(
            storage_folder="", note=None, exp_id=path, new=False, mode="a"
        )
        self.model, _ = PolicyGradient.agent_from_data(
            store, train_steps=None, checkpoint_iteration=-1, testing=True
        )
        self.prodmp_env = self.model.sampler.envs_test


    def _start_env_vars(self):
        self.current_seed += 1
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
        next_vels = []
        for i, (agent_pos, goal_pos, agent_vel) in enumerate(zip(
            self._crowd_poss, self._crowd_goal_poss, self._crowd_vels
        )):
            self.prodmp_env.reset()
            crowd_poss = np.concatenate([
                [self._agent_pos],
                np.delete(self._crowd_poss, i, axis=0)
            ])
            crowd_vels = np.concatenate([
                [self._agent_vel],
                np.delete(self._crowd_vels, i, axis=0)
            ])
            crowd_goal_poss = np.concatenate([
                [self._goal_pos],
                np.delete(self._crowd_goal_poss, i, axis=0)
            ])
            self.prodmp_env.venv.envs[0].hard_set_vars(
                {
                    "_agent_pos": agent_pos,
                    "_agent_vel": agent_vel,
                    "_goal_pos": goal_pos,
                    "_crowd_poss": crowd_poss,
                    "_crowd_vels": crowd_vels,
                    "_crowd_goal_poss": crowd_goal_poss,
                }
            )

            prodmp_obs = self.prodmp_env.venv.envs[0].get_obs()
            prodmp_obs = np.concatenate([prodmp_obs, [0]]).flatten()  # time input
            prodmp_obs = tensorize(prodmp_obs, self.model.cpu, self.model.dtype)

            prodmp_weights = self.model.policy(prodmp_obs, train=False)[0]
            prodmp_weights = [get_numpy(prodmp_weights)]

            _, _, _, infos = self.prodmp_env.step(prodmp_weights)
            next_vels.append(infos[0]["step_actions"][0])

        self._crowd_vels = np.array(next_vels)
        self._crowd_poss += self._crowd_vels * self._dt
