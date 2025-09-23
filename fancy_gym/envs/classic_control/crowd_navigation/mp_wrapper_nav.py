from typing import Tuple, Union

import numpy as np

from fancy_gym.black_box.raw_interface_wrapper import RawInterfaceWrapper
from fancy_gym.envs.classic_control.crowd_navigation.dynamics import (
    gen_mat_pos_acc,
    gen_vec_pos_vel,
    gen_mat_vel_acc,
    gen_mat_vc_pos_vel,
    gen_mat_vc_acc_vel
)
from fancy_gym.envs.classic_control.crowd_navigation.utils import REPLAN_NO_CROWD


class MPWrapper_Navigation(RawInterfaceWrapper):
    mp_config = {
        'ProMP': {
            'controller_kwargs': {
                'p_gains': 0.6,
                'd_gains': 0.075,
            },
            "basis_generator_kwargs": {
                'basis_generator_type': 'zero_rbf',
                'num_basis': 4,
                'num_basis_zero_start': 1
            },
            'black_box_kwargs': {
                'replanning_schedule': lambda pos, vel, obs, action, t: t % 10 == 0,
            }
        },
        'DMP': {
            'controller_kwargs': {
                'p_gains': 0.6,
                'd_gains': 0.075,
            },
            'trajectory_generator_kwargs': {
                'weights_scale': 50,
            },
            'phase_generator_kwargs': {
                'alpha_phase': 2,
            },
        },
        'ProDMP': {
            'phase_generator_kwargs': {
                'tau': 6.,  # self._dt * max_episode_steps
            },
            'controller_kwargs': {
                'controller_type': 'mpc',
                'mpc_type': 'linear_plan',
                'horizon': 20,  # 2 sec to stop (1 extra step is current step)
                'dt': 0.1,
                'physical_space': 0.4,
                'const_dist_crowd': 0.81001,
                'agent_max_vel': 1.,
                'agent_max_acc': 10.,
                'crowd_max_vel': 1.5,
                'crowd_max_acc': 10.,
                'n_crowd': 0,
                # 'uncertainty': 'dist',
                # 'horizon_tries': 3,
            },
            'trajectory_generator_kwargs': {
                'weights_scale': 1,
                'goal_scale': 32,
            },
            'black_box_kwargs': {
                # one second for dt of 0.1
                'replanning_schedule': lambda pos, vel, obs, action, t: t %\
                REPLAN_NO_CROWD == 0,
                # 'duration': (21 + 10) * 0.1  # should be at least replan + MPC horizon
            }
        },
    }


    @property
    def context_mask(self):
        return np.hstack([np.full(self.observation_space.shape, True)])


    @property
    def current_pos(self) -> Union[float, int, np.ndarray, Tuple]:
        return self.env.current_pos


    @property
    def current_vel(self) -> Union[float, int, np.ndarray, Tuple]:
        return self.env.current_vel


class MPWrapper_Navigation_Vel(MPWrapper_Navigation):
    mp_config = {
        'ProDMP': {
            'phase_generator_kwargs': {
                'tau': 6.,  # self._dt * max_episode_steps
            },
            'controller_kwargs': {
                'controller_type': 'mpc',
                'mpc_type': 'velocity_control',
                'horizon': 20,  # 2 sec to stop (1 extra step is current step)
                'dt': 0.1,
                'physical_space': 0.4,
                'const_dist_crowd': 0.81001,
                'agent_max_vel': 1.,
                'agent_max_acc': 10.,
                'crowd_max_vel': 1.5,
                'crowd_max_acc': 10.,
                'n_crowd': 0,
                # 'uncertainty': 'dist',
                # 'horizon_tries': 3,
            },
            'trajectory_generator_kwargs': {
                'weights_scale': 1,
                'goal_scale': 32,
            },
            'black_box_kwargs': {
                # one second for dt of 0.1
                'replanning_schedule': lambda pos, vel, obs, action, t: t %\
                REPLAN_NO_CROWD == 0,
                # 'duration': (21 + 10) * 0.1  # should be at least replan + MPC horizon
            }
        },
    }
