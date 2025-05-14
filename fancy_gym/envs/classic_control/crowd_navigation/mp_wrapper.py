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
from fancy_gym.envs.classic_control.crowd_navigation.utils import REPLAN_MOVING


class MPWrapper_Crowd(RawInterfaceWrapper):
    pd_control = {
        'phase_generator_kwargs': {
            'tau': 10.,
        },
        'controller_kwargs': {
            'p_gains': 5,
            'd_gains': 10,
        },
        'trajectory_generator_kwargs': {
            'weights_scale': 1,
            'goal_scale': 2,
        },
        'basis_generator_kwargs': {
            'num_basis': 4,
        },
        'black_box_kwargs': {
            'replanning_schedule': lambda pos, vel, obs, action, t: t % REPLAN_MOVING == 0
        }
    }
    mpc_control = {
        'phase_generator_kwargs': {
            'tau': 10.,  # self._dt * max_episode_steps
        },
        'controller_kwargs': {
            'controller_type': 'mpc',
            'mat_pos_acc': gen_mat_pos_acc(21, 0.1),
            'mat_pos_vel': gen_vec_pos_vel(21, 0.1),
            'mat_vel_acc': gen_mat_vel_acc(21, 0.1),
            'max_acc': 1.5,
            'max_vel': 3.0,
            'horizon': 21,  # 2 sec to stop (1 extra step is current step)
            'dt': 0.1,
            'min_dist_crowd': 0.81001,  # personal space of the members of the crowd
            'min_dist_wall': 0.41,  # physical space of agent + 0.01
        },
        'trajectory_generator_kwargs': {
            'weights_scale': 1,
            'goal_scale': 2,
        },
        'basis_generator_kwargs': {
            'num_basis': 4,
        },
        'black_box_kwargs': {
            # one second for dt of 0.1
            'replanning_schedule': lambda pos, vel, obs, action, t: t % REPLAN_MOVING == 0
            # 'duration': (21 + 10) * 0.1  # should be at least replan + MPC horizon
        }
    }
    mp_config = {
        'ProMP': {
            'controller_kwargs': {
                'p_gains': 0.6,
                'd_gains': 0.075,
            },
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
        'ProDMP': mpc_control
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


class MPWrapper_Crowd_Vel(MPWrapper_Crowd):
    velocity_control = {
        'phase_generator_kwargs': {
            'tau': 10.,
        },
        'controller_kwargs': {
            'controller_type': 'velocity',
        },
        'trajectory_generator_kwargs': {
            'weights_scale': 1,
            'goal_scale': 2,
        },
        'basis_generator_kwargs': {
            'num_basis': 4,
        },
        'black_box_kwargs': {
            'replanning_schedule': lambda pos, vel, obs, action, t: t % REPLAN_MOVING == 0
            # 'duration': 10.
        }
    }
    mpc_control = {
        'phase_generator_kwargs': {
            'tau': 10.,  # self._dt * max_episode_steps
        },
        'controller_kwargs': {
            'controller_type': 'mpc',
            'mat_vc_pos_vel': gen_mat_vc_pos_vel(21, 0.1),
            'mat_vc_acc_vel': gen_mat_vc_acc_vel(21, 0.1),
            'max_acc': 1.5,
            'max_vel': 3.0,
            'horizon': 21,  # 2 sec to stop (1 extra step is current step)
            'dt': 0.1,
            'velocity_control': True,
            'min_dist_crowd': 0.81001,  # personal space of the members of the crowd
            'min_dist_wall': 0.41,  # physical space of agent + 0.01
        },
        'trajectory_generator_kwargs': {
            'weights_scale': 1,
            'goal_scale': 2,
        },
        'basis_generator_kwargs': {
            'num_basis': 4,
        },
        'black_box_kwargs': {
            # one second for dt of 0.1
            "replanning_schedule": lambda pos, vel, obs, action, t: t % REPLAN_MOVING\
            == 0,
            # 'duration': (21 + 10) * 0.1  # should be at least replan + MPC horizon
        }
    }
    mp_config = {'ProDMP': mpc_control}
