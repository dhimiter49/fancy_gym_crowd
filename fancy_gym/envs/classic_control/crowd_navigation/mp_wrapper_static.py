from typing import Tuple, Union

import numpy as np
import scipy

from fancy_gym.black_box.raw_interface_wrapper import RawInterfaceWrapper


def gen_mat_pos_acc(horizon, dt):
    M_xa = scipy.linalg.toeplitz(np.arange(horizon, 0, -1) * dt ** 2, np.zeros(horizon))
    M_xa[0] = (3 * horizon - 1) / 6 * dt ** 2
    M_xa[-1] = 1 / 6 * dt ** 2
    M_xa = np.stack(
        [np.hstack([M_xa, M_xa * 0]), np.hstack([M_xa * 0, M_xa])]
    ).reshape(2 * horizon,2 * horizon)
    return M_xa


def gen_vec_pos_vel(horizon, dt):
    M_va = np.hstack([np.arange(1, horizon + 1)] * 2) * dt
    M_va[0] *= 0.5
    M_va[-1] *= 0.5
    return M_va


def gen_vec_vel_acc(horizon, dt):
    return np.ones(horizon * 2) * dt


class MPWrapper_CrowdStatic(RawInterfaceWrapper):

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
        # 'ProDMP': {
        #     'phase_generator_kwargs': {
        #         'tau': 10.,
        #     },
        #     'controller_kwargs': {
        #         'p_gains': 0.6,
        #         'd_gains': 0.075,
        #     },
        #     'basis_generator_kwargs': {
        #         'num_basis': 3,
        #     },
        #     'black_box_kwargs': {
        #         'max_planning_times': 6,
        #         'replanning_schedule': lambda pos, vel, obs, action, t: t % 10 == 0
        #     }
        # },
        'ProDMP': {
            'phase_generator_kwargs': {
                'tau': 10.,
            },
            'controller_kwargs': {
                'controller_type': 'mpc',
                'mat_pos_acc': gen_mat_pos_acc(10, 0.1),
                'mat_pos_vel': gen_vec_pos_vel(10, 0.1),
                'mat_vel_acc': gen_vec_vel_acc(10, 0.1),
                'horizon': 10,
                'dt': 0.1,
                'control_limit': [-0.15, 0.15],
            },
            'basis_generator_kwargs': {
                'num_basis': 4,
            },
            'black_box_kwargs': {
                'max_planning_times': 8,
                'replanning_schedule': lambda pos, vel, obs, action, t: t % 10 == 0
            }
        },
    }

    @property
    def context_mask(self):
        return np.hstack([
            [True] * 2 * (self.n_crowd + 2),  # agent vel, crowd-goal pos
        ])

    @property
    def current_pos(self) -> Union[float, int, np.ndarray, Tuple]:
        return self.env.current_pos

    @property
    def current_vel(self) -> Union[float, int, np.ndarray, Tuple]:
        return self.env.current_vel
