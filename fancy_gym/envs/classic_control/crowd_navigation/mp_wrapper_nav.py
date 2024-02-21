from typing import Tuple, Union

import numpy as np

from fancy_gym.black_box.raw_interface_wrapper import RawInterfaceWrapper


class MPWrapper_Navigation(RawInterfaceWrapper):

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
        'ProDMP': {
            'phase_generator_kwargs': {
                'tau': 10.,
            },
            'controller_kwargs': {
                'p_gains': 0.6,
                'd_gains': 0.075,
            },
            'basis_generator_kwargs': {
                'num_basis': 3,
            },
            'black_box_kwargs': {
                # one second for dt of 0.1
                'replanning_schedule': lambda pos, vel, obs, action, t: t % 10 == 0
            }
        },
    }

    @property
    def context_mask(self):
        return np.hstack([
            [True] * 4,  # goal position and agent velocity
        ])

    @property
    def current_pos(self) -> Union[float, int, np.ndarray, Tuple]:
        return self.env.current_pos

    @property
    def current_vel(self) -> Union[float, int, np.ndarray, Tuple]:
        return self.env.current_vel
