from typing import Union, Tuple

from fancy_gym.black_box.controller.base_controller import BaseController
from qpsolvers import solve_qp
import numpy as np


class MPCController(BaseController):
    """
    A MPC controller that computes the acceleration for each time step given the reference
    positions and velocities. The solution is given by a QP problem that minimizes the
    distance to the reference position and state while upholding the boundaries. The
    optimization is computed for a horizon N and time step dt.

    :param horizon : horizon for which to optimize control
    :param dt : time step
    """

    def __init__(
        self,
        mat_pos_acc: np.ndarray,
        mat_pos_vel: np.ndarray,
        mat_vel_acc: np.ndarray,
        horizon: int = 20,
        dt: float = 0.1,
        control_limit: list = [],
    ):
        self.N = horizon
        self.dt = dt
        self.mat_pos_acc = mat_pos_acc
        self.vec_pos_vel = mat_pos_vel
        self.mat_vel_acc = mat_vel_acc
        self.control_limit = control_limit

    def get_action(self, des_pos, des_vel, c_pos, c_vel):
        actions = np.empty((self.N, 2))
        reference_pos = np.repeat(c_pos, self.N) -\
            np.hstack([des_pos[:self.N, 0], des_pos[:self.N, 1]])
        reference_vel = np.repeat(c_vel, self.N) -\
            np.hstack([des_vel[:self.N, 0], des_vel[:self.N, 1]])
        opt_M = 10 * (self.mat_pos_acc ** 2 + self.mat_vel_acc ** 2)
        opt_V =  (reference_pos + self.vec_pos_vel * np.repeat(c_vel, self.N)) @\
            self.mat_pos_acc + reference_vel @ self.mat_vel_acc
        acc_b_min = np.ones(2 * self.N) * self.control_limit[0]
        acc_b_max = np.ones(2 * self.N) * self.control_limit[1]

        acc = solve_qp(opt_M, opt_V, lb=acc_b_min, ub=acc_b_max, solver="clarabel")
        actions[:, 0] = acc[: self.N]
        actions[:, 1] = acc[self.N :]
        return actions
