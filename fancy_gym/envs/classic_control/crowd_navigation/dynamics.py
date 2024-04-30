import numpy as np
import scipy


# Acceleration control matrices
def gen_mat_pos_acc(horizon, dt):
    """
    Matrix representation of position dependency to acceleration control for a 2D enviro-
    nment.

    Args:
        horizon (int): length of the horizon
        dt (float): the length of the discretized time-step
    Return:
        numpy.ndarray: 2D array with dimension 2 * horizon x 2 * horizon
    """
    M_xa = scipy.linalg.toeplitz(
        np.array([(2 * i - 1) / 2 * dt ** 2 for i in range(1, horizon + 1)]),
        np.zeros(horizon)
    )
    M_xa = np.stack(
        [np.hstack([M_xa, M_xa * 0]), np.hstack([M_xa * 0, M_xa])]
    ).reshape(2 * horizon, 2 * horizon)
    return M_xa


def gen_vec_pos_vel(horizon, dt):
    """
    Vector representation of position dependency to initial velocity for a 2D environment.

    Args:
        horizon (int): length of the horizon
        dt (float): the length of the discretized time-step
    Return:
        numpy.ndarray: 1D array with dimension 2 * horizon
    """
    M_xv = np.hstack([np.arange(1, horizon + 1)] * 2) * dt
    return M_xv


def gen_mat_vel_acc(horizon, dt):
    """
    Matrix representation of velocity to acceleration control for a 2D environment.

    Args:
        horizon (int): length of the horizon
        dt (float): the length of the discretized time-step
    Return:
        numpy.ndarray: 2D array with dimension 2 * horizon x 2 * horizon
    """
    M_va = scipy.linalg.toeplitz(np.ones(horizon) * dt, np.zeros(horizon))
    M_va = np.stack(
        [np.hstack([M_va, M_va * 0]), np.hstack([M_va * 0, M_va])]
    ).reshape(2 * horizon, 2 * horizon)
    return M_va
