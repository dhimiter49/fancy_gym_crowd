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


# Velocity control matrices
def gen_mat_vc_pos_vel(horizon, dt):
    """
    Matrix representation of position dependency to velocity control for a 2D environment.
    The last control dimension v_N is explicitly zero.

    Args:
        horizon (int): length of the horizon
        dt (float): the length of the discretized time-step
    Return:
        numpy.ndarray: 2D array with dimension 2 * horizon x 2 * horizon
    """
    MV_xv = scipy.linalg.toeplitz(np.ones(horizon), np.zeros(horizon)) * dt
    np.fill_diagonal(MV_xv, 1 / 2 * dt)
    MV_xv = MV_xv[:, :-1]  # v_N is zero
    MV_xv = np.stack(
        [np.hstack([MV_xv, MV_xv * 0]), np.hstack([MV_xv * 0, MV_xv])]
    ).reshape(2 * horizon, 2 * (horizon - 1))
    return MV_xv


def gen_mat_vc_acc_vel(horizon, dt):
    """
    Vector representation of acceleration dependency to velocity for a 2D environment. The
    acceleration being the differnce between the current and previous velcoity divided by
    the time step.

    Args:
        horizon (int): length of the horizon
        dt (float): the length of the discretized time-step
    Return:
        numpy.ndarray: 1D array with dimension 2 * horizon
    """
    acc_from_vel = np.zeros(horizon)
    acc_from_vel[:2] = np.array([1, -1])
    MV_a = scipy.linalg.toeplitz(acc_from_vel, np.zeros(horizon)) / dt
    MV_a = MV_a[:, :-1]
    MV_a = np.stack(
        [np.hstack([MV_a, MV_a * 0]), np.hstack([MV_a * 0, MV_a])]
    ).reshape(2 * horizon, 2 * (horizon - 1))
    return MV_a
