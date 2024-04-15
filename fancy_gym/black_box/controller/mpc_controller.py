from fancy_gym.black_box.controller.base_controller import BaseController
from qpsolvers import solve_qp
import numpy as np


def gen_polygon(radius, sides=8):
    def rot_mat(rad):
        return np.array([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]])

    polygon = [[radius, 0]]
    for i in range(1, sides + 1):
        polygon.append(rot_mat(2 * np.pi / sides) @ polygon[i - 1])
    polygon_lines = []
    for i in range(sides):
        # y = mx + b (2D line formula)
        m = (polygon[i][1] - polygon[i + 1][1]) / (polygon[i][0] - polygon[i + 1][0])
        b = polygon[i][1] - m * polygon[i][0]
        polygon_lines.append([m, b])
    return polygon_lines


class MPCController(BaseController):
    """
    A MPC controller that computes the acceleration for each time step given the reference
    positions and velocities. The solution is given by a QP problem that minimizes the
    distance to the reference position and velcotities while fulfilling the constraints.
    The optimization is computed for a horizon N and time step dt.

    :param mat_pos_acc : matrix calculating positions in the horizon from acceleartions
    :param mat_pos_vel : matrix calculating positions from initial velocity
    :param mat_vel_acc : matrix calculating velocities from accelerations
    :param max_acc : maximum acceleration is handled as a non-linear circular constraint
    :param max_vel : maximum velcoity is handled as a non-linear circular constraint
    :param horizon : horizon for which to optimize control
    :param dt : time step
    :param min_dist_crowd : if zero ignore crowd, if given constrain distance to crowd
    """

    def __init__(
        self,
        mat_pos_acc: np.ndarray,
        mat_pos_vel: np.ndarray,
        mat_vel_acc: np.ndarray,
        max_acc: float,
        max_vel: float,
        horizon: int = 20,
        dt: float = 0.1,
        min_dist_crowd: float = 0.0,
    ):
        self.N = horizon
        self.dt = dt
        self.mat_pos_acc = mat_pos_acc
        self.vec_pos_vel = mat_pos_vel
        self.mat_vel_acc = mat_vel_acc
        self.polygon_acc_lines = gen_polygon(max_acc)
        self.polygon_vel_lines = gen_polygon(max_vel)
        self.min_dist_crowd = min_dist_crowd

        self.last_braking_traj = None


    def const_acc_vel(self, const_M, const_b, agent_vel):
        for i, line in enumerate(self.polygon_acc_lines):
            sgn = 1 if i < len(self.polygon_acc_lines) / 2 else -1
            M_a = np.hstack([np.eye(self.N) * -line[0], np.eye(self.N)])
            b_a = np.ones(self.N) * line[1]
            const_M.append(sgn * M_a)
            const_b.append(sgn * b_a)

        for i, line in enumerate(self.polygon_vel_lines):
            sgn = 1 if i < len(self.polygon_vel_lines) / 2 else -1
            M_v = np.hstack([np.eye(self.N) * -line[0], np.eye(self.N)])
            b_v = np.ones(self.N) * line[1] - M_v @ np.repeat(agent_vel, self.N)
            const_M.append(sgn * M_v @ self.mat_vel_acc)
            const_b.append(sgn * b_v)


    def calculate_crowd_positions(self, crowd_poss, crowd_vels):
        """
        Calculate the crowd positions for the next horizon given the constant velocity for
        each member. The formula P_i = p_0 + i * v * dt, where for point i in horizon the
        position will be p_0 + i * v * dt.

        Args:
            crowd_poss (numpy.ndarray): an array of size (n_crowd, 2) with the current
                positions of each member
            crowd_vels (numpy.ndarray): an array of size (n_crowd, 2) with the current
                velocities of each member
        Return:
            (numpy.ndarray): predicted positions of the crowd throughout the horizon
        """
        return np.stack([crowd_poss] * self.N) + np.einsum(
            'ijk,i->ijk',
            np.stack([crowd_vels] * self.N, 0) * self.dt,
            np.arange(0, self.N)
        )


    def const_crowd(self, const_M, const_b, crowd, agent_pos, agent_vel):
        crowd_poss, crowd_vels = crowd
        crowd_poss -= agent_pos  # relative crowd position
        if len(crowd_poss.shape) == 2:  # no positions of during horizon provided
            horizon_crowd_poss = self.calculate_crowd_positions(crowd_poss, crowd_vels)
        else:
            horizon_crowd_poss = crowd_poss
        for member in range(len(horizon_crowd_poss[1])):
            poss = horizon_crowd_poss[:, member, :]
            vec = -poss / np.stack([np.linalg.norm(poss, axis=-1)] * 2, axis=-1)
            M_ca = np.hstack([np.eye(self.N) * vec[:, 0], np.eye(self.N) * vec[:, 1]])
            v_cb = M_ca @ (
                -poss.flatten("F") + self.vec_pos_vel * np.repeat(agent_vel, self.N)
            ) - np.array([self.min_dist_crowd] * self.N)
            M_cac = -M_ca @ self.mat_pos_acc
            const_M.append(M_cac)
            const_b.append(v_cb)


    def get_action(self, des_pos, des_vel, curr_pos, curr_vel, crowd=None):
        actions = np.empty((self.N, 2))
        des_pos = des_pos[:self.N]
        des_vel = des_vel[:self.N]
        reference_pos = np.repeat(curr_pos, self.N) -\
            np.hstack([des_pos[:self.N, 0], des_pos[:self.N, 1]])
        reference_vel = np.repeat(curr_vel, self.N) -\
            np.hstack([des_vel[:self.N, 0], des_vel[:self.N, 1]])

        opt_M = self.mat_pos_acc.T @ self.mat_pos_acc +\
            0.2 * self.mat_vel_acc.T @ self.mat_vel_acc +\
            0.2 * np.eye(2 * self.N)
        opt_V = (reference_pos + self.vec_pos_vel * np.repeat(curr_vel, self.N)).T @\
            self.mat_pos_acc + 0.2 * reference_vel.T @ self.mat_vel_acc

        # constraint matrices and bounds
        const_M = []
        const_b = []

        # constrain distance relative to the crowd
        if self.min_dist_crowd > 0:
            self.const_crowd(const_M, const_b, crowd, curr_pos, curr_vel)

        # constrain acceleration and velocity limits by using an inner polygon of a circle
        self.const_acc_vel(const_M, const_b, curr_vel)

        # constrain safety by ensuring a braking trajectory through a terminal constraint
        term_const_M = self.mat_vel_acc[[self.N - 1, 2 * self.N - 1], :]  # last velocity
        term_const_b = -curr_vel

        acc = solve_qp(
            opt_M, opt_V,
            G=np.vstack(const_M), h=np.hstack(const_b),
            A=term_const_M, b=term_const_b,
            solver="clarabel",
            tol_gap_abs=5e-5,
            tol_gap_rel=5e-5,
            tol_feas=1e-4,
            tol_infeas_abs=5e-5,
            tol_infeas_rel=5e-5,
            tol_ktratio=1e-4
        )

        if acc is None:
            acc = np.zeros(2 * self.N)
            acc[0:self.N - 1] = self.last_braking_traj[1:, 0]
            acc[self.N:2 * self.N - 1] = self.last_braking_traj[1:, 1]
        actions[:, 0] = acc[: self.N]
        actions[:, 1] = acc[self.N:]
        self.last_braking_traj = actions  # execute on net step if something goes wrong
        return actions
