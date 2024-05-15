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
    return np.array(polygon_lines)


class MPCController(BaseController):
    """
    A MPC controller for 2D navigation that computes the acceleration or velocity for each
    time step given the reference positions and velocities. The solution is given by a QP
    problem that minimizes the distance to the reference position and velcotities while
    fulfilling the constraints. The optimization is computed for a horizon N and time step
    dt.

    :param max_acc : maximum acceleration is handled as a non-linear circular constraint
    :param max_vel : maximum velcoity is handled as a non-linear circular constraint
    :param mat_pos_acc : matrix calculating positions in the horizon from acceleartions
    :param mat_pos_vel : matrix calculating positions from initial velocity
    :param mat_vel_acc : matrix calculating velocities from accelerations
    :param mat_vc_pos_vel : matrix calculating positions from velocity control
    :param mat_vc_acc_vel : matrix calculating acceleration from velocities
    :param horizon : horizon for which to optimize control
    :param dt : time step
    :param min_dist_crowd : if zero ignore crowd, if given constrain distance to crowd
    """

    def __init__(
        self,
        max_acc: float,
        max_vel: float,
        mat_pos_acc: np.ndarray = None,
        mat_pos_vel: np.ndarray = None,
        mat_vel_acc: np.ndarray = None,
        mat_vc_pos_vel: np.ndarray = None,
        mat_vc_acc_vel: np.ndarray = None,
        horizon: int = 20,
        dt: float = 0.1,
        min_dist_crowd: float = 0.0,
        velocity_control: float = False,
    ):
        self.N = horizon
        self.MAX_STOPPING_TIME = max_vel / max_acc
        self.MAX_STOPPING_DIST = max_vel * self.MAX_STOPPING_TIME -\
            0.5 * max_acc * self.MAX_STOPPING_TIME ** 2
        self.dt = dt
        self.velocity_control = velocity_control
        self.mat_pos_acc = mat_pos_acc
        self.vec_pos_vel = mat_pos_vel
        self.mat_vel_acc = mat_vel_acc
        self.mat_vc_pos_vel = mat_vc_pos_vel
        self.mat_vc_acc_vel = mat_vc_acc_vel
        if self.velocity_control:
            self.mat_pos_control = self.mat_vc_pos_vel
            self.vec_pos_vel = 0.5 * self.dt
        else:
            self.mat_pos_control = self.mat_pos_acc
        self.polygon_acc_lines = gen_polygon(max_acc)
        self.polygon_vel_lines = gen_polygon(max_vel)
        self.min_dist_crowd = min_dist_crowd

        if not self.velocity_control:
            M_v_ = np.vstack([
                np.eye(self.N) * -line[0] for line in self.polygon_vel_lines
            ])
            M_v_ = np.hstack([
                M_v_, np.vstack([np.eye(self.N)] * len(self.polygon_vel_lines))
            ])
            sgn_vel = np.ones(len(self.polygon_vel_lines))
            sgn_vel[len(self.polygon_vel_lines) // 2:] = -1
            sgn_vel = np.repeat(sgn_vel, self.N)
            b_v_ = np.repeat(self.polygon_vel_lines[:, 1], self.N)

            self.vel_mat_constraint = ((M_v_ @ self.mat_vel_acc).T * sgn_vel).T
            self.vel_vec_constraint = lambda agent_vel: sgn_vel *\
                (b_v_ - M_v_ @ np.repeat(agent_vel, self.N))

            M_a_ = np.vstack([
                np.eye(self.N) * -line[0] for line in self.polygon_acc_lines
            ])
            M_a_ = np.hstack([
                M_a_, np.vstack([np.eye(self.N)] * len(self.polygon_acc_lines))
            ])
            sgn_acc = np.ones(len(self.polygon_acc_lines))
            sgn_acc[len(self.polygon_acc_lines) // 2:] = -1
            sgn_acc = np.repeat(sgn_acc, self.N)
            b_a_ = np.repeat(self.polygon_acc_lines[:, 1], self.N)

            self.acc_mat_constraint = (M_a_.T * sgn_acc).T
            self.acc_vec_constraint = sgn_acc * b_a_
        else:
            MV_v_ = np.vstack([
                np.eye(self.N - 1) * -line[0] for line in self.polygon_vel_lines
            ])
            MV_v_ = np.hstack([
                MV_v_, np.vstack([np.eye(self.N - 1)] * len(self.polygon_vel_lines))
            ])
            sgn_vel = np.ones(len(self.polygon_vel_lines))
            sgn_vel[len(self.polygon_vel_lines) // 2:] = -1
            sgn_vel = np.repeat(sgn_vel, self.N - 1)
            b_a_ = np.repeat(self.polygon_vel_lines[:, 1], self.N - 1)

            self.vel_mat_constraint = (MV_v_.T * sgn_vel).T
            self.vel_vec_constraint = sgn_vel * b_a_


            MV_a_ = np.vstack([
                np.eye(self.N) * -line[0] for line in self.polygon_acc_lines])
            MV_a_ = np.hstack([
                MV_a_, np.vstack([np.eye(self.N)] * len(self.polygon_acc_lines))
            ])
            sgn_acc = np.ones(len(self.polygon_acc_lines))
            sgn_acc[len(self.polygon_acc_lines) // 2:] = -1
            sgn_acc = np.repeat(sgn_acc, self.N)
            bv_a_ = np.repeat(self.polygon_acc_lines[:, 1], self.N)

            self.acc_mat_constraint = ((MV_a_ @ self.mat_vc_acc_vel).T * sgn_acc).T
            self.acc_vec_constraint = lambda agent_vel: sgn_acc *\
                (bv_a_ + MV_a_ @ agent_vel / self.dt)

        self.last_braking_traj = np.zeros((self.N, 2))


    def flush(self):
        """
        Flush state which consists only of the lastr braking trajectory.
        """
        self.last_braking_traj *= 0


    def const_acc_vel(self, const_M, const_b, agent_vel):
        if not self.velocity_control:
            const_M.append(self.vel_mat_constraint)
            const_b.append(self.vel_vec_constraint(agent_vel))
            const_M.append(self.acc_mat_constraint)
            const_b.append(self.acc_vec_constraint)
        else:
            const_M.append(self.vel_mat_constraint)
            const_b.append(self.vel_vec_constraint)
            const_M.append(self.acc_mat_constraint)
            agent_vel_ = np.zeros(2 * (self.N))
            agent_vel_[0], agent_vel_[self.N] = agent_vel
            const_b.append(self.acc_vec_constraint(agent_vel_))


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
            if np.all(np.linalg.norm(poss, axis=-1) > self.MAX_STOPPING_DIST):
                continue
            vec = -poss / np.stack([np.linalg.norm(poss, axis=-1)] * 2, axis=-1)
            M_ca = np.hstack([np.eye(self.N) * vec[:, 0], np.eye(self.N) * vec[:, 1]])
            v_cb = M_ca @ (
                -poss.flatten("F") + self.vec_pos_vel * np.repeat(agent_vel, self.N)
            ) - np.array([self.min_dist_crowd] * self.N)
            M_cac = -M_ca @ self.mat_pos_control
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

        if self.velocity_control:
            opt_M = self.mat_vc_pos_vel.T @ self.mat_vc_pos_vel +\
                0.25 * np.eye(2 * (self.N - 1))
            reference_vel = np.append(
                reference_vel[:self.N - 1], reference_vel[self.N:2 * self.N - 1]
            )
            opt_V = (reference_pos + 0.5 * self.dt * np.repeat(curr_vel, self.N)).T @\
                self.mat_vc_pos_vel - 0.25 * reference_vel.T
        else:
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

        term_const_M = None
        term_const_b = None
        if not self.velocity_control:
            # constrain safety by ensuring a braking trajectory through a terminal const
            # self.N - 1 and 2 * self.N - 1 repreent the last velocity of the horizon
            term_const_M = self.mat_vel_acc[[self.N - 1, 2 * self.N - 1], :]
            term_const_b = -curr_vel

        control = solve_qp(
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

        if control is None:
            control = np.zeros(2 * self.N)
            control[0:self.N - 1] = self.last_braking_traj[1:, 0]
            control[self.N:2 * self.N - 1] = self.last_braking_traj[1:, 1]
        elif self.velocity_control:
            actions = np.array([
                np.append(control[:self.N - 1], 0),
                np.append(control[self.N - 1:], 0)]
            ).T
        if not self.velocity_control:
            actions = np.array([control[:self.N], control[self.N:]]).T
        self.last_braking_traj = actions  # save last trajecotry in case next step fails
        return actions
