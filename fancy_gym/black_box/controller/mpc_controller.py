from fancy_gym.black_box.controller.base_controller import BaseController
from qpsolvers import solve_qp
from scipy import sparse
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
    :param horizon_crowd_pred: define explicitly a shorter horizon for the crowd, future
        crowd positions are complitly neglected as the memebr of the crowd would dissapear
    :param horizon_tries: if the qp doesnt hind a solution theme many times again with a
        shorter horizon (the horizon is shortned arbitrarily by half of the current
        horizon e.g. for horizon_tries=3 and horizon 21 -> 21, 11, 6, 3)
    :param replan_steps : how often to replan
    :param dt : time step
    :param min_dist_crowd : if zero ignore crowd, if given constrain distance to crowd
    :param min_dist_wall: minimum distance to the wall
    :param velocity_control: use velocity to control agent or not
    :param uncertainty: uncertainty type, '' no uncertainty, 'dir' velocity direction,
        'vel' for speed and direction
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
        horizon_crowd_pred: int = None,
        horizon_tries: int = 0,
        replan_steps: int = None,
        dt: float = 0.1,
        min_dist_crowd: float = 0.0,
        min_dist_wall: float = 0.4,
        velocity_control: float = False,
        uncertainty: str = '',
    ):
        self.N = horizon
        self.horizon_tries = horizon_tries
        self.short_hor_only_crowd = True
        self.N_crowd = self.N if horizon_crowd_pred is None else horizon_crowd_pred
        self.replan = replan_steps if replan_steps is not None else self.N
        self.MAX_STOPPING_TIME = max_vel / max_acc
        self.MAX_STOPPING_DIST = 2 * (
            max_vel * self.MAX_STOPPING_TIME - 0.5 * max_acc * self.MAX_STOPPING_TIME ** 2
        )
        self.MAX_VEL = max_vel
        self.MAX_ACC = max_acc
        self.dt = dt
        self.velocity_control = velocity_control
        self.mat_pos_acc = mat_pos_acc
        self.vec_pos_vel = mat_pos_vel
        self.mat_vel_acc = mat_vel_acc
        self.mat_vc_pos_vel = mat_vc_pos_vel
        self.mat_vc_acc_vel = mat_vc_acc_vel
        if self.velocity_control:
            self.mat_pos_control = self.mat_vc_pos_vel
            self.vec_pos_vel = self.vec_pos_vel_crowd = 0.5 * self.dt
        else:
            self.mat_pos_control = self.mat_pos_acc
            self.vec_pos_vel_crowd = np.concatenate([
                self.vec_pos_vel[:self.N_crowd],
                self.vec_pos_vel[self.N: self.N + self.N_crowd]
            ])

        self.mat_pos_control_crowd = np.concatenate([
            self.mat_pos_control[:self.N_crowd],
            self.mat_pos_control[self.N: self.N + self.N_crowd]
        ])
        self.lin_sides = 8
        self.polygon_acc_lines = gen_polygon(max_acc, self.lin_sides)
        self.polygon_vel_lines = gen_polygon(max_vel, self.lin_sides)
        self.min_dist_crowd = min_dist_crowd
        self.min_dist_wall = min_dist_wall

        if self.velocity_control:
            self.opt_M = self.mat_vc_pos_vel.T @ self.mat_vc_pos_vel +\
                1.0 * np.eye(2 * (self.N - 1))
        else:
            self.opt_M = self.mat_pos_acc.T @ self.mat_pos_acc +\
                2.0 * self.mat_vel_acc.T @ self.mat_vel_acc +\
                0.2 * np.eye(2 * self.N)
        self.opt_M = sparse.csr_matrix(self.opt_M)
        self.uncertainty = uncertainty

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
            self.vel_vec_constraint = lambda agent_vel, idxs: sgn_vel[idxs] *\
                (b_v_[idxs] - M_v_[idxs] @ np.repeat(agent_vel, self.N))

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


    def relevant_vel_idxs(self, agent_vel):
        horizon = self.N - 1 if self.velocity_control else self.N
        angle = np.arctan2(agent_vel[1], agent_vel[0])
        angle = 2 * np.pi + angle if angle < 0 else angle
        angle_idx = angle // (2 * np.pi / self.lin_sides)
        idxs = [
            angle_idx, (angle_idx + 1) % self.lin_sides, (angle_idx - 1) % self.lin_sides
        ]
        idxs = np.hstack(list(idxs) * horizon) +\
            np.repeat(np.arange(0, horizon * self.lin_sides, self.lin_sides), 3)
        return np.array(idxs, dtype=int)


    def const_acc_vel(self, const_M, const_b, agent_vel):
        idxs = self.relevant_vel_idxs(agent_vel)
        if not self.velocity_control:
            const_M.append(self.vel_mat_constraint[idxs])
            const_b.append(self.vel_vec_constraint(agent_vel, idxs))
            const_M.append(self.acc_mat_constraint)
            const_b.append(self.acc_vec_constraint)
        else:
            const_M.append(self.vel_mat_constraint[idxs])
            const_b.append(self.vel_vec_constraint[idxs])
            const_M.append(self.acc_mat_constraint)
            agent_vel_ = np.zeros(2 * (self.N))
            agent_vel_[0], agent_vel_[self.N] = agent_vel
            const_b.append(self.acc_vec_constraint(agent_vel_))


    def calculate_crowd_positions(self, crowd_poss, crowd_vels):
        """
        Calculate the crowd positions for the next horizon given the constant velocity for
        each member. The formula P_i = p_0 + i * v * dt, where for point i in horizon the
        position will be p_0 + i * v * dt.

        Uncertainty handling by copying positions and generating new plausible velocities
        depending on the direction and speed uncertainty.

        Args:
            crowd_poss (numpy.ndarray): an array of size (#crowd, 2) with the current
                positions of each member
            crowd_vels (numpy.ndarray): an array of size (#crowd, 2) with the current
                velocities of each member
        Return:
            (numpy.ndarray): predicted positions of the crowd throughout the horizon
        """
        num_crowd = crowd_poss.shape[0]
        new_crowd_vels = []
        if self.uncertainty in ["dir", "vel"]:
            alphas = np.pi - 5 * np.pi / 6 * (
                np.linalg.norm(crowd_vels, axis=-1) / self.MAX_VEL
            )
            n_trajs = np.where(alphas > np.pi / 2, 5, 3)  # 3 traj if less then 90, else 5
            n_trajs = n_trajs.reshape(num_crowd)
            angles = alphas * (1 / (n_trajs - 1))
            all_dir_crowd_vels = np.repeat(crowd_vels, n_trajs, axis=0)
            all_dir_angles = np.repeat(angles, n_trajs, axis=0)

            # start from current angle (0) then remove alpha (-1) then add alpha (1)
            # then for five remove twice alpha (-2) and add twice alpha (2)
            mult_angles = np.array([0, -1, 1, -2, 2])
            all_dir_angles *= np.concatenate([mult_angles[:i] for i in n_trajs])
            dir_matrix = np.stack([
                np.cos(all_dir_angles), -np.sin(all_dir_angles),
                np.sin(all_dir_angles), np.cos(all_dir_angles)
            ], axis=-1).reshape(len(all_dir_crowd_vels), 2, 2)

            new_crowd_vels = np.einsum('ijk,ij->ij', dir_matrix, all_dir_crowd_vels)
            crowd_poss = np.repeat(crowd_poss, n_trajs, axis=0)
            crowd_vels = new_crowd_vels

        if self.uncertainty == "vel":
            crowd_poss = np.repeat(crowd_poss, 3, axis=0)
            new_crowd_vels = np.repeat(crowd_vels, 3, axis=0)
            uncertainty = np.stack([
                np.array([0, self.MAX_ACC, self.MAX_ACC]) / np.sqrt(2) *
                self.dt * np.array([1, 1, -1]),
            ] * len(crowd_vels)).reshape(-1, 1)
            new_crowd_vels += uncertainty
            crowd_vels = new_crowd_vels

        return np.stack([crowd_poss] * self.N_crowd) + np.einsum(
            'ijk,i->ijk',
            np.stack([crowd_vels] * self.N_crowd, 0) * self.dt,
            np.arange(1, self.N_crowd + 1)
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
            dist = np.linalg.norm(poss, axis=-1)
            zero_idx = np.where(np.linalg.norm(poss, axis=-1) == 0)[0]
            poss[zero_idx] += 1e-8
            vec = -(poss.T / np.linalg.norm(poss, axis=-1)).T
            angle = np.arccos(np.clip(np.dot(-vec, agent_vel), -1, 1)) > np.pi / 4
            if np.all(dist > self.MAX_STOPPING_DIST) or\
               (np.all(dist > self.MAX_STOPPING_DIST / 2) and np.all(angle)):
                continue
            M_ca = np.hstack([
                np.eye(self.N_crowd) * vec[:, 0], np.eye(self.N_crowd) * vec[:, 1]
            ])
            v_cb = M_ca @ (
                -poss.flatten("F") + self.vec_pos_vel_crowd *
                np.repeat(agent_vel, self.N_crowd)
            ) - np.array([self.min_dist_crowd] * self.N_crowd)
            M_cac = -M_ca @ self.mat_pos_control_crowd
            const_M.append(M_cac)
            const_b.append(v_cb)


    def wall_eq(self, wall_dist):
        """
        Reutrns the equation for all four walls knowing that the index are:
             2    To represent this in the format ax+by+c, a nd b are one of [0, 1, -1],
           1   0  e.g. for index 0 in the graph to the left b=0 and a=-1 while for index
             3    1 a=1, while c is the distance to the wall.
        """
        eqs = np.stack(
            [
                np.array([-1, 1, 0, 0]),
                np.array([0, 0, -1, 1]),
                wall_dist - self.min_dist_wall
            ],
            axis=1
        )
        return eqs[wall_dist < self.MAX_STOPPING_DIST * 0.8]


    def const_lin_pos(self, const_M, const_b, line_eq, agent_vel):
        """The linear position constraint is given by the equation ax+yb+c"""
        for line in line_eq:
            M_ca = np.hstack([np.eye(self.N) * line[0], np.eye(self.N) * line[1]])
            if not self.velocity_control:
                v_c = -M_ca @ (self.vec_pos_vel * np.repeat(agent_vel, self.N)) - line[2]
                const_M.append(-M_ca @ self.mat_pos_acc)
            else:
                v_c = -M_ca @ (0.5 * self.dt * np.repeat(agent_vel, self.N)) - line[2]
                const_M.append(-M_ca @ self.mat_vc_pos_vel)
            const_b.append(-v_c)


    def get_action(self, des_pos, des_vel, curr_pos, curr_vel, wall_dist, crowd=None):
        des_pos = des_pos[:self.N]
        des_vel = des_vel[:self.N]
        reference_pos = np.repeat(curr_pos, self.N) -\
            np.hstack([des_pos[:self.N, 0], des_pos[:self.N, 1]])
        reference_vel = np.repeat(curr_vel, self.N) -\
            np.hstack([des_vel[:self.N, 0], des_vel[:self.N, 1]])
        if self.velocity_control:
            reference_vel = -np.hstack([des_vel[:self.N, 0], des_vel[:self.N, 1]])

        if self.velocity_control:
            reference_vel = np.append(
                reference_vel[:self.N - 1], reference_vel[self.N:2 * self.N - 1]
            )
            vec = reference_pos + 0.5 * self.dt * np.repeat(curr_vel, self.N)
            vec[self.replan:self.N] *= 0
            vec[self.N + self.replan:] *= 0
            reference_vel[self.replan:self.N] *= 0
            reference_vel[self.N + self.replan:] *= 0
            opt_V = vec.T @ self.mat_vc_pos_vel + 1.0 * reference_vel.T
        else:
            vec = reference_pos + self.vec_pos_vel * np.repeat(curr_vel, self.N)
            vec[self.replan:self.N] *= 0
            vec[self.N + self.replan:] *= 0
            reference_vel[self.replan:self.N] *= 0
            reference_vel[self.N + self.replan:] *= 0
            opt_V = vec.T @ self.mat_pos_acc + 2.0 * reference_vel.T @ self.mat_vel_acc

        # constraint matrices and bounds
        const_M = []
        const_b = []

        # constrain distance relative to the crowd
        if self.min_dist_crowd > 0:
            self.const_crowd(const_M, const_b, crowd, curr_pos, curr_vel)
        crowd_const_dim = len(const_M)

        # constraint distance to the wall
        wall_eqs = self.wall_eq(wall_dist)
        if len(wall_eqs) != 0:
            self.const_lin_pos(const_M, const_b, wall_eqs, curr_vel)
        wall_const_dim = len(const_M) - crowd_const_dim

        # constrain acceleration and velocity limits by using an inner polygon of a circle
        self.const_acc_vel(const_M, const_b, curr_vel)
        acc_vel_const_dim = len(const_M) - crowd_const_dim - wall_const_dim

        term_const_M = None
        term_const_b = None
        if not self.velocity_control:
            # constrain safety by ensuring a braking trajectory through a terminal const
            # self.N - 1 and 2 * self.N - 1 repreent the last velocity of the horizon
            term_const_M = self.mat_vel_acc[[self.N - 1, 2 * self.N - 1], :]
            term_const_b = -curr_vel

        const_M = sparse.csr_matrix(np.vstack(const_M))
        const_b = np.hstack(const_b)
        control = solve_qp(
            self.opt_M, opt_V,
            G=const_M, h=const_b,
            A=term_const_M, b=term_const_b,
            solver="clarabel",
            tol_gap_abs=5e-5,
            tol_gap_rel=5e-5,
            tol_feas=1e-4,
            tol_infeas_abs=5e-5,
            tol_infeas_rel=5e-5,
            tol_ktratio=1e-4
        )

        if control is None and self.horizon_tries > 0:
            horizon = self.N
            full_dim = crowd_const_dim + wall_const_dim + acc_vel_const_dim
            opt_M = self.opt_M.toarray() if not self.short_hor_only_crowd else self.opt_M
            short_dims = crowd_const_dim if self.short_hor_only_crowd else full_dim
            while self.horizon_tries > 0:
                shorten_by = horizon // 2
                # print("Using a shorter crowd horizon of: ", horizon - shorten_by)
                del_idx = list(np.array([
                    np.arange(horizon - shorten_by, horizon) + horizon * i
                    for i in range(0, short_dims)
                ]).flatten())
                const_M = np.delete(const_M.toarray(), del_idx, axis=0)
                const_b = np.delete(const_b, del_idx, axis=0)

                if not self.short_hor_only_crowd:
                    # remove indeces from the objective
                    obj_horizon = horizon - 1 if self.velocity_control else horizon
                    del_idx = list(np.array([
                        np.arange(obj_horizon - shorten_by, obj_horizon) + obj_horizon * i
                        for i in range(0, 2)  # 2 dims x and y
                    ]).flatten())
                    opt_M = np.delete(opt_M, del_idx, axis=0)
                    opt_M = np.delete(opt_M, del_idx, axis=1)
                    const_M = np.delete(const_M, del_idx, axis=1)
                    opt_V = np.delete(opt_V, del_idx, axis=0)
                    opt_M = sparse.csr_matrix(opt_M)

                const_M = sparse.csr_matrix(const_M)

                # try again with shorter crowd horizon
                control = solve_qp(
                    opt_M, opt_V,
                    G=const_M, h=const_b,
                    A=term_const_M, b=term_const_b,
                    solver="clarabel",
                    tol_gap_abs=5e-5,
                    tol_gap_rel=5e-5,
                    tol_feas=1e-4,
                    tol_infeas_abs=5e-5,
                    tol_infeas_rel=5e-5,
                    tol_ktratio=1e-4
                )
                opt_M = opt_M.toarray() if not self.short_hor_only_crowd else opt_M
                if control is not None:
                    break
                horizon -= shorten_by
                self.horizon_tries -= 1
            self.horizon_tries = 3


        if control is None:
            horizon = len(self.last_braking_traj.flatten()) // 2
            control = np.zeros(2 * horizon)
            control[0:horizon - 1] = self.last_braking_traj[1:, 0]
            control[horizon:2 * horizon - 1] = self.last_braking_traj[1:, 1]
            actions = np.array([control[:horizon], control[horizon:]]).T
        else:
            if not self.velocity_control:
                actions = np.array([control[:self.N], control[self.N:]]).T
            else:
                actions = np.array([
                    np.append(control[:len(control) // 2], 0),
                    np.append(control[len(control) // 2:], 0)]
                ).T
        self.last_braking_traj = actions  # save last trajecotry in case next step fails
        return actions
