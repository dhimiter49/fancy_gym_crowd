from typing import Tuple, Optional, Any, Dict
import numpy as np
from gymnasium.core import ObsType
import rvo2

from fancy_gym.envs.classic_control.crowd_navigation.crowd_navigation\
    import CrowdNavigationEnv


class CrowdNavigationORCAEnv(CrowdNavigationEnv):
    """
    Crowd with ORCA policy.

    Args:
        lidar_rays: number of lidar rays, if 0 no lidar is used
        const_vel: sets the dynamics to using constant velocity
        polar: polar observation and action space
        time_frame: time from which to sample and stack the last frames of obs
        lidar_vel: use a velocity representation for each direction of the lidar
        n_frames: number of frames to stack for lidar, irrelevant if lidar_vel
        avoid_agent_parameter: 1 to avoid completely, lower to increase safety distance
            higher values to avoid the agent less
        intersect_crowd: crowd goal position will try to intersect with the agents traj
            to the its goal
    """
    def __init__(
        self,
        n_crowd: int,
        dt: float = 0.1,
        width: int = 20,
        height: int = 20,
        interceptor_percentage: float = 0.5,
        discrete_action: bool = False,
        velocity_control: bool = False,
        lidar_rays: int = 0,
        sequence_obs: bool = False,
        const_vel: bool = False,
        one_way: bool = False,
        polar: bool = False,
        time_frame: int = 0,
        lidar_vel: bool = False,
        n_frames: int = 4,
        lidar_max: float = 0.0,
        intrinsic_rew: bool = False,
        avoid_agent_parameter: float = 4.,
        one_goal: bool = True,
        intersect_crowd: bool = False,
        obs_noise: bool = False
    ):
        self.intersect_crowd = intersect_crowd
        super().__init__(
            n_crowd,
            dt,
            width,
            height,
            interceptor_percentage,
            discrete_action=discrete_action,
            velocity_control=velocity_control,
            lidar_rays=lidar_rays,
            sequence_obs=sequence_obs,
            const_vel=const_vel,
            one_way=one_way,
            polar=polar,
            time_frame=time_frame,
            lidar_vel=lidar_vel,
            n_frames=n_frames,
            lidar_max=lidar_max,
            intrinsic_rew=intrinsic_rew,
            one_goal=one_goal,
            obs_noise=obs_noise,
        )
        assert avoid_agent_parameter > 0
        self.Ci = -1.
        self.neighbor_dist = np.inf
        self.safety_space = np.max(self.PHYSICAL_SPACE[1:]) / 2
        self.time_horizon = 5.
        self.time_horizon_obst = 5.
        self.avoid_agent_parameter = avoid_agent_parameter
        self._start_sim()


    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[ObsType, Dict[str, Any]]:
        if self.lidar:
            self._last_frames *= 0
        obs, info = super().reset(seed=seed, options=options)
        self._start_sim()
        return obs, info


    def _start_sim(self):
        max_neighbors = self.n_crowd
        params = (
            self.neighbor_dist, max_neighbors, self.time_horizon, self.time_horizon_obst
        )
        self.sim = rvo2.PyRVOSimulator(
            self._dt, *params, self.PHYSICAL_SPACE[0], self.CROWD_MAX_VEL
        )
        self.sim.addAgent(
            tuple(self._agent_pos),
            *params,
            self.PHYSICAL_SPACE[0] / self.avoid_agent_parameter,
            self.AGENT_MAX_VEL,
            tuple(self._agent_vel)
        )
        for i, (pos, vel) in enumerate(zip(self._crowd_poss, self._crowd_vels)):
            self.sim.addAgent(
                tuple(pos),
                *params,
                self.PHYSICAL_SPACE[i] + self.safety_space,
                self.CROWD_MAX_VEL,
                tuple(vel)
            )


    def find_dist_between_segs(self, x1, x2, y1, y2):
        if_one_pt = False
        if x2.shape == (2,):
            x2 = x2.reshape((1, 2))
            y2 = y2.reshape((1, 2))
            if_one_pt = True

        end_dist = np.linalg.norm(x2 - y2, axis=1)
        critical_dist = end_dist.copy()
        z_bar = (x2 - x1) - (y2 - y1)
        inds = np.where((np.linalg.norm(z_bar, axis=1) > 0))[0]
        t_bar = - np.sum((x1 - y1) * z_bar[inds, :], axis=1) /\
            np.sum(z_bar[inds, :] * z_bar[inds, :], axis=1)
        t_bar_rep = np.tile(t_bar, (2, 1)).transpose()
        dist_bar = np.linalg.norm(
            x1 + (x2[inds, :] - x1) * t_bar_rep - y1 - (y2[inds, :] - y1) * t_bar_rep,
            axis=1
        )
        inds_2 = np.where((t_bar > 0) & (t_bar < 1.0))
        critical_dist[inds[inds_2]] = dist_bar[inds_2]

        min_dist = np.amin(np.vstack((end_dist, critical_dist)), axis=0)
        # print 'min_dist', min_dist

        if if_one_pt:
            return min_dist[0]
        else:
            return min_dist


    def distPointToSegment(self, p1, p2, p3):
        d = p2 - p1
        if np.linalg.norm(d) < 1e-5:
            u = 0.0
        else:
            u = np.dot(d, (p3 - p1)) / (np.linalg.norm(d) ** 2.0)
        u = max(0.0, min(u, 1.0))

        inter = p1 + u * d
        dist = np.linalg.norm(p3 - inter)
        return dist


    def if_permitStraightLineSoln(self, x1, x2, s1, y1, y2, s2, radius):
        t1 = np.linalg.norm(x2 - x1) / s1
        t2 = np.linalg.norm(y2 - y1) / s2
        if t1 < t2:
            x_crit = x2
            y_crit = y1 + t1 * (y2 - y1) / t2
            if self.distPointToSegment(y_crit, y2, x_crit) < radius:
                return False
        else:
            x_crit = x1 + t2 * (x2 - x1) / t1
            y_crit = y2
            if self.distPointToSegment(x_crit, x2, y_crit) < radius:
                return False
        start_dist = np.linalg.norm(x1 - y1)
        end_dist = np.linalg.norm(x_crit - y_crit)
        mid_dist = self.find_dist_between_segs(x1, x_crit, y1, y_crit)
        dist = min(start_dist, end_dist, mid_dist)
        if dist < radius:
            return False
        return True


    def _start_env_vars(self):
        self.current_seed += 1
        test_case = np.zeros((self.n_crowd + 1, 4))
        assert self.WIDTH == self.HEIGHT
        length = self.WIDTH // 2.8
        for i in range(self.n_crowd + 1):
            counter = 0
            while True:
                # generate random starting/ending points
                counter += 1
                length = length * 1.01
                start = length * 2 * np.random.rand(2,) - length
                end = length * 2 * np.random.rand(2,) - length

                # if colliding with previous test cases
                if_collide = False
                for j in range(i):
                    radius_start = self.PHYSICAL_SPACE[j] + 1.5 * self.PHYSICAL_SPACE[i]
                    radius_end = self.PHYSICAL_SPACE[j] + 1.5 * self.PHYSICAL_SPACE[i]
                    # start
                    if np.linalg.norm(start - test_case[j, 0:2]) < radius_start:
                        if_collide = True
                        break
                    # end
                    if np.linalg.norm(end - test_case[j, 2:4]) < radius_end:
                        if_collide = True
                        break
                if if_collide:
                    continue

                # if straight line is permited
                if i >= 1:
                    if_straightLineSoln = True
                    for j in range(0, i):
                        x1, x2 = test_case[j, 0:2], test_case[j, 2:4]
                        y1, y2 = start, end
                        s1, s2 = self.AGENT_MAX_VEL, self.AGENT_MAX_VEL
                        radius = self.PHYSICAL_SPACE[j] + 1.5 * self.PHYSICAL_SPACE[i]
                        if not self.if_permitStraightLineSoln(
                            x1, x2, s1, y1, y2, s2, radius
                        ):
                            # print 'num_agents %d; i %d; j %d'%  (num_agents, i, j)
                            if_straightLineSoln = False
                            break
                    if if_straightLineSoln:
                        continue


                if np.linalg.norm(start - end) > length * 0.5:
                    break

            # record test case
            test_case[i, 0:2] = start
            test_case[i, 2:4] = end
        agent_pos = test_case[0, :2]
        agent_vel = np.zeros(2)
        goal_pos = test_case[0, 2:4]
        crowd_poss = test_case[1:, :2]
        self._crowd_goal_poss = test_case[1:, 2:4]

        return agent_pos, agent_vel, goal_pos, crowd_poss, crowd_poss * 0


    def _gen_crowd_goal(self, crowd_poss, agent_pos, goal_pos):
        """
        Generated random goals for each member of the crowd.

        Args:
            crowd_poss (numpy.ndarray): list of crowd members

        Returns:
            (numpy.ndarray, numpy.ndarray, numpy.ndarray): the goal positions
        """
        if len(crowd_poss.shape) == 1:
            crowd_poss = np.array([crowd_poss])
        if self.intersect_crowd:
            agent_traj = goal_pos - agent_pos
            intersect_coeff = np.random.uniform(size=len(crowd_poss))
            intersect_point_traj = agent_pos + np.einsum(
                "i,j->ij", intersect_coeff, agent_traj
            )
            crowd_goal_traj = intersect_point_traj - crowd_poss
            crowd_goal_poss = crowd_poss + np.einsum(
                "i,ij->ij", np.random.uniform(1.1, 2., size=len(crowd_poss)), crowd_goal_traj
            )
            crowd_goal_poss = np.clip(
                crowd_goal_poss,
                [-self.W_BORDER, -self.H_BORDER],
                [self.W_BORDER, self.H_BORDER],
            )
        else:
            crowd_goal_poss = np.random.uniform(
                [-self.W_BORDER, -self.H_BORDER],
                [self.W_BORDER, self.H_BORDER],
                (len(crowd_poss), 2)
            )

        return crowd_goal_poss


    def update_crowd(self):
        """
        Create a rvo2 simulation at each time step and run one step

        Agent doesn't stop moving after it reaches the goal,
        because once it stops moving, the reciprocal rule is broken
        """
        self.sim.setAgentPosition(0, tuple(self._agent_pos))
        self.sim.setAgentVelocity(0, tuple(self._agent_vel))
        for i, (pos, vel) in enumerate(zip(self._crowd_poss, self._crowd_vels)):
            self.sim.setAgentPosition(i + 1, tuple(pos))
            self.sim.setAgentVelocity(i + 1, tuple(vel))

        # Set the preferred velocity to be a vector of unit magnitude (speed) in the
        # direction of the goal.
        velocity = np.array(self._goal_pos - self._agent_pos)
        speed = np.linalg.norm(velocity)
        pref_vel = velocity / speed if speed > 1 else velocity
        self.sim.setAgentPrefVelocity(0, tuple(pref_vel))

        crowd_pref_vels = self._crowd_goal_poss - self._crowd_poss
        crowd_pref_vels[
            np.linalg.norm(crowd_pref_vels, axis=-1) <
            self.PHYSICAL_SPACE[1:1 + self.n_crowd]
        ] = 0
        crowd_pref_vels_speed = np.linalg.norm(crowd_pref_vels, axis=-1)

        # update crowd goals
        if not self.one_goal and not self.run_test_case:
            crowd_goal_complete = np.logical_and(
                crowd_pref_vels_speed < self.PHYSICAL_SPACE[1:1 + self.n_crowd],
                np.linalg.norm(self._crowd_vels, axis=-1) < self.MAX_ACC * self._dt
            )
            if len(crowd_goal_complete) > 0:
                self._crowd_goal_poss[crowd_goal_complete] = self._gen_crowd_goal(
                    self._crowd_poss[crowd_goal_complete]
                )
                crowd_pref_vels = self._crowd_goal_poss - self._crowd_poss
                crowd_pref_vels_speed = np.linalg.norm(crowd_pref_vels, axis=-1)

        diff_vel = crowd_pref_vels - self._crowd_vels
        diff_speed = np.linalg.norm(diff_vel, axis=-1)

        over = diff_speed > self.MAX_ACC * self._dt
        under = diff_speed < -self.MAX_ACC * self._dt
        if np.any(over):
            crowd_pref_vels[over] = self._crowd_vels[over] + np.einsum(
                "ij,i->ij", diff_vel[over], 1 / diff_speed[over]
            ) * self.MAX_ACC * self._dt
        if np.any(under):
            crowd_pref_vels[under] = self._crowd_vels[under] - np.einsum(
                "ij,i->ij", diff_vel[under], 1 / diff_speed[under]
            ) * self.MAX_ACC * self._dt
        crowd_pref_vels_speed = np.linalg.norm(crowd_pref_vels, axis=-1)

        over_vel = crowd_pref_vels_speed > self.CROWD_MAX_VEL
        crowd_pref_vels[over_vel] = np.einsum(
            "ij,i->ij", crowd_pref_vels[over_vel], 1 / crowd_pref_vels_speed[over_vel]
        ) * self.CROWD_MAX_VEL
        for i in range(self.n_crowd):
            self.sim.setAgentPrefVelocity(i + 1, tuple(crowd_pref_vels[i]))


        self.sim.doStep()

        actions = np.empty((self.n_crowd, 2))
        for i in range(self.n_crowd):
            actions[i] = np.array(self.sim.getAgentVelocity(i + 1))

        self._crowd_vels = actions
        self._crowd_poss += self._crowd_vels * self._dt
