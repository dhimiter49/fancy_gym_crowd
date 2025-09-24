from typing import Union
from fancy_gym.black_box.controller.base_controller import BaseController
import numpy as np
import fancy_gym.black_box.controller.mpc.factory as mpc_factory


class MPCController(BaseController):
    """
    """

    def __init__(
        self,
        mpc_type: str,
        horizon: int,
        dt: float,
        physical_space: float,
        const_dist_crowd: float,
        agent_max_vel: float,
        agent_max_acc: float,
        crowd_max_vel: float,
        crowd_max_acc: float,
        radius_crowd: Union[list[float], None] = None,
        horizon_tries: int = 0,
        replan_steps: Union[int, None] = None,
        uncertainty: str = '',
        stability_coeff: float = 1.0,
    ):
        self.replan = replan_steps
        self.N = horizon
        self.velocity_control = "velocity" in mpc_type
        self.breaking_steps = 0
        self.mpc = mpc_factory.get_mpc(
            mpc_type,
            horizon=horizon,
            dt=dt,
            physical_space=physical_space,
            const_dist_crowd=const_dist_crowd,
            radius_crowd=radius_crowd,
            agent_max_vel=agent_max_vel,
            agent_max_acc=agent_max_acc,
            crowd_max_vel=crowd_max_vel,
            crowd_max_acc=crowd_max_acc,
            horizon_tries=horizon_tries,
            stability_coeff=stability_coeff,
            uncertainty=uncertainty,
        )


    def get_action(
            self, des_pos, des_vel, curr_pos, curr_vel, wall_dist, crowd, goal
    ):
        des_pos = des_pos[:self.N]
        des_vel = des_vel[:self.N]
        crowd_poss = np.array(crowd[0]) - curr_pos
        crowd_vels = np.array(crowd[1])
        reference_pos = np.repeat(curr_pos, self.N) -\
            np.hstack([des_pos[:self.N, 0], des_pos[:self.N, 1]])
        reference_vel = np.repeat(curr_vel, self.N) -\
            np.hstack([des_vel[:self.N, 0], des_vel[:self.N, 1]])
        control_plan, breaking_flag = self.mpc.get_action(
            (-reference_pos, reference_vel),
            (goal, crowd_poss, curr_vel, crowd_vels, wall_dist, None)
        )
        self.breaking_steps += 1 if breaking_flag and not self.old_breaking_flag else 0
        self.old_breaking_flag = breaking_flag
        return control_plan


    def flush(self):
        self.mpc.reset()

