import numpy as np


REPLAN_MOVING = 2
REPLAN_STATIC = 10
REPLAN_NO_CROWD = 10


def replan_close(pos, vel, obs, action, t):
    return t % 10 == 0 or t % max(int(np.linalg.norm(obs[:2]) ** 2 * 10 / 4), 1) == 0
