from copy import deepcopy

import numpy as np
from gymnasium import register as gym_register
from .registry import register, upgrade

from . import classic_control, mujoco
from .classic_control.simple_reacher.simple_reacher import SimpleReacherEnv
from .classic_control.simple_reacher import MPWrapper as MPWrapper_SimpleReacher
from .classic_control.hole_reacher.hole_reacher import HoleReacherEnv
from .classic_control.hole_reacher import MPWrapper as MPWrapper_HoleReacher
from .classic_control.crowd_navigation.crowd_navigation import CrowdNavigationEnv
from .classic_control.crowd_navigation.crowd_navigation_inter import CrowdNavigationInterEnv
from .classic_control.crowd_navigation.crowd_navigation_static import CrowdNavigationStaticEnv
from .classic_control.crowd_navigation.l_shape_env import LShapeCrowdNavigationEnv
from .classic_control.crowd_navigation.navigation import NavigationEnv
from .classic_control.crowd_navigation.crowd_navigation_orca import CrowdNavigationORCAEnv
from .classic_control.crowd_navigation.crowd_navigation_sfm import CrowdNavigationSFMEnv
from .classic_control.crowd_navigation import (
    MPWrapper_Crowd,
    MPWrapper_Navigation,
    MPWrapper_CrowdStatic,
    MPWrapper_Crowd_Vel,
    MPWrapper_Navigation_Vel,
    MPWrapper_CrowdStatic_Vel
)
from .classic_control.viapoint_reacher.viapoint_reacher import ViaPointReacherEnv
from .classic_control.viapoint_reacher import MPWrapper as MPWrapper_ViaPointReacher
from .mujoco.reacher.reacher import ReacherEnv, MAX_EPISODE_STEPS_REACHER
from .mujoco.reacher.mp_wrapper import MPWrapper as MPWrapper_Reacher
from .mujoco.ant_jump.ant_jump import MAX_EPISODE_STEPS_ANTJUMP
from .mujoco.beerpong.beerpong import MAX_EPISODE_STEPS_BEERPONG, FIXED_RELEASE_STEP
from .mujoco.beerpong.mp_wrapper import MPWrapper as MPWrapper_Beerpong
from .mujoco.beerpong.mp_wrapper import MPWrapper_FixedRelease as MPWrapper_Beerpong_FixedRelease
from .mujoco.half_cheetah_jump.half_cheetah_jump import MAX_EPISODE_STEPS_HALFCHEETAHJUMP
from .mujoco.hopper_jump.hopper_jump import MAX_EPISODE_STEPS_HOPPERJUMP
from .mujoco.hopper_jump.hopper_jump_on_box import MAX_EPISODE_STEPS_HOPPERJUMPONBOX
from .mujoco.hopper_throw.hopper_throw import MAX_EPISODE_STEPS_HOPPERTHROW
from .mujoco.hopper_throw.hopper_throw_in_basket import MAX_EPISODE_STEPS_HOPPERTHROWINBASKET
from .mujoco.walker_2d_jump.walker_2d_jump import MAX_EPISODE_STEPS_WALKERJUMP
from .mujoco.box_pushing.box_pushing_env import BoxPushingDense, BoxPushingTemporalSparse, \
    BoxPushingTemporalSpatialSparse, MAX_EPISODE_STEPS_BOX_PUSHING
from .mujoco.table_tennis.table_tennis_env import TableTennisEnv, TableTennisWind, TableTennisGoalSwitching, TableTennisMarkov, \
    MAX_EPISODE_STEPS_TABLE_TENNIS, MAX_EPISODE_STEPS_TABLE_TENNIS_MARKOV_VER
from .mujoco.table_tennis.mp_wrapper import TT_MPWrapper as MPWrapper_TableTennis
from .mujoco.table_tennis.mp_wrapper import TT_MPWrapper_Replan as MPWrapper_TableTennis_Replan
from .mujoco.table_tennis.mp_wrapper import TTRndRobot_MPWrapper as MPWrapper_TableTennis_Rnd
from .mujoco.table_tennis.mp_wrapper import TTVelObs_MPWrapper as MPWrapper_TableTennis_VelObs
from .mujoco.table_tennis.mp_wrapper import TTVelObs_MPWrapper_Replan as MPWrapper_TableTennis_VelObs_Replan
from fancy_gym.envs.classic_control.crowd_navigation.dynamics import (
    gen_mat_vc_pos_vel,
    gen_mat_vc_acc_vel
)

# Classic Control
# Simple Reacher
register(
    id='fancy/SimpleReacher-v0',
    entry_point=SimpleReacherEnv,
    mp_wrapper=MPWrapper_SimpleReacher,
    max_episode_steps=200,
    kwargs={
        "n_links": 2,
    }
)

register(
    id='fancy/CrowdNavigation-v0',
    entry_point=CrowdNavigationEnv,
    mp_wrapper=MPWrapper_Crowd,
    max_episode_steps=100,
    kwargs={
        "n_crowd": 4,
        "width": 18,
        "height": 18,
        "interceptor_percentage": 2,
    }
)

register(
    id='fancy/CrowdNavigationORCA-v0',
    entry_point=CrowdNavigationORCAEnv,
    mp_wrapper=MPWrapper_Crowd,
    max_episode_steps=100,
    kwargs={
        "n_crowd": 6,
        "width": 18,
        "height": 18,
        "interceptor_percentage": 2,
    }
)

register(
    id='fancy/CrowdNavigationORCAVel-v0',
    entry_point=CrowdNavigationORCAEnv,
    mp_wrapper=MPWrapper_Crowd,
    max_episode_steps=100,
    kwargs={
        "n_crowd": 6,
        "width": 18,
        "height": 18,
        "velocity_control": True,
        "interceptor_percentage": 2,
    }
)

register(
    id='fancy/CrowdNavigationSFM-v0',
    entry_point=CrowdNavigationSFMEnv,
    mp_wrapper=MPWrapper_Crowd,
    max_episode_steps=100,
    kwargs={
        "n_crowd": 6,
        "width": 18,
        "height": 18,
        "interceptor_percentage": 2,
    }
)

register(
    id='fancy/CrowdNavigationSFMVel-v0',
    entry_point=CrowdNavigationSFMEnv,
    mp_wrapper=MPWrapper_Crowd,
    max_episode_steps=100,
    kwargs={
        "n_crowd": 6,
        "width": 18,
        "height": 18,
        "velocity_control": True,
        "interceptor_percentage": 2,
    }
)

register(
    id='fancy/CrowdNavigationConst-v0',
    entry_point=CrowdNavigationEnv,
    mp_wrapper=MPWrapper_Crowd,
    max_episode_steps=100,
    kwargs={
        "n_crowd": 6,
        "width": 20,
        "height": 8,
        "interceptor_percentage": 2,
        "const_vel": True,
    }
)

register(
    id='fancy/CrowdNavigationConstVel-v0',
    entry_point=CrowdNavigationEnv,
    mp_wrapper=MPWrapper_Crowd_Vel,
    max_episode_steps=100,
    kwargs={
        "n_crowd": 6,
        "width": 20,
        "height": 8,
        "interceptor_percentage": 2,
        "const_vel": True,
        "velocity_control": True,
    }
)

for dt in [0.1, 0.2, 0.3, 0.4, 0.5]:
    steps = int(-(-10 // dt))
    traj = int(-(-2.01 // dt))  # assuming you need 2 seconds to stop
    register(
        id=f'fancy/CrowdNavigationConstVel{dt}-v0',
        entry_point=CrowdNavigationEnv,
        mp_wrapper=MPWrapper_Crowd_Vel,
        max_episode_steps=steps,
        mp_config_override={
            'ProDMP': {
                'controller_kwargs': {
                    'controller_type': 'mpc',
                    'mat_vc_pos_vel': gen_mat_vc_pos_vel(traj, dt),
                    'mat_vc_acc_vel': gen_mat_vc_acc_vel(traj, dt),
                    'max_acc': 1.5,
                    'max_vel': 3.0,
                    'horizon': traj,
                    'dt': dt,
                    'velocity_control': True,
                    'min_dist_crowd': 0.8001,  # 2 * personal space
                    'min_dist_wall': 0.41,  # physical space of agent + 0.01
                },
                'black_box_kwargs': {
                    'replanning_schedule':
                        lambda pos, vel, obs, action, t: t % int(1 // dt) == 0,
                }
            }
        },
        kwargs={
            "dt": dt,
            "n_crowd": 6,
            "width": 20,
            "height": 8,
            "interceptor_percentage": 2,
            "const_vel": True,
            "velocity_control": True,
        }
    )

register(
    id='fancy/CrowdNavigationConstSeqVel-v0',
    entry_point=CrowdNavigationEnv,
    mp_wrapper=MPWrapper_Crowd_Vel,
    max_episode_steps=100,
    kwargs={
        "n_crowd": 6,
        "width": 20,
        "height": 8,
        "interceptor_percentage": 2,
        "const_vel": True,
        "velocity_control": True,
        "sequence_obs": True,
    }
)

register(
    id='fancy/CrowdNavigationConstSeqPolarVel-v0',
    entry_point=CrowdNavigationEnv,
    mp_wrapper=MPWrapper_Crowd_Vel,
    max_episode_steps=100,
    kwargs={
        "n_crowd": 6,
        "width": 20,
        "height": 8,
        "interceptor_percentage": 2,
        "const_vel": True,
        "velocity_control": True,
        "sequence_obs": True,
        "polar": True
    }
)

register(
    id='fancy/CrowdNavigationConstLiDAR-v0',
    entry_point=CrowdNavigationEnv,
    mp_wrapper=MPWrapper_Crowd,
    max_episode_steps=100,
    kwargs={
        "n_crowd": 6,
        "width": 20,
        "height": 8,
        "interceptor_percentage": 2,
        "const_vel": True,
        "lidar_rays": 40,
    }
)

register(
    id='fancy/CrowdNavigationConstLiDARSnd-v0',
    entry_point=CrowdNavigationEnv,
    mp_wrapper=MPWrapper_Crowd,
    max_episode_steps=100,
    kwargs={
        "n_crowd": 6,
        "width": 20,
        "height": 8,
        "interceptor_percentage": 2,
        "const_vel": True,
        "lidar_rays": 40,
        "time_frame": 1,
    }
)

register(
    id='fancy/CrowdNavigationConstLiDARVel-v0',
    entry_point=CrowdNavigationEnv,
    mp_wrapper=MPWrapper_Crowd_Vel,
    max_episode_steps=100,
    kwargs={
        "n_crowd": 6,
        "width": 20,
        "height": 8,
        "interceptor_percentage": 2,
        "const_vel": True,
        "lidar_rays": 40,
        "velocity_control": True,
        "lidar_vel": True,
    }
)

register(
    id='fancy/CrowdNavigationConstLiDARPolarVel-v0',
    entry_point=CrowdNavigationEnv,
    max_episode_steps=100,
    kwargs={
        "n_crowd": 6,
        "width": 20,
        "height": 8,
        "interceptor_percentage": 2,
        "const_vel": True,
        "lidar_rays": 40,
        "polar": True,
        "velocity_control": True,
    }
)

register(
    id='fancy/CrowdNavigationConstOneWay-v0',
    entry_point=CrowdNavigationEnv,
    max_episode_steps=100,
    kwargs={
        "n_crowd": 20,
        "width": 30,
        "height": 10,
        "interceptor_percentage": 2,
        "const_vel": True,
        "one_way": True,
    }
)

register(
    id='fancy/CrowdNavigationConstOneWayVel-v0',
    entry_point=CrowdNavigationEnv,
    max_episode_steps=100,
    kwargs={
        "n_crowd": 20,
        "width": 30,
        "height": 10,
        "interceptor_percentage": 2,
        "const_vel": True,
        "one_way": True,
        "velocity_control": True,
    }
)

register(
    id='fancy/CrowdNavigationConstOneWayLiDARVel-v0',
    entry_point=CrowdNavigationEnv,
    max_episode_steps=100,
    kwargs={
        "n_crowd": 20,
        "width": 30,
        "height": 10,
        "interceptor_percentage": 2,
        "const_vel": True,
        "one_way": True,
        "velocity_control": True,
        "lidar_rays": 40,
        "lidar_vel": True
    }
)

register(
    id='fancy/CrowdNavigationLiDAR-v0',
    entry_point=CrowdNavigationEnv,
    mp_wrapper=MPWrapper_Crowd,
    max_episode_steps=100,
    kwargs={
        "n_crowd": 8,
        "width": 18,
        "height": 18,
        "interceptor_percentage": 2,
        "lidar_rays": 40,
    }
)

register(
    id='fancy/CrowdNavigationVel-v0',
    entry_point=CrowdNavigationEnv,
    mp_wrapper=MPWrapper_Crowd_Vel,
    max_episode_steps=100,
    kwargs={
        "n_crowd": 4,
        "width": 18,
        "height": 18,
        "interceptor_percentage": 2,
        "velocity_control": True,
    }
)

register(
    id='fancy/CrowdNavigationStatic-v0',
    entry_point=CrowdNavigationStaticEnv,
    mp_wrapper=MPWrapper_CrowdStatic,
    max_episode_steps=100,
    kwargs={
        "n_crowd": 4,
        "width": 16,
        "height": 16,
        "interceptor_percentage": 2,
    }
)

register(
    id='fancy/CrowdNavigationStaticPolar-v0',
    entry_point=CrowdNavigationStaticEnv,
    mp_wrapper=MPWrapper_CrowdStatic,
    max_episode_steps=100,
    kwargs={
        "n_crowd": 4,
        "width": 16,
        "height": 16,
        "interceptor_percentage": 2,
        "polar": True,
    }
)

register(
    id='fancy/CrowdNavigationStaticPolarVel-v0',
    entry_point=CrowdNavigationStaticEnv,
    max_episode_steps=100,
    kwargs={
        "n_crowd": 4,
        "width": 16,
        "height": 16,
        "interceptor_percentage": 2,
        "velocity_control": True,
        "polar": True,
    }
)

register(
    id='fancy/CrowdNavigationStaticLiDAR-v0',
    entry_point=CrowdNavigationStaticEnv,
    mp_wrapper=MPWrapper_CrowdStatic,
    max_episode_steps=100,
    kwargs={
        "n_crowd": 4,
        "width": 16,
        "height": 16,
        "interceptor_percentage": 2,
        "lidar_rays": 40,  # 0 means no lidar, cartesian obs
    }
)

register(
    id='fancy/CrowdNavigationStaticLiDARVel-v0',
    entry_point=CrowdNavigationStaticEnv,
    mp_wrapper=MPWrapper_CrowdStatic_Vel,
    max_episode_steps=100,
    kwargs={
        "n_crowd": 4,
        "width": 16,
        "height": 16,
        "interceptor_percentage": 2,
        "velocity_control": True,
        "lidar_rays": 40,  # 0 means no lidar, cartesian obs
    }
)

register(
    id='fancy/CrowdNavigationStaticLiDARPolarVel-v0',
    entry_point=CrowdNavigationStaticEnv,
    max_episode_steps=100,
    kwargs={
        "n_crowd": 4,
        "width": 16,
        "height": 16,
        "interceptor_percentage": 2,
        "velocity_control": True,
        "lidar_rays": 40,  # 0 means no lidar, cartesian obs
        "polar": True
    }
)

register(
    id='fancy/CrowdNavigationStaticVel-v0',
    entry_point=CrowdNavigationStaticEnv,
    mp_wrapper=MPWrapper_CrowdStatic_Vel,
    max_episode_steps=100,
    kwargs={
        "n_crowd": 4,
        "width": 16,
        "height": 16,
        "interceptor_percentage": 2,
        "velocity_control": True,
    }
)

register(
    id='fancy/CrowdNavigationStaticSeqVel-v0',
    entry_point=CrowdNavigationStaticEnv,
    mp_wrapper=MPWrapper_CrowdStatic_Vel,
    max_episode_steps=100,
    kwargs={
        "n_crowd": 4,
        "width": 16,
        "height": 16,
        "interceptor_percentage": 2,
        "velocity_control": True,
        "sequence_obs": True,
    }
)

register(
    id='fancy/Navigation-v0',
    entry_point=NavigationEnv,
    mp_wrapper=MPWrapper_Navigation,
    max_episode_steps=60,
    kwargs={
        "width": 10,
        "height": 10,
    }
)

register(
    id='fancy/NavigationPolar-v0',
    entry_point=NavigationEnv,
    mp_wrapper=MPWrapper_Navigation,
    max_episode_steps=60,
    kwargs={
        "width": 10,
        "height": 10,
        "polar": True,
    }
)

register(
    id='fancy/NavigationPolarVel-v0',
    entry_point=NavigationEnv,
    max_episode_steps=60,
    kwargs={
        "width": 10,
        "height": 10,
        "velocity_control": True,
        "polar": True,
    }
)

register(
    id='fancy/NavigationVel-v0',
    entry_point=NavigationEnv,
    mp_wrapper=MPWrapper_Navigation_Vel,
    max_episode_steps=60,
    kwargs={
        "width": 10,
        "height": 10,
        "velocity_control": True,
    }
)

register(
    id='fancy/NavigationSeqVel-v0',
    entry_point=NavigationEnv,
    mp_wrapper=MPWrapper_Crowd_Vel,
    max_episode_steps=100,
    kwargs={
        "width": 10,
        "height": 10,
        "velocity_control": True,
        "sequence_obs": True,
    }
)

register(
    id='fancy/LShapeNavigation-v0',
    entry_point=LShapeCrowdNavigationEnv,
    mp_wrapper=MPWrapper_Navigation_Vel,
    max_episode_steps=60,
    kwargs={
        "width": 10,
        "height": 10,
    }
)

register(
    id='fancy/LShapeCrowdNavigation-v0',
    entry_point=LShapeCrowdNavigationEnv,
    mp_wrapper=MPWrapper_Navigation_Vel,
    max_episode_steps=80,
    kwargs={
        "n_crowd": 4,
        "width": 16,
        "height": 16,
        "interceptor_percentage": 2,
    }
)

register(
    id='fancy/CrowdNavigationInter-v0',
    entry_point=CrowdNavigationInterEnv,
    mp_wrapper=MPWrapper_Crowd,
    max_episode_steps=100,
    kwargs={
        "n_crowd": 8,
        "width": 16,
        "height": 16,
        "interceptor_percentage": 2,
    }
)

register(
    id='fancy/CrowdNavigationInterVel-v0',
    entry_point=CrowdNavigationInterEnv,
    mp_wrapper=MPWrapper_Crowd,
    max_episode_steps=100,
    kwargs={
        "n_crowd": 8,
        "width": 16,
        "height": 16,
        "interceptor_percentage": 2,
        "velocity_control": True,
    }
)

register(
    id='fancy/CrowdNavigationInterLiDAR-v0',
    entry_point=CrowdNavigationInterEnv,
    mp_wrapper=MPWrapper_Crowd,
    max_episode_steps=100,
    kwargs={
        "n_crowd": 8,
        "width": 16,
        "height": 16,
        "interceptor_percentage": 2,
        "lidar_rays": 40,
        "lidar_vel": True,
    }
)

register(
    id='fancy/CrowdNavigationInterLiDARVel-v0',
    entry_point=CrowdNavigationInterEnv,
    mp_wrapper=MPWrapper_Crowd_Vel,
    max_episode_steps=100,
    kwargs={
        "n_crowd": 8,
        "width": 16,
        "height": 16,
        "interceptor_percentage": 2,
        "lidar_rays": 40,
        "lidar_vel": True,
        "velocity_control": True,
    }
)

register(
    id='fancy/CrowdNavigationInterSeq-v0',
    entry_point=CrowdNavigationInterEnv,
    mp_wrapper=MPWrapper_Crowd_Vel,
    max_episode_steps=100,
    kwargs={
        "n_crowd": 8,
        "width": 16,
        "height": 16,
        "interceptor_percentage": 2,
        "sequence_obs": True,
    }
)

register(
    id='fancy/CrowdNavigationInterSeqVel-v0',
    entry_point=CrowdNavigationInterEnv,
    mp_wrapper=MPWrapper_Crowd_Vel,
    max_episode_steps=100,
    kwargs={
        "n_crowd": 8,
        "width": 16,
        "height": 16,
        "interceptor_percentage": 2,
        "sequence_obs": True,
        "velocity_control": True,
    }
)

register(
    id='fancy/LongSimpleReacher-v0',
    entry_point=SimpleReacherEnv,
    mp_wrapper=MPWrapper_SimpleReacher,
    max_episode_steps=200,
    kwargs={
        "n_links": 5,
    }
)

# Viapoint Reacher
register(
    id='fancy/ViaPointReacher-v0',
    entry_point=ViaPointReacherEnv,
    mp_wrapper=MPWrapper_ViaPointReacher,
    max_episode_steps=200,
    kwargs={
        "n_links": 5,
        "allow_self_collision": False,
        "collision_penalty": 1000
    }
)

# Hole Reacher
register(
    id='fancy/HoleReacher-v0',
    entry_point=HoleReacherEnv,
    mp_wrapper=MPWrapper_HoleReacher,
    max_episode_steps=200,
    kwargs={
        "n_links": 5,
        "random_start": True,
        "allow_self_collision": False,
        "allow_wall_collision": False,
        "hole_width": None,
        "hole_depth": 1,
        "hole_x": None,
        "collision_penalty": 100,
    }
)

# Mujoco

# Mujoco Reacher
for dims in [5, 7]:
    register(
        id=f'fancy/Reacher{dims}d-v0',
        entry_point=ReacherEnv,
        mp_wrapper=MPWrapper_Reacher,
        max_episode_steps=MAX_EPISODE_STEPS_REACHER,
        kwargs={
            "n_links": dims,
        }
    )

    register(
        id=f'fancy/Reacher{dims}dSparse-v0',
        entry_point=ReacherEnv,
        mp_wrapper=MPWrapper_Reacher,
        max_episode_steps=MAX_EPISODE_STEPS_REACHER,
        kwargs={
            "sparse": True,
            'reward_weight': 200,
            "n_links": dims,
        }
    )


register(
    id='fancy/HopperJumpSparse-v0',
    entry_point='fancy_gym.envs.mujoco:HopperJumpEnv',
    mp_wrapper=mujoco.hopper_jump.MPWrapper,
    max_episode_steps=MAX_EPISODE_STEPS_HOPPERJUMP,
    kwargs={
        "sparse": True,
    }
)

register(
    id='fancy/HopperJump-v0',
    entry_point='fancy_gym.envs.mujoco:HopperJumpEnv',
    mp_wrapper=mujoco.hopper_jump.MPWrapper,
    max_episode_steps=MAX_EPISODE_STEPS_HOPPERJUMP,
    kwargs={
        "sparse": False,
        "healthy_reward": 1.0,
        "contact_weight": 0.0,
        "height_weight": 3.0,
    }
)

register(
    id='fancy/HopperJumpMarkov-v0',
    entry_point='fancy_gym.envs.mujoco:HopperJumpMarkovRew',
    mp_wrapper=mujoco.hopper_jump.MPWrapper,
    max_episode_steps=MAX_EPISODE_STEPS_HOPPERJUMP,
    kwargs={
        "sparse": False,
        "healthy_reward": 1.0,
        "contact_weight": 0.0,
        "height_weight": 3.0,
    }
)

# TODO: Add [MPs] later when finished (old TODO I moved here during refactor)
register(
    id='fancy/AntJump-v0',
    entry_point='fancy_gym.envs.mujoco:AntJumpEnv',
    max_episode_steps=MAX_EPISODE_STEPS_ANTJUMP,
    add_mp_types=[],
)

register(
    id='fancy/HalfCheetahJump-v0',
    entry_point='fancy_gym.envs.mujoco:HalfCheetahJumpEnv',
    max_episode_steps=MAX_EPISODE_STEPS_HALFCHEETAHJUMP,
    add_mp_types=[],
)

register(
    id='fancy/HopperJumpOnBox-v0',
    entry_point='fancy_gym.envs.mujoco:HopperJumpOnBoxEnv',
    max_episode_steps=MAX_EPISODE_STEPS_HOPPERJUMPONBOX,
    add_mp_types=[],
)

register(
    id='fancy/HopperThrow-v0',
    entry_point='fancy_gym.envs.mujoco:HopperThrowEnv',
    max_episode_steps=MAX_EPISODE_STEPS_HOPPERTHROW,
    add_mp_types=[],
)

register(
    id='fancy/HopperThrowInBasket-v0',
    entry_point='fancy_gym.envs.mujoco:HopperThrowInBasketEnv',
    max_episode_steps=MAX_EPISODE_STEPS_HOPPERTHROWINBASKET,
    add_mp_types=[],
)

register(
    id='fancy/Walker2DJump-v0',
    entry_point='fancy_gym.envs.mujoco:Walker2dJumpEnv',
    max_episode_steps=MAX_EPISODE_STEPS_WALKERJUMP,
    add_mp_types=[],
)

register(  # [MPDone
    id='fancy/BeerPong-v0',
    entry_point='fancy_gym.envs.mujoco:BeerPongEnv',
    mp_wrapper=MPWrapper_Beerpong,
    max_episode_steps=MAX_EPISODE_STEPS_BEERPONG,
    add_mp_types=['ProMP'],
)

# Here we use the same reward as in BeerPong-v0, but now consider after the release,
# only one time step, i.e. we simulate until the end of th episode
register(
    id='fancy/BeerPongStepBased-v0',
    entry_point='fancy_gym.envs.mujoco:BeerPongEnvStepBasedEpisodicReward',
    mp_wrapper=MPWrapper_Beerpong_FixedRelease,
    max_episode_steps=FIXED_RELEASE_STEP,
    add_mp_types=['ProMP'],
)

register(
    id='fancy/BeerPongFixedRelease-v0',
    entry_point='fancy_gym.envs.mujoco:BeerPongEnv',
    mp_wrapper=MPWrapper_Beerpong_FixedRelease,
    max_episode_steps=FIXED_RELEASE_STEP,
    add_mp_types=['ProMP'],
)

# Box pushing environments with different rewards
for reward_type in ["Dense", "TemporalSparse", "TemporalSpatialSparse"]:
    register(
        id='fancy/BoxPushing{}-v0'.format(reward_type),
        entry_point='fancy_gym.envs.mujoco:BoxPushing{}'.format(reward_type),
        mp_wrapper=mujoco.box_pushing.MPWrapper,
        max_episode_steps=MAX_EPISODE_STEPS_BOX_PUSHING,
    )
    register(
        id='fancy/BoxPushingRandomInit{}-v0'.format(reward_type),
        entry_point='fancy_gym.envs.mujoco:BoxPushing{}'.format(reward_type),
        mp_wrapper=mujoco.box_pushing.MPWrapper,
        max_episode_steps=MAX_EPISODE_STEPS_BOX_PUSHING,
        kwargs={"random_init": True}
    )

    upgrade(
        id='fancy/BoxPushing{}Replan-v0'.format(reward_type),
        base_id='fancy/BoxPushing{}-v0'.format(reward_type),
        mp_wrapper=mujoco.box_pushing.ReplanMPWrapper,
    )

# Table Tennis environments
for ctxt_dim in [2, 4]:
    register(
        id='fancy/TableTennis{}D-v0'.format(ctxt_dim),
        entry_point='fancy_gym.envs.mujoco:TableTennisEnv',
        mp_wrapper=MPWrapper_TableTennis,
        max_episode_steps=MAX_EPISODE_STEPS_TABLE_TENNIS,
        add_mp_types=['ProMP', 'ProDMP'],
        kwargs={
            "ctxt_dim": ctxt_dim,
            'frame_skip': 4,
        }
    )

    register(
        id='fancy/TableTennis{}DReplan-v0'.format(ctxt_dim),
        entry_point='fancy_gym.envs.mujoco:TableTennisEnv',
        mp_wrapper=MPWrapper_TableTennis,
        max_episode_steps=MAX_EPISODE_STEPS_TABLE_TENNIS,
        add_mp_types=['ProDMP'],
        kwargs={
            "ctxt_dim": ctxt_dim,
            'frame_skip': 4,
        }
    )

register(
    id='fancy/TableTennisWind-v0',
    entry_point='fancy_gym.envs.mujoco:TableTennisWind',
    mp_wrapper=MPWrapper_TableTennis_VelObs,
    add_mp_types=['ProMP', 'ProDMP'],
    max_episode_steps=MAX_EPISODE_STEPS_TABLE_TENNIS,
)

register(
    id='fancy/TableTennisWindReplan-v0',
    entry_point='fancy_gym.envs.mujoco:TableTennisWind',
    mp_wrapper=MPWrapper_TableTennis_VelObs_Replan,
    add_mp_types=['ProDMP'],
    max_episode_steps=MAX_EPISODE_STEPS_TABLE_TENNIS,
)

register(
    id='fancy/TableTennisGoalSwitching-v0',
    entry_point='fancy_gym.envs.mujoco:TableTennisGoalSwitching',
    mp_wrapper=MPWrapper_TableTennis,
    add_mp_types=['ProMP', 'ProDMP'],
    max_episode_steps=MAX_EPISODE_STEPS_TABLE_TENNIS,
    kwargs={
        'goal_switching_step': 99
    }
)

register(
    id='fancy/TableTennisGoalSwitchingReplan-v0',
    entry_point='fancy_gym.envs.mujoco:TableTennisGoalSwitching',
    mp_wrapper=MPWrapper_TableTennis_Replan,
    add_mp_types=['ProDMP'],
    max_episode_steps=MAX_EPISODE_STEPS_TABLE_TENNIS,
    kwargs={
        'goal_switching_step': 99
    }
)

register(
    id='fancy/TableTennisRndRobot-v0',
    entry_point='fancy_gym.envs.mujoco:TableTennisRandomInit',
    mp_wrapper=MPWrapper_TableTennis_Rnd,
    max_episode_steps=MAX_EPISODE_STEPS_TABLE_TENNIS,
    kwargs={
        'random_pos_scale': 0.1,
        'random_vel_scale': 0.0,
    }
)

register(
    id='fancy/TableTennisMarkov-v0',
    mp_wrapper=MPWrapper_TableTennis,
    entry_point='fancy_gym.envs.mujoco:TableTennisMarkov',
    max_episode_steps=MAX_EPISODE_STEPS_TABLE_TENNIS_MARKOV_VER,
    kwargs={
    }
)

register(
    id='fancy/TableTennisRndRobotMarkov-v0',
    mp_wrapper=MPWrapper_TableTennis_Rnd,
    entry_point='fancy_gym.envs.mujoco:TableTennisMarkov',
    max_episode_steps=MAX_EPISODE_STEPS_TABLE_TENNIS_MARKOV_VER,
    kwargs={
        'random_pos_scale': 0.1,
        'random_vel_scale': 0.0,
    }
)

# Air Hockey environments
for env_mode in ["7dof-hit", "7dof-defend", "3dof-hit", "3dof-defend", "7dof-hit-airhockit2023", "7dof-defend-airhockit2023"]:
    register(
        id=f'fancy/AirHockey-{env_mode}-v0',
        entry_point='fancy_gym.envs.mujoco:AirHockeyEnv',
        max_episode_steps=500,
        add_mp_types=[],
        kwargs={
            'env_mode': env_mode
        }
    )

register(
    id=f'fancy/AirHockey-tournament-v0',
    entry_point='fancy_gym.envs.mujoco:AirHockeyEnv',
    max_episode_steps=15000,
    add_mp_types=[],
    kwargs={
        'env_mode': 'tournament'
    }
)
